"""LC-MAPPO training.

  python -m clause.train                       # writes runs/clause/
  python -m clause.train --iters 100 --out runs/short --env-sp-w 0.5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .config import ROOT, EnvConfig, TrainConfig
from .data import Resources, TokenCounter
from .env import AGENTS, ClauseEnv
from .models import AgentNet, CentralCritic

OBS_KEYS = ("q", "glob", "path_seq", "path_len", "cand_rel", "cand_seq", "cand_scal",
            "cand_price", "cand_mask", "spec_mask")


def to_t(o, rows=None):
    if rows is None:
        return {k: torch.from_numpy(o[k]) for k in OBS_KEYS}
    return {k: torch.from_numpy(o[k][rows]) for k in OBS_KEYS}


def build_env(res, ecfg):
    return ClauseEnv(res, ecfg, TokenCounter())


# ----------------------------------------------------------------------------- rollout

def rollout(env, nets, items, qvecs, lam, training, greedy, rng=None, record=False):
    env.reset(items, qvecs, lam, training, rng)
    for i, e in enumerate(env.eps):
        e.uid = i
    rounds = []
    while True:
        act = env.active()
        if not act:
            break
        ro = {}
        for ag in AGENTS:
            o = env.observe(ag, act)
            with torch.no_grad():
                logits, _ = nets[ag](to_t(o))
            a = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
            a = torch.where(torch.from_numpy(o["acting"]), a, torch.full_like(a, -1))   # no-op rows
            env.apply(ag, act, a.numpy())
            if record:
                lp = F.log_softmax(logits, -1)
                o["act"] = a.numpy()
                o["logp"] = lp.gather(1, a.clamp(min=0)[:, None]).squeeze(1).numpy()
                o["probs"] = lp.exp().numpy()
            ro[ag] = o
        outs = env.end_round(act)
        if record:
            ro["ep"] = np.array([e.uid for e in act])
            ro["out"] = np.array(outs, dtype=np.float32)
            rounds.append(ro)
    return rounds


def flatten(rounds):
    data = {ag: {k: np.concatenate([r[ag][k] for r in rounds]) for k in rounds[0][ag]} for ag in AGENTS}
    data["ep"] = np.concatenate([r["ep"] for r in rounds])
    data["out"] = np.concatenate([r["out"] for r in rounds])
    n = len(data["ep"])
    nxt = np.full(n, -1, dtype=np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        ep = data["ep"][i]
        nxt[i] = last.get(ep, -1)
        last[ep] = i
    data["next"] = nxt
    return data


def run_eval(env, nets, items, qvecs, lam, bs=300, caps=None):
    """Greedy decentralised execution; returns the finished episodes. `caps` switches on cap mode."""
    eps = []
    for s in range(0, len(items), bs):
        env.reset(items[s:s + bs], qvecs[s:s + bs], lam, False)
        if caps:
            env.set_caps(**caps)
        greedy_rollout(env, nets)
        eps += env.eps
    return eps


def greedy_rollout(env, nets):
    """Run the episodes of the current batch to the end, every agent taking its arg-max action."""
    while True:
        act = env.active()
        if not act:
            break
        for ag in AGENTS:
            o = env.observe(ag, act)
            with torch.no_grad():
                logits, _ = nets[ag](to_t(o))
            env.apply(ag, act, logits.argmax(-1).numpy())
        env.end_round(act)


def summarize(stats, beta=(TrainConfig.beta_edge, TrainConfig.beta_lat)):
    """Means over all questions and per question type; `feasible` = within the edge and step targets."""
    out = {}
    for hop in [None] + sorted({x["hop"] for x in stats}, key=str):
        s = [x for x in stats if hop is None or x["hop"] == hop]
        key = "all" if hop is None else str(hop)
        out[key] = {k: float(np.mean([x[k] for x in s])) for k in
                    ("proxy", "ans_recall", "sp_recall", "n_text", "n_kg", "C_edge", "C_lat", "C_tok", "n_sel",
                     "edge_frac", "lat_frac", "tok_frac", "truncated")}
        out[key]["feasible"] = float(np.mean([x["edge_frac"] <= beta[0] and x["lat_frac"] <= beta[1] for x in s]))
        out[key]["n"] = len(s)
    return out


# ----------------------------------------------------------------------------- LC-MAPPO

class Trainer:
    def __init__(self, res, ecfg, tcfg, out_dir):
        self.res, self.ecfg, self.tcfg, self.out = res, ecfg, tcfg, out_dir
        torch.manual_seed(tcfg.seed)
        self.rng = np.random.default_rng(tcfg.seed)
        H = ecfg.hop_cap
        self.nets = {a: AgentNet(a, H, tcfg.hidden) for a in AGENTS}
        self.critic = CentralCritic(H, 4, tcfg.hidden)       # heads: task, edge, lat, tok
        self.opt = {a: torch.optim.Adam(n.parameters(), lr=tcfg.lr_actor) for a, n in self.nets.items()}
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=tcfg.lr_critic)
        self.lam = np.zeros(3, dtype=np.float32)
        self.beta = np.array([tcfg.beta_edge, tcfg.beta_lat, tcfg.beta_tok], dtype=np.float32)
        self.integral = np.zeros(3, dtype=np.float32)
        self.env = build_env(res, ecfg)
        self.train_items = res.questions["train"]
        self.train_q = res.q_emb["train"]
        self.dev_items = res.questions["dev"]
        self.dev_q = res.q_emb["dev"]

    def returns(self, data, R):
        """Per-head returns-to-go from every round (episodes are short; Monte-Carlo, discount gamma)."""
        G = np.zeros_like(R)
        done = data["out"][:, 4] > 0.5
        for i in range(len(R) - 1, -1, -1):
            j = data["next"][i]
            G[i] = R[i] if (done[i] or j < 0) else R[i] + self.tcfg.gamma * G[j]
        return G

    def fit_critic(self, data, G):
        """Regress Q_i^h(s_i, a_i) of every agent on the per-head returns."""
        tc = self.tcfg
        losses = []
        for _ in range(tc.critic_epochs):
            for ag in AGENTS:
                perm = self.rng.permutation(np.flatnonzero(data[ag]["act"] >= 0))
                for s in range(0, len(perm), tc.minibatch):
                    rows = perm[s:s + tc.minibatch]
                    pred = self.critic.q_taken(ag, to_t(data[ag], rows), torch.from_numpy(data[ag]["act"][rows]))
                    loss = F.mse_loss(pred, torch.from_numpy(G[rows]))
                    self.copt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
                    self.copt.step()
                    losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    def advantages(self, data, G, ag):
        """Per-head counterfactual advantage A_i^h = G^h - sum_a' pi_i(a' | s_i) Q_i^h(s_i, a') of agent `ag`."""
        n = len(G)
        A = np.zeros_like(G)
        with torch.no_grad():
            for s in range(0, n, 2048):
                rows = np.arange(s, min(s + 2048, n))
                qa = self.critic.q_all(ag, to_t(data[ag], rows))                    # B, A, heads
                p = torch.from_numpy(data[ag]["probs"][rows]).unsqueeze(-1)
                A[rows] = G[rows] - (p * qa).sum(1).numpy()
        return A

    def update(self, data):
        tc = self.tcfg
        out = data["out"]
        R = np.concatenate([out[:, :1], out[:, 1:4]], 1).astype(np.float32)   # task reward + normalised costs
        G = self.returns(data, R)
        closs = self.fit_critic(data, G)
        price = np.concatenate([[1.0], -self.lam])
        info = {"critic_loss": closs}
        for ag, net in self.nets.items():
            # Lagrangian advantage: A_task - sum_k lambda_k A_k
            A = (self.advantages(data, G, ag) * price).sum(1)
            nvalid = data[ag]["cand_mask"].sum(1) + data[ag]["spec_mask"].sum(1)
            m = (nvalid > 1) & (data[ag]["act"] >= 0)
            if m.sum() < 2:
                continue
            A = (A - A[m].mean()) / (A[m].std() + 1e-8)
            rows_all = np.flatnonzero(m)
            pl, ent = [], []
            for _ in range(tc.ppo_epochs):
                perm = self.rng.permutation(rows_all)
                for s in range(0, len(perm), tc.minibatch):
                    rows = perm[s:s + tc.minibatch]
                    logits, mask = net(to_t(data[ag], rows))
                    lp = F.log_softmax(logits, -1)
                    act = torch.from_numpy(data[ag]["act"][rows])
                    new = lp.gather(1, act[:, None]).squeeze(1)
                    ratio = torch.exp(new - torch.from_numpy(data[ag]["logp"][rows]))
                    adv = torch.from_numpy(A[rows].astype(np.float32))
                    surr = torch.min(ratio * adv, ratio.clamp(1 - tc.clip, 1 + tc.clip) * adv)
                    p = lp.exp()
                    entropy = -(p * lp.masked_fill(~mask, 0.0)).sum(-1)
                    loss = -surr.mean() - tc.ent_coef * entropy.mean()
                    self.opt[ag].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    self.opt[ag].step()
                    pl.append(loss.item())
                    ent.append(entropy.mean().item())
            info[f"{ag}_loss"] = float(np.mean(pl))
            info[f"{ag}_ent"] = float(np.mean(ent))
        return info

    def dual_update(self, cost_means):
        """lambda_k = [Kp e_k + Ki I_k]_+ with I_k = [I_k + e_k]_+ and e_k = E[C_k] / cap_k - beta_k."""
        tc = self.tcfg
        err = (cost_means - self.beta).astype(np.float32)
        self.integral = np.maximum(0.0, self.integral + err)
        self.lam = np.maximum(0.0, tc.dual_kp * err + tc.dual_ki * self.integral).astype(np.float32)

    def save(self, name, extra=None):
        ck = {"nets": {a: n.state_dict() for a, n in self.nets.items()}, "critic": self.critic.state_dict(),
              "lam": self.lam.tolist(), "tcfg": self.tcfg.to_dict(), "ecfg": self.ecfg.__dict__, "extra": extra or {}}
        torch.save(ck, self.out / name)

    def train(self):
        tc = self.tcfg
        log = open(self.out / "log.jsonl", "w", encoding="utf-8")
        best = -1.0
        n_train = len(self.train_items)
        t0 = time.time()
        for it in range(1, tc.iters + 1):
            idx = self.rng.choice(n_train, tc.episodes_per_iter, replace=False)
            items = [self.train_items[i] for i in idx]
            rounds = rollout(self.env, self.nets, items, self.train_q[idx], self.lam,
                             training=True, greedy=False, rng=self.rng, record=True)
            stats = [self.env.episode_stats(e) for e in self.env.eps]
            data = flatten(rounds)
            info = self.update(data)
            costs = np.array([np.mean([s[k] for s in stats]) for k in ("edge_frac", "lat_frac", "tok_frac")])
            self.dual_update(costs)
            feas = np.mean([s["edge_frac"] <= self.beta[0] and s["lat_frac"] <= self.beta[1]
                            and s["tok_frac"] <= self.beta[2] for s in stats])
            rec = {"iter": it, "time": round(time.time() - t0, 1),
                   "proxy": float(np.mean([s["proxy"] for s in stats])),
                   "edge_frac": costs[0], "lat_frac": costs[1], "tok_frac": costs[2], "feasible": float(feas),
                   "lam": self.lam.tolist(), "steps": len(data["ep"]), **info}
            if it % tc.eval_every == 0 or it == tc.iters:
                eps = run_eval(self.env, self.nets, self.dev_items, self.dev_q, self.lam)
                dev = summarize([self.env.episode_stats(e) for e in eps])
                rec["dev"] = {k: round(v["proxy"], 4) for k, v in dev.items()}
                rec["dev_cost"] = {k: round(dev["all"][k], 3) for k in ("edge_frac", "lat_frac", "tok_frac")}
                if dev["all"]["proxy"] > best:
                    best = dev["all"]["proxy"]
                    self.save("best.pt", {"iter": it, "dev": dev})
                self.save("last.pt", {"iter": it, "dev": dev})
            log.write(json.dumps(rec, default=float) + "\n")
            log.flush()
            if it % 5 == 0 or "dev" in rec:
                msg = (f"it {it} {rec['time']}s proxy {rec['proxy']:.3f} feas {feas:.2f} "
                       f"cost e/l/t {costs[0]:.2f}/{costs[1]:.2f}/{costs[2]:.2f} lam {np.round(self.lam, 3)} "
                       f"closs {info['critic_loss']:.4f}")
                if "dev" in rec:
                    msg += f" | dev proxy {rec['dev']}"
                print(msg, flush=True)
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/clause")
    for k, v in TrainConfig().to_dict().items():           # training settings: --<name>
        typ = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(v, bool) else type(v)
        ap.add_argument(f"--{k.replace('_', '-')}", type=typ, default=v)
    for k, v in EnvConfig().__dict__.items():           # environment settings: --env-<name>
        typ = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(v, bool) else type(v)
        ap.add_argument(f"--env-{k.replace('_', '-')}", type=typ, default=v)
    a = ap.parse_args()
    tcfg = TrainConfig(**{k: getattr(a, k) for k in TrainConfig().to_dict()})
    ecfg = EnvConfig(**{k: getattr(a, f"env_{k}") for k in EnvConfig().__dict__})
    torch.set_num_threads(tcfg.threads)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    res = Resources()
    res.load_questions("train")
    res.load_questions("dev")
    json.dump({"train": tcfg.to_dict(), "env": ecfg.__dict__}, open(out / "config.json", "w"), indent=1)
    Trainer(res, ecfg, tcfg, out).train()


if __name__ == "__main__":
    main()
