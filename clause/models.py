"""Light-weight neural scorers (actors) and the multi-head centralised critic of LC-MAPPO."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_REL
from .env import AGENTS, GLOB_DIM, N_SPECIAL, PAD, SCAL


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class SeqEnc(nn.Module):
    """Position-aware bag of relations: sum_p E[p, r_p]."""

    def __init__(self, H, h):
        super().__init__()
        self.H = H
        self.emb = nn.Embedding(H * (N_REL + 1), h)

    def forward(self, seq):
        pos = torch.arange(self.H, device=seq.device)
        e = self.emb(pos * (N_REL + 1) + seq)
        return (e * (seq != PAD).unsqueeze(-1).float()).sum(-2)


# indices of the frozen-encoder similarity features in each agent's candidate features (see env.py):
# (relation-question cosine, entity-question cosine)
SIM_FEATS = {"arch": (2, 3), "nav": (0, 1), "cur": (9, 4)}


class AgentNet(nn.Module):
    """Scores every symbolic candidate action + the agent's special actions (STOP / BACKTRACK).

    logit(a) = w^T phi(a) + f_theta(a | o) - softplus(kappa) * lambda_k * c_k(a)
    w starts at (prior, prior / 2) on the two similarity features and at zero elsewhere; f_theta is an MLP over
    relation-sequence / question interactions; the price term lets lambda steer the policy at inference.
    """

    def __init__(self, agent, H, h=64, d_q=384, prior=1.0):
        super().__init__()
        self.agent, self.H = agent, H
        n_scal = SCAL[agent]
        self.q_proj = nn.Sequential(nn.Linear(d_q, h), nn.Tanh())
        self.rel = nn.Embedding(N_REL + 1, h)
        self.seq = SeqEnc(H, h)
        self.cand = mlp(3 * h + n_scal + GLOB_DIM, 128, 1)
        self.fused = nn.Linear(n_scal, 1, bias=False)
        nn.init.zeros_(self.fused.weight)
        with torch.no_grad():
            i_rel, i_ent = SIM_FEATS[agent]
            self.fused.weight[0, i_rel] = prior
            self.fused.weight[0, i_ent] = prior / 2
        self.spec = mlp(3 * h + GLOB_DIM + H + 1, 128, N_SPECIAL[agent])
        self.price_raw = nn.Parameter(torch.zeros(1))

    def forward(self, o):
        hq = self.q_proj(o["q"])
        hs = self.seq(o["cand_seq"])
        hr = self.rel(o["cand_rel"])
        K = hs.shape[1]
        g = o["glob"].unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([hq[:, None] * hs, hq[:, None] * hr, hs, o["cand_scal"], g], -1)
        cl = (self.fused(o["cand_scal"]).squeeze(-1) + self.cand(x).squeeze(-1)
              - F.softplus(self.price_raw) * o["cand_price"])
        hp = self.seq(o["path_seq"])
        lenoh = F.one_hot(o["path_len"].clamp(max=self.H), self.H + 1).float()
        sl = self.spec(torch.cat([hq, hp, hq * hp, o["glob"], lenoh], -1))
        mask = torch.cat([o["cand_mask"], o["spec_mask"]], -1)
        return torch.cat([cl, sl], -1).masked_fill(~mask, -1e9), mask


class CentralCritic(nn.Module):
    """Multi-head centralised critic (heads: task, edge, lat, tok), evaluated at each agent's decision state.

    The agents act in sequence within a round (edit -> traverse -> curate), so agent i's counterfactual
    baseline marginalises its own action at the state where it decides:
        b_i^h(s_i) = sum_a' pi_i(a' | s_i) Q_i^h(s_i, a').
    """

    def __init__(self, H, n_heads=4, h=64, d_q=384):
        super().__init__()
        self.H, self.n_heads = H, n_heads
        self.q_proj = nn.Sequential(nn.Linear(d_q, h), nn.Tanh())
        self.rel = nn.Embedding(N_REL + 1, h)
        self.seq = SeqEnc(H, h)
        self.state = nn.ModuleDict({a: mlp(2 * h + GLOB_DIM + H + 1, 128, 128) for a in AGENTS})
        self.act = nn.ModuleDict({a: mlp(3 * h + SCAL[a], 64, 64) for a in AGENTS})
        self.spec = nn.ModuleDict({a: nn.Embedding(N_SPECIAL[a], 64) for a in AGENTS})
        self.util = nn.ModuleDict({a: mlp(128 + 64, 128, n_heads) for a in AGENTS})

    def state_enc(self, agent, o):
        hq = self.q_proj(o["q"])
        hp = self.seq(o["path_seq"])
        lenoh = F.one_hot(o["path_len"].clamp(max=self.H), self.H + 1).float()
        return F.relu(self.state[agent](torch.cat([hq, hp, o["glob"], lenoh], -1))), hq

    def _cand_enc(self, agent, hq, seq, rel, scal):
        hs = self.seq(seq)
        hr = self.rel(rel)
        return self.act[agent](torch.cat([hq.unsqueeze(-2) * hs, hq.unsqueeze(-2) * hr, hs, scal], -1))

    def q_all(self, agent, o):
        """Q_i^h(s_i, a') for every action a' of agent i."""
        hstate, hq = self.state_enc(agent, o)
        a = self._cand_enc(agent, hq, o["cand_seq"], o["cand_rel"], o["cand_scal"])
        sp = self.spec[agent].weight.unsqueeze(0).expand(a.shape[0], -1, -1)
        acts = torch.cat([a, sp], 1)
        hx = hstate.unsqueeze(1).expand(-1, acts.shape[1], -1)
        return self.util[agent](torch.cat([hx, acts], -1))           # B, A_i, heads

    def q_taken(self, agent, o, act):
        """Q_i^h(s_i, a_i) for the executed action only (cheap path for critic regression)."""
        hstate, hq = self.state_enc(agent, o)
        B, K = o["cand_mask"].shape
        act = act.clamp(min=0)
        ci = act.clamp(max=K - 1)
        ar = torch.arange(B)
        c = self._cand_enc(agent, hq, o["cand_seq"][ar, ci][:, None], o["cand_rel"][ar, ci][:, None],
                           o["cand_scal"][ar, ci][:, None]).squeeze(1)
        s = self.spec[agent]((act - K).clamp(min=0))
        enc = torch.where((act < K).unsqueeze(-1), c, s)
        return self.util[agent](torch.cat([hstate, enc], -1))         # B, heads
