"""Evaluate a trained checkpoint with greedy decentralised execution.

  python -m clause.evaluate                      # runs/clause/best.pt on the test split: evidence statistics
  python -m clause.evaluate --llm                # + EM / F1 of the LLM reader (LLM_* settings in .env)
  python -m clause.evaluate --caps 0,8,256       # cap mode: tighter step / token budgets at inference (0 = default)
  python -m clause.evaluate --lam 0,1.0,0        # price mode: a price on interaction steps

Reader answers are cached by (model, prompt, question, evidence) in results/llm_cache/. Writes
results/<name>_<split>.json (and _answers.jsonl with --llm) and the curated evidence to results/evidence/.
"""
import argparse
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from .config import RESULTS_DIR, ROOT, EnvConfig
from .data import Resources
from .env import AGENTS
from .models import AgentNet
from .reader import PROMPT, LLMReader, f1, lenient_em, strict_em
from .train import build_env, run_eval, summarize


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ecfg = EnvConfig(**ck["ecfg"])
    nets = {}
    for a in AGENTS:
        nets[a] = AgentNet(a, ecfg.hop_cap, ck["tcfg"]["hidden"])
        nets[a].load_state_dict(ck["nets"][a])
        nets[a].eval()
    return nets, np.array(ck["lam"], dtype=np.float32), ecfg, ck


def read_all(reader, rows, workers):
    """LLM answers for rows of {question, evidence}, through a content-addressed cache."""
    cache_path = RESULTS_DIR / "llm_cache" / (re.sub(r"[^\w.-]+", "_", reader.model) + ".jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[r["key"]] = r["pred"]

    def key(r):
        return hashlib.sha1(json.dumps([reader.model, PROMPT, r["question"], r["evidence"]]).encode()).hexdigest()

    keys = [key(r) for r in rows]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        print(f"querying {reader.model} for {len(todo)} of {len(rows)} questions")
    lock = threading.Lock()
    with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(reader, rows[i]["question"], rows[i]["evidence"]): i for i in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            i = futs[fu]
            pred = fu.result()
            with lock:
                cache[keys[i]] = pred
                f.write(json.dumps({"key": keys[i], "pred": pred}, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    return [cache[k] for k in keys]


def evaluate(ckpt, name, split="test", lam=None, caps=None, llm=False, workers=32):
    torch.set_num_threads(4)
    nets, ck_lam, ecfg, ck = load(ckpt)
    lam = ck_lam if lam is None else np.asarray(lam, dtype=np.float32)
    res = Resources()
    items = res.load_questions(split)
    env = build_env(res, ecfg)
    eps = run_eval(env, nets, items, res.q_emb[split], lam, caps=caps)
    stats = [env.episode_stats(e) for e in eps]
    rows = [{"id": e.item["id"], "type": e.item["hop"], "question": e.item["text"], "answer": e.item["answers"][0],
             "evidence": [e.pool[i].text for i in e.selected]} for e in eps]
    (RESULTS_DIR / "evidence").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "evidence" / f"{name}_{split}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    summary = summarize(stats, (ck["tcfg"]["beta_edge"], ck["tcfg"]["beta_lat"]))
    result = {"ckpt": _rel(ckpt), "split": split, "train_iter": ck["extra"].get("iter"), "lam": lam.tolist(),
              "caps": caps, "summary": summary}
    if llm:
        reader = LLMReader()
        preds = read_all(reader, rows, workers)
        answers = [{k: r[k] for k in ("id", "type", "question", "answer")} | {
            "pred": p, "em": strict_em(p, r["answer"]), "lenient": lenient_em(p, r["answer"]), "f1": f1(p, r["answer"])}
            for r, p in zip(rows, preds)]
        with open(RESULTS_DIR / f"{name}_{split}_answers.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(a, ensure_ascii=False) + "\n" for a in answers)
        for k, s in summary.items():
            sel = [a for a in answers if k == "all" or a["type"] == k]
            s.update({m: float(np.mean([a[m] for a in sel])) for m in ("em", "lenient", "f1")})
        result["reader"] = {"model": reader.model, "base_url": reader.base_url}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(RESULTS_DIR / f"{name}_{split}.json", "w"), indent=1)
    for k, s in summary.items():
        pct = "  ".join(f"{m} {100 * s[m]:.1f}" for m in ("em", "lenient", "f1", "ans_recall", "sp_recall") if m in s)
        print(f"{k:10} n {s['n']:4}  {pct}  tokens {s['C_tok']:.0f}  paragraphs / KG units {s['n_text']:.1f} / "
              f"{s['n_kg']:.1f}  budget use edge / steps / tokens {s['edge_frac']:.2f} / {s['lat_frac']:.2f} / "
              f"{s['tok_frac']:.2f}")
    return result


def _rel(path):
    """Repo-relative path, so results do not depend on the checkout location."""
    p = Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(path)


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/clause/best.pt")
    ap.add_argument("--name", default="clause")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--llm", action="store_true", help="also score the answers of the LLM reader")
    ap.add_argument("--workers", type=int, default=32, help="concurrent reader requests")
    ap.add_argument("--lam", default=None, help="price mode: 'edge,lat,tok' prices instead of the trained duals")
    ap.add_argument("--caps", default=None, help="cap mode: 'edge,lat,tok' budgets (0 keeps the default)")
    a = ap.parse_args()
    caps = None
    if a.caps:
        e, l, t = (int(x) for x in a.caps.split(","))
        caps = {"edge": e or None, "lat": l or None, "tok": t or None}
    lam = [float(x) for x in a.lam.split(",")] if a.lam else None
    ck = Path(a.ckpt) if Path(a.ckpt).is_absolute() else ROOT / a.ckpt
    evaluate(ck, a.name, a.split, lam, caps, a.llm, a.workers)
