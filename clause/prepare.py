"""Build the local KG, the question splits and the embedding caches for HotpotQA (distractor) in data/processed/.

  python -m clause.prepare        # after downloading HotpotQA and Wikidata5M (see README)

Paragraph titles are linked to Wikidata5M entities through its alias table. The local KG is the 1-hop
neighbourhood of the linked paragraph entities (hub entities only keep edges to other linked entities), plus
`mentions` edges from a paragraph entity to the entities whose title its text mentions. Paragraph entities are
named after their Wikipedia title.

Other datasets plug in through the same files, so the environment, training and evaluation run unchanged:
  relations.json        [{"label", "fwd", "inv"}, ...]; the last entry is the text pseudo-relation of paragraph units
  kg_names.txt          one entity name per line; kg_triples.npy: int64 (head, relation, tail) rows
  ent_emb.npy           gte-small embedding per entity; rel_emb.npy: forward then inverse relation verbalisations
  q_<split>.jsonl       one question per line: text, answers, anchors (entity ids), gold (answer entity ids),
                        titles / title_ents / sents (paragraphs, their entity ids and sentences), support
                        (supporting-paragraph entity ids), hop (question type, used to group the results)
  q_<split>_emb.npy     question embeddings; q_<split>_sent_emb.npy: sentence embeddings (rows from sent_off)
A KG-only benchmark such as MetaQA leaves titles / sents / support empty and trains with --env-sp-w 0
--env-text-fallback 0; the reward is then whether the curated facts contain the gold answer.
"""
import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .config import CACHE_DIR, HOTPOT_DIR, WD5M_DIR

HUB_DEGREE = 2000          # a seed with more neighbours only keeps edges to other seeds
MENTION = {"pid": "MENTION", "label": "mentions", "fwd": "article mentions", "inv": "is mentioned in article"}
TEXT = {"pid": "TEXT", "label": "passage", "fwd": "text passage about", "inv": "text passage about"}


def _norm_title(t):
    return re.sub(r"\s*\(.*\)$", "", t).strip()


def load_wikidata5m():
    label, display, alias = {}, {}, defaultdict(list)
    with open(WD5M_DIR / "wikidata5m_entity.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            names = parts[1:] or parts[:1]
            label[parts[0]] = names[0]          # first alias: used for exact-title linking
            # display name: the first alias that is not all lower case (several entries list "foo bar" first)
            display[parts[0]] = next((a for a in names if a != a.lower()), names[0])
            for a in set(p.lower() for p in parts[1:]):
                alias[a].append(parts[0])
    rel_label = {}
    with open(WD5M_DIR / "wikidata5m_relation.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rel_label[parts[0]] = parts[1] if len(parts) > 1 else parts[0]
    tr = pd.read_csv(WD5M_DIR / "wikidata5m_all_triplet.txt.gz", sep="\t", header=None, names=["h", "r", "t"],
                     dtype=str, quoting=3, engine="c")
    print(f"Wikidata5M: {len(label):,} entities with aliases, {len(tr):,} triples")
    return label, display, alias, rel_label, tr


def read_split(path, n, seed, levels=None, exclude=()):
    d = pd.read_parquet(path)
    if levels:
        d = d[d.level.isin(levels)]
    d = d[~d.id.isin(set(exclude))]
    return d.sample(n=min(n, len(d)), random_state=seed) if n else d


def prepare(n_train, n_dev, n_test, seed=0):
    from .data import TokenCounter, encode
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    label, display, alias, rel_label, tr = load_wikidata5m()
    codes, uniq = pd.factorize(pd.concat([tr.h, tr.t], ignore_index=True))
    h_all, t_all = codes[:len(tr)].astype(np.int64), codes[len(tr):].astype(np.int64)
    r_codes, r_uniq = pd.factorize(tr.r)
    q2g = {q: i for i, q in enumerate(uniq)}
    n_glob = len(uniq)
    deg = np.bincount(np.r_[h_all, t_all], minlength=n_glob)

    def link(title):
        """Wikipedia title -> global node (exact label, then alias, then title without '(film)' etc.)."""
        for key in (title, _norm_title(title)):
            cands = [q for q in alias.get(key.lower(), []) if q in q2g]
            if cands:
                exact = [q for q in cands if label.get(q) == title]
                return q2g[max(exact or cands, key=lambda q: deg[q2g[q]])]
        return -1

    def link_answer(ans):
        return [q2g[q] for q in alias.get(ans.lower().strip(), []) if q in q2g]

    # train: medium + hard questions of the first train shard; test and dev: disjoint samples of the official dev set
    splits = {
        "train": read_split(HOTPOT_DIR / "train-00000.parquet", n_train, seed, levels=("medium", "hard")),
        "test": read_split(HOTPOT_DIR / "validation.parquet", n_test, seed),
    }
    splits["dev"] = read_split(HOTPOT_DIR / "validation.parquet", n_dev, seed + 1, exclude=splits["test"].id)

    raw = {}
    seeds = set()
    for split, d in splits.items():
        rows = []
        for i, r in enumerate(d.itertuples(index=False)):
            titles = [str(t) for t in r.context["title"]]
            sents = [list(map(str, s)) for s in r.context["sentences"]]
            g_titles = [link(t) for t in titles]
            seeds.update(g for g in g_titles if g >= 0)
            rows.append({"split": split, "idx": i, "id": r.id, "hop": r.type, "level": r.level,
                         "text": r.question, "answers": [r.answer], "g_titles": g_titles, "titles": titles,
                         "paras": sents, "sp_titles": sorted(set(map(str, r.supporting_facts["title"]))),
                         "g_answer": link_answer(r.answer)})
        raw[split] = rows
    print(f"linked paragraph entities: {len(seeds):,}")

    # local KG: seeds + their neighbours (hubs keep only seed-seed edges), induced edges
    seed_mask = np.zeros(n_glob, dtype=bool)
    seed_mask[list(seeds)] = True
    small_seed = seed_mask & (deg <= HUB_DEGREE)
    keep = seed_mask.copy()
    keep[h_all[small_seed[t_all]]] = True
    keep[t_all[small_seed[h_all]]] = True
    e_mask = keep[h_all] & keep[t_all] & (h_all != t_all)
    nodes = np.flatnonzero(keep)
    g2l = np.full(n_glob, -1, dtype=np.int64)
    g2l[nodes] = np.arange(len(nodes))
    used_rels = np.unique(r_codes[e_mask])
    r2l = np.full(len(r_uniq), -1, dtype=np.int64)
    r2l[used_rels] = np.arange(len(used_rels))
    triples = np.unique(np.stack([g2l[h_all[e_mask]], r2l[r_codes[e_mask]], g2l[t_all[e_mask]]], 1), axis=0)
    names = [display.get(uniq[g], uniq[g]).replace("\n", " ") for g in nodes]
    rels = []
    for j in used_rels:
        pid = r_uniq[j]
        lab = rel_label.get(pid, pid)
        inv = f"has {lab[:-3]}" if lab.endswith(" of") else f"is {lab} of"
        rels.append({"pid": pid, "label": lab, "fwd": lab, "inv": inv})
    print(f"Wikidata neighbourhood: {len(nodes):,} nodes, {len(triples):,} triples, {len(rels)} relations")
    ent_emb = encode(names)

    # questions: paragraph sentences (with token counts), anchors, embeddings
    tok = TokenCounter()
    items = {}
    for split, rows in raw.items():
        items[split], sent_texts = [], []
        for r in rows:
            loc = [int(g2l[g]) if g >= 0 else -1 for g in r["g_titles"]]
            sents = []
            for p, (title, para) in enumerate(zip(r["titles"], r["paras"])):
                for s, text in enumerate(para):
                    text = text.strip()
                    if text:
                        unit = f"{title}: {text}"
                        sents.append([p, s, unit, tok(unit)])
            item = {k: r[k] for k in ("split", "idx", "id", "hop", "level", "text", "answers", "titles")}
            item.update({
                "title_ents": loc,
                "gold": sorted({int(g2l[g]) for g in r["g_answer"] if g2l[g] >= 0}),
                "support": sorted({loc[r["titles"].index(t)] for t in r["sp_titles"]
                                   if t in r["titles"] and loc[r["titles"].index(t)] >= 0}),
                "sents": sents,
                "sent_off": len(sent_texts),
            })
            sent_texts += [s[2] for s in sents]
            items[split].append(item)
        q_emb = encode([it["text"] for it in items[split]])
        s_emb = encode(sent_texts)
        for it, qv in zip(items[split], q_emb):
            anchor(it, qv, s_emb)
        np.save(CACHE_DIR / f"q_{split}_emb.npy", q_emb)
        np.save(CACHE_DIR / f"q_{split}_sent_emb.npy", s_emb.astype(np.float16))
        modes = Counter(it["anchor_mode"] for it in items[split])
        print(f"{split}: {len(items[split])} questions, {len(sent_texts)} sentences, anchoring {dict(modes)}, "
              f"answer entity in KG {np.mean([bool(it['gold']) for it in items[split]]):.1%}")

    # mentions edges between the paragraph entities of each question
    rels += [MENTION, TEXT]
    r_m = len(rels) - 2
    new = set()
    for split in ("train", "dev", "test"):
        for it in items[split]:
            ents = it["title_ents"]
            for p, q in _mentions(it):
                if ents[p] >= 0 and ents[q] >= 0 and ents[p] != ents[q]:
                    new.add((ents[p], r_m, ents[q]))
    triples = np.unique(np.concatenate([triples, np.array(sorted(new), dtype=np.int64)]), axis=0)
    print(f"mentions: {len(new):,} triples; local KG: {len(triples):,} triples, {len(rels) - 1} relations")

    # paragraph entities named after their (most frequent) Wikipedia title
    titles = defaultdict(Counter)
    for split in ("train", "dev", "test"):
        for it in items[split]:
            for t, x in zip(it["titles"], it["title_ents"]):
                if x >= 0:
                    titles[x][t] += 1
    renamed = sorted(x for x in titles if titles[x].most_common(1)[0][0] != names[x])
    for x in renamed:
        names[x] = titles[x].most_common(1)[0][0].replace("\n", " ")
    ent_emb[renamed] = encode([names[x] for x in renamed])
    print(f"renamed {len(renamed):,} paragraph entities")

    json.dump(rels, open(CACHE_DIR / "relations.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    with open(CACHE_DIR / "kg_names.txt", "w", encoding="utf-8") as f:
        f.writelines(n + "\n" for n in names)
    np.save(CACHE_DIR / "kg_triples.npy", triples)
    np.save(CACHE_DIR / "ent_emb.npy", ent_emb)
    np.save(CACHE_DIR / "rel_emb.npy", encode([r["fwd"] for r in rels] + [r["inv"] for r in rels]))
    for split in ("train", "dev", "test"):
        with open(CACHE_DIR / f"q_{split}.jsonl", "w", encoding="utf-8") as f:
            for it in items[split]:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    json.dump({"nodes": int(len(nodes)), "triples": int(len(triples)), "relations": len(rels) - 1,
               "seeds": len(seeds), "hub_degree": HUB_DEGREE, "mention_triples": len(new), "renamed": len(renamed)},
              open(CACHE_DIR / "kg_stats.json", "w"))


def _mentions(it):
    """(p, q) paragraph pairs of one question where the text of paragraph p mentions the title of paragraph q."""
    out = []
    keys = [_norm_title(t).lower() for t in it["titles"]]
    texts = {}
    for p, _, unit, _ in it["sents"]:
        texts[p] = texts.get(p, "") + " " + unit[len(it["titles"][p]) + 2:].lower()
    for p, text in texts.items():
        for q, key in enumerate(keys):
            if q != p and len(key) >= 4 and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text):
                out.append((p, q))
    return out


def anchor(it, qv, s_emb):
    """Question anchors: linked paragraph titles mentioned in the question (in order of mention); otherwise the
    linked paragraph whose first sentence is closest to the question (dense fallback)."""
    q = it["text"].lower()
    hits = []
    for title, ent in zip(it["titles"], it["title_ents"]):
        if ent < 0:
            continue
        pos = -1
        for key in (title, _norm_title(title)):
            pos = q.find(key.lower())
            if pos >= 0:
                break
        if pos >= 0 and ent not in [e for _, e in hits]:
            hits.append((pos, ent))
    if hits:
        it["anchors"], it["anchor_mode"] = [e for _, e in sorted(hits)], "alias"
        return
    best, best_s = None, -9.0
    for p, ent in enumerate(it["title_ents"]):
        if ent < 0:
            continue
        rows = [k for k, s in enumerate(it["sents"]) if s[0] == p]
        if rows:
            sc = float(s_emb[it["sent_off"] + rows[0]].astype(np.float32) @ qv)
            if sc > best_s:
                best, best_s = ent, sc
    it["anchors"], it["anchor_mode"] = ([best], "dense") if best is not None else ([], "none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    prepare(a.n_train, a.n_dev, a.n_test, a.seed)
