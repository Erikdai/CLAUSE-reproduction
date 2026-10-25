"""Local KG, frozen encoders, token counting and the cached resources the environment reads.

Everything is read from the files that `python -m clause.prepare` writes to data/processed/.
"""
import json
from functools import lru_cache

import numpy as np

from .config import CACHE_DIR, ENCODER_NAME, TOKENIZER_NAME

GROUP_CAP = 32      # tails kept per (node, relation) group: Wikidata has hubs with 10^5-10^6 edges

# relations of the local KG: Wikidata relations, `mentions`, and the text pseudo-relation of paragraph units (last)
RELATIONS = json.load(open(CACHE_DIR / "relations.json", encoding="utf-8")) \
    if (CACHE_DIR / "relations.json").exists() else []
REL_NAMES = [r["label"] for r in RELATIONS]
N_BASE_REL = len(REL_NAMES)
N_REL = 2 * N_BASE_REL  # forward + inverse
TEXT_REL = N_BASE_REL - 1


def inv_rel(r):
    return (r + N_BASE_REL) % N_REL


class KG:
    """Typed KG with inverse relations, CSR adjacency (sorted by relation per node)."""

    def __init__(self):
        self.names = open(CACHE_DIR / "kg_names.txt", encoding="utf-8").read().split("\n")[:-1]
        self.triples = np.load(CACHE_DIR / "kg_triples.npy")      # forward triples only
        self.n_ent = len(self.names)
        h, r, t = self.triples.T
        heads = np.concatenate([h, t])
        rels = np.concatenate([r, r + N_BASE_REL])
        tails = np.concatenate([t, h])
        order = np.lexsort((tails, rels, heads))
        heads, rels, tails = heads[order], rels[order], tails[order]
        self.indptr = np.zeros(self.n_ent + 1, dtype=np.int64)
        np.add.at(self.indptr, heads + 1, 1)
        self.indptr = np.cumsum(self.indptr)
        self.adj_rel = rels.astype(np.int64)
        self.adj_tail = tails.astype(np.int64)
        self.degree = np.diff(self.indptr)
        self.log_deg = np.log1p(self.degree).astype(np.float32)

    def out_edges(self, u):
        s, e = self.indptr[u], self.indptr[u + 1]
        return self.adj_rel[s:e], self.adj_tail[s:e]

    @lru_cache(maxsize=200_000)
    def groups(self, u):
        """Relation groups of node u: tuple of (rel, tails sorted by degree desc then id)."""
        rels, tails = self.out_edges(u)
        out = []
        if len(rels) == 0:
            return tuple()
        bounds = np.flatnonzero(np.diff(rels)) + 1
        for chunk_r, chunk_t in zip(np.split(rels, bounds), np.split(tails, bounds)):
            order = np.lexsort((chunk_t, -self.degree[chunk_t]))[:GROUP_CAP]
            out.append((int(chunk_r[0]), tuple(int(x) for x in chunk_t[order])))
        return tuple(out)

    def edge_text(self, u, r, v):
        """Textualised triple in canonical (forward) direction, e.g. 'Fatal Attraction — director: Adrian Lyne'."""
        if r >= N_BASE_REL:
            u, v, r = v, u, r - N_BASE_REL
        return f"{self.names[u]} — {REL_NAMES[r]}: {self.names[v]}"

    def path_text(self, nodes, rels):
        s = self.names[nodes[0]]
        for r, v in zip(rels, nodes[1:]):
            if r >= N_BASE_REL:
                s += f" <-[{REL_NAMES[r - N_BASE_REL]}]- {self.names[v]}"
            else:
                s += f" -[{REL_NAMES[r]}]-> {self.names[v]}"
        return s


# ----------------------------------------------------------------------------- encoders

_ENCODER = None
_TOKENIZER = None


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer(ENCODER_NAME, device="cpu")
    return _ENCODER


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _TOKENIZER


class TokenCounter:
    def __init__(self):
        self.tok = get_tokenizer()
        self.cache = {}

    def __call__(self, text):
        n = self.cache.get(text)
        if n is None:
            n = len(self.tok(text, add_special_tokens=False)["input_ids"])
            self.cache[text] = n
        return n


def encode(texts, batch_size=256):
    return get_encoder().encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=len(texts) > 5000).astype(np.float32)


class Resources:
    """KG, entity / relation embeddings and question splits, loaded once."""

    def __init__(self):
        self.kg = KG()
        # memory-mapped (~0.6 GB) so that parallel runs share one copy
        self.ent_emb = np.load(CACHE_DIR / "ent_emb.npy", mmap_mode="r")
        self.rel_emb = np.load(CACHE_DIR / "rel_emb.npy")
        self.questions = {}
        self.q_emb = {}

    def load_questions(self, split):
        items = [json.loads(l) for l in open(CACHE_DIR / f"q_{split}.jsonl", encoding="utf-8")]
        se = np.load(CACHE_DIR / f"q_{split}_sent_emb.npy").astype(np.float32)
        for it in items:
            it["sent_emb"] = se[it["sent_off"]:it["sent_off"] + len(it["sents"])]
        self.questions[split] = items
        self.q_emb[split] = np.load(CACHE_DIR / f"q_{split}_emb.npy")
        return items
