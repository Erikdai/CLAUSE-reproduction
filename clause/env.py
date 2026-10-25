"""Symbolic CLAUSE environment, vectorised over a batch of questions.

Exploration rounds (edit -> traverse):
  * Subgraph Architect : ADD(e) | DELETE(e) | STOP over frontier-adjacent edges (one edit per round)
  * Path Navigator     : CONTINUE(r, v') along G_t | BACKTRACK | STOP
Curation (list-wise, dynamic K):
  * Context Curator    : SELECT(d) from the evidence pool P_t | STOP  (>= k_min picks)
Costs: edits -> C_edge, interaction steps -> C_lat, selected tokens -> C_tok.

Evidence units are KG facts, navigator paths and paragraphs. The task reward is computed without the reader:
the gold answer string in the curated evidence, mixed with supporting-paragraph coverage.
"""
import re
import string

import numpy as np
import scipy.sparse as sp

from .data import N_REL, TEXT_REL, inv_rel

PUNCT = set(string.punctuation)
STOPWORDS = set("the a an of in on at to for from by with and or is was were are be been which what who whom whose "
                "when where how why that this these those did does do has have had as it its his her their there "
                "than then both same also not".split())
WORD = re.compile(r"\w+")

PAD = N_REL
GLOB_DIM = 20
SCAL = {"arch": 17, "nav": 8, "cur": 17}      # candidate feature sizes (see _obs_*)
N_SPECIAL = {"arch": 1, "nav": 2, "cur": 1}   # arch: STOP | nav: STOP, BACKTRACK | cur: STOP
AGENTS = ("arch", "nav", "cur")
STOP, BACKTRACK = 0, 1
EXPLORE, CURATE = 0, 1


class Unit:
    """One evidence unit: kind "e" (KG fact), "p" (navigator path) or "s" (paragraph)."""
    __slots__ = ("key", "kind", "text", "tok", "answer", "relseq", "born", "nodes", "sim", "hit", "lex", "bm25")

    def __init__(self, key, kind, text, tok, answer, relseq, born, nodes=(), sim=0.0, hit=False, lex=0.0, bm25=0.0):
        self.key, self.kind, self.text, self.tok = key, kind, text, tok
        self.answer, self.relseq, self.born, self.nodes = answer, relseq, born, nodes
        self.sim, self.hit, self.lex, self.bm25 = sim, hit, lex, bm25


def norm_answer(s):
    """HotpotQA answer normalisation (lower case, no punctuation / articles / extra whitespace)."""
    s = "".join(ch for ch in s.lower() if ch not in PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def content_words(s):
    return {w for w in re.findall(r"\w+", s.lower()) if len(w) > 2 and w not in STOPWORDS}


class BM25:
    """Sparse-matrix BM25 (Okapi, k1=1.5, b=0.75)."""

    def __init__(self, texts, k1=1.5, b=0.75):
        vocab, rows, cols, vals = {}, [], [], []
        lens = np.zeros(len(texts), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = WORD.findall(t.lower())
            lens[i] = len(toks)
            tf = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            for w, c in tf.items():
                j = vocab.setdefault(w, len(vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)
        tf = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(texts), len(vocab)))
        df = np.bincount(np.array(cols), minlength=len(vocab))
        idf = np.log((len(texts) - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        norm = k1 * (1 - b + b * lens / lens.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.W = sp.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.vocab = vocab

    def scores(self, query):
        cols = [self.vocab[w] for w in set(WORD.findall(query.lower())) if w in self.vocab]
        if not cols:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, cols].sum(1)).ravel()


class DistMap:
    """Hop distance to a few target nodes, up to 2 hops (hub nodes are not expanded); 3 otherwise."""

    def __init__(self, kg, targets, hub=2000):
        self.levels = [np.unique(np.asarray(list(targets), dtype=np.int64))]
        seen = self.levels[0]
        for _ in range(2):
            fr = [u for u in self.levels[-1] if kg.degree[u] <= hub]
            nb = np.unique(np.concatenate([kg.adj_tail[kg.indptr[u]:kg.indptr[u + 1]] for u in fr])) if fr \
                else np.empty(0, dtype=np.int64)
            nb = np.setdiff1d(nb, seen, assume_unique=True)
            self.levels.append(nb)
            seen = np.union1d(seen, nb)

    def __getitem__(self, v):
        for d, lv in enumerate(self.levels):
            i = np.searchsorted(lv, v)
            if i < len(lv) and lv[i] == v:
                return d
        return 3


def _z(x):
    """Standardise similarity scores within one candidate set (0 for singletons)."""
    if len(x) < 2:
        return np.zeros_like(x)
    return (x - x.mean()) / (x.std() + 1e-6)


class Episode:
    def __init__(self, item, qvec, caps, dist, rel_cos):
        self.item = item
        self.q = qvec
        # question-relation cosines, z-scored per question
        self.qr = (rel_cos - rel_cos.mean()) / (rel_cos.std() + 1e-6)
        self.anchors = list(item["anchors"])
        self.gold = set(item["gold"])
        self.dist = dist
        self.edge_cap, self.lat_cap, self.tok_cap = caps
        a0 = self.anchors[0] if self.anchors else None
        self.depth = {a: 0 for a in self.anchors}
        self.chain = {a: () for a in self.anchors}
        self.node_order = list(self.anchors)
        self.edges = {}             # (u, r, v) -> round added
        self.edge_list = []
        self.out_adj = {}           # u -> list[(r, v)]
        self.deleted = set()        # deleted edges are not re-proposed
        self.dead = set()           # (u, r, v) branches abandoned by BACKTRACK
        self.path_nodes = [a0] if a0 is not None else []
        self.path_rels = []
        self.seen_tips = set(self.path_nodes) | set(self.anchors)
        self.pool = []
        self.pool_keys = {}
        self.selected = []
        self.sel_answers = set()
        self.C_edge = self.C_lat = self.C_tok = 0
        self.t = 0
        self.phase = EXPLORE
        self.done = a0 is None
        self.truncated = False
        self.stopped = {"arch": False, "nav": False, "cur": False}
        self.acted = False
        self.round_cost = [0, 0, 0]
        self.sel_reward = 0.0
        self.proxy = 0.0
        self.phi = self._phi()
        self.cands = {}
        # labels for the reward and the statistics
        self.yesno = norm_answer(item["answers"][0]) in ("yes", "no")
        self.ans_norm = f" {norm_answer(item['answers'][0])} "
        self.support = set(item["support"])
        self.any_hit = False
        # the question's given paragraphs (distractor setting) and their entities
        self.prio = np.array(sorted({x for x in item["title_ents"] if x >= 0}), dtype=np.int64)
        self.para_of = {}
        for p, x in enumerate(item["title_ents"]):
            if x >= 0:
                self.para_of.setdefault(x, []).append(p)
        self.unlocked = set()
        self.q_words = content_words(item["text"])
        self.bm25 = {}

    def _phi(self, H=4):
        """Shaping potential -(d_tip + d_G) / (2 (H + 1)): hop distance of the navigator tip and of the subgraph
        to the answer."""
        if self.dist is None or not self.path_nodes:
            return 0.0
        d_tip = min(int(self.dist[self.path_nodes[-1]]), H + 1)
        d_g = min(min(int(self.dist[v]) for v in self.depth), H + 1)
        return -(d_tip + d_g) / (2 * (H + 1))


class ClauseEnv:
    def __init__(self, res, cfg, tok_counter):
        self.res, self.kg, self.cfg = res, res.kg, cfg
        self.ent_emb, self.rel_emb = res.ent_emb, res.rel_emb
        self.tok = tok_counter
        self.H = cfg.hop_cap
        self.dist_cache = {}
        self.K = {"arch": cfg.max_arch_cands + cfg.max_del_cands, "nav": cfg.max_nav_cands, "cur": cfg.max_pool}
        self.nbr_cache = {}

    def _nbrs(self, u):
        s = self.nbr_cache.get(u)
        if s is None:
            s = self.nbr_cache[u] = frozenset(self.kg.adj_tail[self.kg.indptr[u]:self.kg.indptr[u + 1]].tolist())
        return s

    # ------------------------------------------------------------------ episode management
    def reset(self, items, qvecs, lam, training, rng=None):
        self.lam = np.asarray(lam, dtype=np.float32)
        self.training = training
        self.eps = []
        for item, qv in zip(items, qvecs):
            caps = (self.cfg.edge_cap, self.cfg.lat_cap, self.cfg.tok_cap)
            if training and self.cfg.randomize_caps and rng is not None and rng.random() < 0.5:
                caps = (int(rng.integers(8, self.cfg.edge_cap + 1)),
                        int(rng.integers(8, self.cfg.lat_cap + 1)),
                        int(rng.choice([128, 256, 384, 512])))
            dist = self._dist(item) if training else None
            e = Episode(item, qv, caps, dist, self.rel_emb @ qv)
            self.eps.append(e)
            if not e.done:
                self._seed_text(e)
        return self.eps

    def _seed_text(self, e):
        """Paragraphs of the anchors, then the dense-fallback paragraphs, as the initial pool."""
        e.sent_sims = e.item["sent_emb"] @ e.q
        n_par = len(e.item["titles"])
        e.para_sims = np.full(n_par, -1.0, dtype=np.float32)
        for k, s in enumerate(e.item["sents"]):
            e.para_sims[s[0]] = max(e.para_sims[s[0]], e.sent_sims[k])
        pz = _z(e.para_sims)
        e.para_z = {x: float(max(pz[p] for p in ps)) for x, ps in e.para_of.items()}
        e.bm25 = self._bm25(e)
        for a in e.anchors:
            self._unlock(e, a)
        for p in np.argsort(-e.para_sims)[:self.cfg.text_fallback]:
            self._add_para(e, int(p))

    def _unlock(self, e, v):
        for p in e.para_of.get(v, ()):
            if p not in e.unlocked:
                e.unlocked.add(p)
                self._add_para(e, p)

    def _para_sents(self, e, p):
        return [k for k, s in enumerate(e.item["sents"]) if s[0] == p]

    def _add_para(self, e, p):
        """Paragraph p as one evidence unit: 'Title: s1 s2 ...'."""
        ks = self._para_sents(e, p)
        if not ks:
            return
        title = e.item["titles"][p]
        ent = e.item["title_ents"][p]
        ans = ent if ent >= 0 else e.anchors[0]
        text = f"{title}: " + " ".join(e.item["sents"][k][2][len(title) + 2:] for k in ks)
        self._add_unit(e, ("P", p), "s", text, ans, (TEXT_REL,), (ans,) if ent >= 0 else (),
                       sim=float(e.sent_sims[ks].max()))

    def _bm25(self, e):
        """z-scored BM25 of the question against each of its paragraphs, cached on the item since training
        revisits questions."""
        if "_bm25" not in e.item:
            units = [(("P", p), ks) for p in range(len(e.item["titles"])) if (ks := self._para_sents(e, p))]
            texts = [" ".join(e.item["sents"][k][2] for k in ks) for _, ks in units]
            sc = _z(BM25(texts).scores(e.item["text"]).astype(np.float32)) if texts else []
            e.item["_bm25"] = {key: float(v) for (key, _), v in zip(units, sc)}
        return e.item["_bm25"]

    def set_caps(self, edge=None, lat=None, tok=None):
        """Cap mode at inference: override budgets of the current batch."""
        for e in self.eps:
            e.edge_cap = edge or e.edge_cap
            e.lat_cap = lat or e.lat_cap
            e.tok_cap = tok or e.tok_cap

    def _dist(self, item):
        key = (item["split"], item["hop"], item["idx"])
        d = self.dist_cache.get(key)
        if d is None:
            d = self.dist_cache[key] = DistMap(self.kg, item["gold"] or item["support"] or item["anchors"])
        return d

    def active(self):
        return [e for e in self.eps if not e.done]

    def acting(self, agent, e):
        if e.done:
            return False
        if agent == "cur":
            return e.phase == CURATE
        return e.phase == EXPLORE and not e.stopped[agent]

    # ------------------------------------------------------------------ helpers
    def _pad_seq(self, rels):
        s = np.full(self.H, PAD, dtype=np.int64)
        s[:len(rels)] = rels[:self.H]
        return s

    def glob(self, e):
        tip = e.path_nodes[-1]
        tip_out = sum(1 for (r, v) in e.out_adj.get(tip, ()) if v not in e.path_nodes and (tip, r, v) not in e.dead)
        n_unsel = len(e.pool) - len(e.selected)
        return np.array([
            1 - e.C_edge / e.edge_cap, 1 - e.C_lat / e.lat_cap, 1 - e.C_tok / e.tok_cap,
            self.lam[0], self.lam[1], self.lam[2],
            len(e.path_rels) / self.H, len(e.depth) / 16.0, len(e.edges) / e.edge_cap,
            len(e.selected) / 8.0,
            float(e.stopped["arch"]), float(e.stopped["nav"]),
            tip_out / 4.0, self.kg.log_deg[tip] / 8.0, n_unsel / 16.0,
            float(e.phase == CURATE),
            e.tok_cap / 512.0, e.edge_cap / 16.0, e.lat_cap / 15.0,
            float(tip in e.anchors),
        ], dtype=np.float32)

    # ------------------------------------------------------------------ candidate generation
    def arch_cands(self, e):
        """Frontier-adjacent edges; the frontier is the set of nodes touched by traversal."""
        cfg, kg, H = self.cfg, self.kg, self.H
        tip = e.path_nodes[-1]
        rest = [n for n in e.node_order if n != tip and n in e.seen_tips and e.depth[n] < H]
        rest.sort(key=lambda n: (n not in e.path_nodes, -e.depth[n]))
        adds = []
        for u in [tip] + rest:
            if e.depth[u] >= H:
                continue
            if len(e.prio):
                # edges to the question's paragraph entities come first
                rels, tails = kg.out_edges(u)
                for j in np.flatnonzero(np.isin(tails, e.prio)):
                    r, v = int(rels[j]), int(tails[j])
                    if v == u or (u, r, v) in e.edges or (v, inv_rel(r), u) in e.edges or (u, r, v) in e.deleted:
                        continue
                    adds.append((u, r, v, 1))
            for r, tails in kg.groups(u):
                taken = 0
                ir = inv_rel(r)
                for v in tails:
                    if taken >= cfg.per_group:
                        break
                    if v == u or (u, r, v) in e.edges or (v, ir, u) in e.edges or (u, r, v) in e.deleted:
                        continue
                    if v in e.para_of:
                        continue                     # already proposed above
                    adds.append((u, r, v, len(tails)))
                    taken += 1
                if len(adds) >= cfg.max_arch_cands:
                    break
            if len(adds) >= cfg.max_arch_cands:
                break
        adds = adds[:cfg.max_arch_cands]
        path_edges = set(zip(e.path_nodes[:-1], e.path_rels, e.path_nodes[1:]))
        dels = [ed for ed in reversed(e.edge_list) if ed not in path_edges][:cfg.max_del_cands]
        return adds, dels

    def nav_cands(self, e):
        if len(e.path_rels) >= self.H:
            return []
        tip = e.path_nodes[-1]
        on_path = set(e.path_nodes)
        out = [(r, v) for (r, v) in e.out_adj.get(tip, ()) if v not in on_path and (tip, r, v) not in e.dead]
        return out[-self.cfg.max_nav_cands:]

    def cur_cands(self, e):
        rem = e.tok_cap - e.C_tok
        sel = set(e.selected)
        out = [i for i in range(len(e.pool) - 1, -1, -1) if i not in sel and e.pool[i].tok <= rem]
        return out[:self.cfg.max_pool]

    # ------------------------------------------------------------------ observations
    def observe(self, agent, eps):
        """Padded observation batch for one agent. Rows whose agent has no decision this round are
        marked acting=False (no-op) and only expose a dummy STOP."""
        K, B, S, H = self.K[agent], len(eps), SCAL[agent], self.H
        obs = {
            "q": np.stack([e.q for e in eps]).astype(np.float32),
            "glob": np.stack([self.glob(e) for e in eps]),
            "path_seq": np.stack([self._pad_seq(e.path_rels) for e in eps]),
            "path_len": np.array([len(e.path_rels) for e in eps], dtype=np.int64),
            "cand_rel": np.full((B, K), PAD, dtype=np.int64),
            "cand_seq": np.full((B, K, H), PAD, dtype=np.int64),
            "cand_scal": np.zeros((B, K, S), dtype=np.float32),
            "cand_price": np.zeros((B, K), dtype=np.float32),
            "cand_mask": np.zeros((B, K), dtype=bool),
            "spec_mask": np.ones((B, N_SPECIAL[agent]), dtype=bool),
            "acting": np.array([self.acting(agent, e) for e in eps], dtype=bool),
        }
        fill = getattr(self, f"_obs_{agent}")
        for b, e in enumerate(eps):
            if obs["acting"][b]:
                fill(e, b, obs)
            else:
                obs["spec_mask"][b, 1:] = False
        return obs

    def _obs_arch(self, e, b, obs):
        kg, H = self.kg, self.H
        adds, dels = self.arch_cands(e)
        if e.C_edge >= e.edge_cap:          # edge budget exhausted: only STOP
            adds, dels = [], []
        e.cands["arch"] = [("add", c) for c in adds] + [("del", d) for d in dels]
        if not e.cands["arch"]:
            return
        if not e.edges and e.C_edge == 0 and adds:
            obs["spec_mask"][b, STOP] = False   # G_0 is seeded around the anchors before stopping
        tip = e.path_nodes[-1]
        on_path = set(e.path_nodes)
        path_edges = set(zip(e.path_nodes[:-1], e.path_rels, e.path_nodes[1:]))
        us = np.array([c[1][0] for c in e.cands["arch"]])
        vs = np.array([c[1][2] for c in e.cands["arch"]])
        cqv = _z(self.ent_emb[vs] @ e.q)
        cqu = _z(self.ent_emb[us] @ e.q)
        for k, (kind, c) in enumerate(e.cands["arch"]):
            u, r, v = c[0], c[1], c[2]
            gsize = c[3] if kind == "add" else 1
            obs["cand_rel"][b, k] = r
            obs["cand_seq"][b, k] = self._pad_seq(list(e.chain[u]) + [r])
            obs["cand_scal"][b, k] = (
                kind == "add", kind == "del", e.qr[r], cqv[k], cqu[k], e.depth[u] / H, u == tip,
                u in on_path, kg.log_deg[u] / 8, kg.log_deg[v] / 8, np.log1p(gsize) / 6,
                v in e.depth, len(e.out_adj.get(u, ())) / 4, v in e.anchors, (u, r, v) in path_edges,
                # does the edge reach one of the question's paragraphs, and how well does that paragraph match
                v in e.para_of, e.para_z.get(v, 0.0))
            obs["cand_price"][b, k] = self.lam[0]
            obs["cand_mask"][b, k] = True

    def _obs_nav(self, e, b, obs):
        cands = self.nav_cands(e)
        e.cands["nav"] = cands
        obs["spec_mask"][b, BACKTRACK] = len(e.path_rels) > 0
        if not cands:
            return
        vs = np.array([v for _, v in cands])
        cqv = _z(self.ent_emb[vs] @ e.q)
        for k, (r, v) in enumerate(cands):
            obs["cand_rel"][b, k] = r
            obs["cand_seq"][b, k] = self._pad_seq(e.path_rels + [r])
            obs["cand_scal"][b, k] = (
                e.qr[r], cqv[k], self.kg.log_deg[v] / 8, len(e.out_adj.get(v, ())) > 0,
                len(self.kg.groups(v)) / 10, v in e.sel_answers, v in e.seen_tips,
                e.edges.get((e.path_nodes[-1], r, v), -1) == e.t)
            obs["cand_price"][b, k] = self.lam[1]
            obs["cand_mask"][b, k] = True

    def _obs_cur(self, e, b, obs):
        idx = self.cur_cands(e)
        e.cands["cur"] = idx
        if idx and len(e.selected) < self.cfg.k_min:
            obs["spec_mask"][b, STOP] = False   # at least k_min picks
        if not idx:
            return
        units = [e.pool[i] for i in idx]
        cqa = _z(np.array([u.sim for u in units], dtype=np.float32))
        cur_key = ("p", tuple(e.path_nodes), tuple(e.path_rels))
        sel_last = {}
        for i in e.selected:
            lr = e.pool[i].relseq[-1]
            sel_last[lr] = sel_last.get(lr, 0) + 1
        # paragraph entities of the selected paragraphs (for the KG-adjacency feature)
        sel_text = {e.pool[i].answer for i in e.selected if e.pool[i].kind == "s" and e.pool[i].nodes}
        for k, u in enumerate(units):
            last = u.relseq[-1]
            obs["cand_rel"][b, k] = last
            obs["cand_seq"][b, k] = self._pad_seq(list(u.relseq))
            d = e.depth.get(u.answer, self.H + 1)
            adj = u.answer not in sel_text and any(u.answer in self._nbrs(x) for x in sel_text)
            obs["cand_scal"][b, k] = (
                u.kind == "p", u.kind == "e", len(u.relseq) / self.H, u.tok / e.tok_cap, cqa[k],
                u.answer in e.anchors, u.answer in e.sel_answers, u.key == cur_key,
                (e.t - u.born) / e.lat_cap, e.qr[last], self.kg.log_deg[u.answer] / 8,
                sel_last.get(last, 0) / 4,
                # lexical overlap with the question; whether a paragraph's entity was reached through G_t
                u.lex, u.kind == "s" and 0 < d <= self.H, d / (self.H + 1),
                # BM25 against the question; KG-adjacent to an entity of the selected paragraphs
                u.bm25, adj)
            obs["cand_price"][b, k] = self.lam[2] * u.tok / e.tok_cap
            obs["cand_mask"][b, k] = True

    # ------------------------------------------------------------------ actions
    def _add_unit(self, e, key, kind, text, answer, relseq, nodes, sim=None):
        if key in e.pool_keys:
            return
        e.pool_keys[key] = len(e.pool)
        if sim is None:
            sim = float(self.ent_emb[answer] @ e.q)
        # reward label: contains the gold answer string (yes/no questions: is a supporting paragraph)
        if e.yesno:
            hit = kind == "s" and bool(set(nodes) & e.support)
        else:
            hit = e.ans_norm in f" {norm_answer(text)} "
        lex = len(e.q_words & content_words(text)) / len(e.q_words) if e.q_words else 0.0
        e.pool.append(Unit(key, kind, text, self.tok(text), answer, tuple(relseq), e.t, tuple(nodes), sim, hit, lex,
                           e.bm25.get(key, 0.0)))

    def _add_edge(self, e, u, r, v):
        e.edges[(u, r, v)] = e.t
        e.edge_list.append((u, r, v))
        e.out_adj.setdefault(u, []).append((r, v))
        if v not in e.depth:
            e.depth[v] = e.depth[u] + 1
            e.chain[v] = e.chain[u] + (r,)
            e.node_order.append(v)
            self._unlock(e, v)
        self._add_unit(e, ("e", u, r, v), "e", self.kg.edge_text(u, r, v), v, e.chain[u] + (r,), (u, v))

    def _del_edge(self, e, u, r, v):
        e.deleted.add((u, r, v))
        e.edges.pop((u, r, v), None)
        e.edge_list.remove((u, r, v))
        e.out_adj[u].remove((r, v))
        i = e.pool_keys.get(("e", u, r, v))
        if i is not None and i not in e.selected:
            e.pool[i].tok = 10 ** 9      # keep indices stable: the unit can no longer be selected

    def apply(self, agent, eps, actions):
        K = self.K[agent]
        for e, a in zip(eps, actions):
            a = int(a)
            if a < 0 or not self.acting(agent, e):
                continue
            getattr(self, f"_apply_{agent}")(e, a, K)

    def _apply_arch(self, e, a, K):
        if a >= K:
            e.stopped["arch"] = True
            return
        kind, c = e.cands["arch"][a]
        if kind == "add":
            self._add_edge(e, c[0], c[1], c[2])
        else:
            self._del_edge(e, c[0], c[1], c[2])
        e.C_edge += 1
        e.round_cost[0] += 1
        e.acted = True

    def _apply_nav(self, e, a, K):
        if a >= K:
            if a - K == STOP:
                e.stopped["nav"] = True
                return
            v = e.path_nodes.pop()
            r = e.path_rels.pop()
            e.dead.add((e.path_nodes[-1], r, v))
            e.acted = True
            return
        r, v = e.cands["nav"][a]
        e.path_nodes.append(v)
        e.path_rels.append(r)
        e.seen_tips.add(v)
        self._add_unit(e, ("p", tuple(e.path_nodes), tuple(e.path_rels)), "p",
                       self.kg.path_text(e.path_nodes, e.path_rels), v, e.path_rels, e.path_nodes)
        e.acted = True

    def _apply_cur(self, e, a, K):
        if a >= K:
            e.stopped["cur"] = True
            return
        i = e.cands["cur"][a]
        u = e.pool[i]
        e.selected.append(i)
        e.sel_answers.add(u.answer)
        e.C_tok += u.tok
        e.round_cost[2] += u.tok
        e.acted = True
        if self.training and e.dist is not None and not e.any_hit and u.hit:
            e.sel_reward += self.cfg.sel_w      # first selected unit with the answer
        if u.hit:
            e.any_hit = True

    # ------------------------------------------------------------------ round bookkeeping
    def end_round(self, eps):
        """Returns per-episode (task reward, edge cost, lat cost, tok cost, done)."""
        out = []
        for e in eps:
            c_edge, _, c_tok = e.round_cost
            c_lat = 0
            if e.acted:
                e.t += 1
                e.C_lat += 1
                c_lat = 1
            if e.phase == EXPLORE:
                explore_over = (e.stopped["arch"] and e.stopped["nav"]) or e.C_lat >= e.lat_cap - self.cfg.k_min
                if explore_over:
                    e.truncated = not (e.stopped["arch"] and e.stopped["nav"])
                    e.phase = CURATE
                    if not self.cur_cands(e):
                        e.done = True
            elif e.stopped["cur"] or e.C_lat >= e.lat_cap or not self.cur_cands(e):
                e.done = True
            r = e.sel_reward
            e.sel_reward = 0.0
            if self.training and e.dist is not None:
                phi = e._phi(self.H)
                r += self.cfg.shape_w * (phi - e.phi)
                e.phi = phi
            if e.done:
                e.proxy = self.proxy_reward(e, [e.pool[i] for i in e.selected])
                r += e.proxy
            out.append((r, c_edge / e.edge_cap, c_lat / e.lat_cap, c_tok / e.tok_cap, e.done))
            e.round_cost = [0, 0, 0]
            e.acted = False
        return out

    # ------------------------------------------------------------------ task reward and statistics
    def sp_cover(self, e, units):
        """Fraction of the supporting paragraphs whose text is in the evidence."""
        if not e.support:
            return 0.0
        nodes = {n for u in units if u.kind == "s" for n in u.nodes}
        return len(nodes & e.support) / len(e.support)

    def answer_hit(self, e, units):
        """Gold answer string in the curated evidence; for yes/no questions the supporting-paragraph coverage."""
        if e.yesno:
            return self.sp_cover(e, units)
        return float(any(u.hit for u in units))

    def proxy_reward(self, e, units):
        """Task reward: answer_hit mixed with supporting-paragraph coverage."""
        if e.yesno:
            return self.answer_hit(e, units)
        return (1 - self.cfg.sp_w) * self.answer_hit(e, units) + self.cfg.sp_w * self.sp_cover(e, units)

    def episode_stats(self, e):
        units = [e.pool[i] for i in e.selected]
        return {
            "proxy": e.proxy, "ans_recall": self.answer_hit(e, units), "sp_recall": self.sp_cover(e, units),
            "hop": e.item["hop"], "C_edge": e.C_edge, "C_lat": e.C_lat, "C_tok": e.C_tok, "n_sel": len(e.selected),
            "n_text": sum(u.kind == "s" for u in units), "n_kg": sum(u.kind != "s" for u in units),
            "edge_frac": e.C_edge / e.edge_cap, "lat_frac": e.C_lat / e.lat_cap, "tok_frac": e.C_tok / e.tok_cap,
            "truncated": e.truncated,
        }
