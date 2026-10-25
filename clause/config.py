"""Paths and default hyper-parameters."""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HOTPOT_DIR = DATA_DIR / "hotpotqa"        # HotpotQA distractor parquet files
WD5M_DIR = DATA_DIR / "wikidata5m"        # Wikidata5M triples and aliases
CACHE_DIR = DATA_DIR / "processed"        # local KG, questions and embeddings written by `python -m clause.prepare`
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

ENCODER_NAME = "thenlper/gte-small"       # frozen sentence encoder
TOKENIZER_NAME = "Qwen/Qwen3-32B"         # tokenizer that counts the token budget


@dataclass
class EnvConfig:
    hop_cap: int = 4            # traversal depth
    edge_cap: int = 16          # accepted subgraph edits
    lat_cap: int = 15           # interaction steps
    tok_cap: int = 512          # selected tokens
    k_min: int = 2              # minimum curated picks
    max_arch_cands: int = 48    # architect candidates (degree/type-aware neighbour sampling)
    max_del_cands: int = 16
    per_group: int = 2          # edges kept per (node, relation) group
    max_nav_cands: int = 16
    max_pool: int = 48          # curator candidates per step
    # max_pool: int = 128
    text_fallback: int = 10     # paragraphs added to the initial pool by dense similarity to the question
    # text_fallback: int = 64
    sp_w: float = 0.3           # weight of supporting-paragraph coverage in the task reward
    shape_w: float = 0.5        # progress-to-answer shaping (training only)
    sel_w: float = 0.1          # first selected unit that contains the answer (training only)
    randomize_caps: bool = True  # per-query budgets during training


@dataclass
class TrainConfig:
    iters: int = 400
    episodes_per_iter: int = 192
    ppo_epochs: int = 4
    minibatch: int = 512
    clip: float = 0.2
    ent_coef: float = 0.02
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    critic_epochs: int = 4
    gamma: float = 1.0
    # budget targets as fractions of the caps
    beta_edge: float = 0.25
    # beta_edge: float = 0.5
    beta_lat: float = 0.7
    beta_tok: float = 1.0
    dual_kp: float = 1.0        # PI dual update: proportional gain
    dual_ki: float = 0.2        # integral gain
    hidden: int = 64
    seed: int = 0
    threads: int = 4
    eval_every: int = 25        # dev evaluation interval; best.pt has the highest dev reward

    def to_dict(self):
        return asdict(self)
