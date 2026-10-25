"""LLM reader over the curated evidence and the answer metrics.

The reader is any OpenAI-compatible chat endpoint, set by LLM_API_KEY, LLM_BASE_URL and LLM_MODEL.
"""
import os
import re
import time
from collections import Counter

from .env import norm_answer

PROMPT = (
    "Answer the question using only the evidence below (knowledge-graph facts and Wikipedia sentences). "
    "Give the shortest possible answer: an entity name, a date, a number or a short phrase; "
    "for yes/no questions answer yes or no. Reply with the answer only, nothing else.\n\n"
    "Evidence:\n{evidence}\n\nQuestion: {question}\nAnswer:"
)


def strict_em(pred, gold):
    """HotpotQA exact match: identical after normalisation."""
    return float(norm_answer(pred) == norm_answer(gold))


def lenient_em(pred, gold):
    """The normalised gold answer occurs as whole words in the normalised prediction ("Japanese yen" for "yen")."""
    p, g = norm_answer(pred), norm_answer(gold)
    return float(bool(p) and f" {g} " in f" {p} ")


def f1(pred, gold):
    p, g = norm_answer(pred).split(), norm_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


class LLMReader:
    def __init__(self, max_tokens=64):
        self.model = os.environ.get("LLM_MODEL", "")
        if not self.model:
            raise RuntimeError("LLM_MODEL is not set (see .env.example)")
        self.base_url = os.environ.get("LLM_BASE_URL") or None
        self.max_tokens = max_tokens
        self._client = None

    def message(self, question, texts):
        return PROMPT.format(evidence="\n".join(f"- {t}" for t in texts) or "(none)", question=question)

    def client(self):
        if self._client is None:                    # created on the first query, so cached runs need no key
            from openai import OpenAI
            key = os.environ.get("LLM_API_KEY")
            if not key:
                raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
            self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=120, max_retries=0)
        return self._client

    def __call__(self, question, texts, retries=6):
        msg = self.message(question, texts)
        for k in range(retries):
            try:
                r = self.client().chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": msg}], temperature=0.0,
                    max_tokens=self.max_tokens)
                break
            except Exception:
                if k == retries - 1:
                    raise
                time.sleep(2 ** k)
        out = r.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().split("\n")[0]
