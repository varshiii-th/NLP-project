"""Error analysis: tag each test instance with linguistic phenomena and measure accuracy per phenomenon.

Phenomena (heuristic, rule-based tags):
  negation        - sentence contains a negator (not, never, n't, no, without, ...)
  contrast        - sentence contains a contrastive connective (but, however, although, ...)
  multi_aspect    - sentence has > 1 annotated aspect
  multi_polarity  - sentence's aspects carry > 1 distinct polarity
  implicit        - no opinion-lexicon word anywhere in the sentence
  long_range      - nearest opinion word is >= 5 tokens away from the aspect
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import POLARITIES
from .data import TOKEN_RE, parallel_map

NEG_RE = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|without|hardly|barely|cannot|cant|dont|"
                    r"doesnt|didnt|isnt|wasnt|wont|wouldnt|couldnt|shouldnt)\b|n't", re.I)
CON_RE = re.compile(r"\b(but|however|although|though|yet|whereas|while|despite|except|nevertheless|"
                    r"nonetheless|otherwise|unfortunately)\b", re.I)

_FALLBACK = """good great excellent amazing awesome fantastic love loved like liked nice best better perfect
wonderful happy pleased fast quick easy beautiful delicious tasty fresh friendly helpful attentive reasonable
affordable cheap superb impressive solid reliable comfortable recommend enjoy enjoyed favorite incredible
outstanding decent fine bad worst worse terrible awful horrible poor slow hate hated disappointing disappointed
disappointment expensive overpriced rude dirty cold bland broken problem problems issue issues annoying noisy
loud crap junk useless mediocre unfortunately fail failed failure died dead crashed crashes slowly lacking
mess sucks sucked difficult hard heavy hot cramped small tiny overcooked undercooked stale soggy greasy
salty burnt unhelpful slowest inattentive pricey weak flimsy cheaply buggy laggy""".split()


@lru_cache(maxsize=1)
def opinion_lexicon() -> frozenset:
    try:
        import nltk
        try:
            from nltk.corpus import opinion_lexicon as ol
            words = set(ol.positive()) | set(ol.negative())
        except LookupError:
            nltk.download("opinion_lexicon", quiet=True)
            from nltk.corpus import opinion_lexicon as ol
            words = set(ol.positive()) | set(ol.negative())
        if words:
            return frozenset(words)
    except Exception:
        pass
    return frozenset(_FALLBACK)


def _tag_row(args):
    text, frm, to, n_asp, n_pol = args
    lex = opinion_lexicon()  # cached once per worker process
    toks = [(m.group().lower(), m.start()) for m in TOKEN_RE.finditer(text)]
    op_pos = [i for i, (t, _) in enumerate(toks) if t in lex]
    asp_idx = [i for i, (_, s) in enumerate(toks) if frm <= s < to] or [0]
    dist = min((min(abs(o - a) for a in asp_idx) for o in op_pos), default=None)
    return {"negation": bool(NEG_RE.search(text)), "contrast": bool(CON_RE.search(text)),
            "multi_aspect": n_asp > 1, "multi_polarity": n_pol > 1, "implicit": not op_pos,
            "long_range": dist is not None and dist >= 5}


PHENOMENA = ["negation", "contrast", "multi_aspect", "multi_polarity", "implicit", "long_range"]


def tag_phenomena(df: pd.DataFrame) -> pd.DataFrame:
    opinion_lexicon()  # make sure the lexicon is downloaded before workers start
    rows = parallel_map(_tag_row, [(r.text, r["from"], r.to, r.n_aspects, r.n_polarities)
                                   for _, r in df.iterrows()])
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def accuracy_by_phenomenon(tagged: pd.DataFrame, pred: np.ndarray) -> dict:
    correct = tagged.label.values == np.asarray(pred)
    out = {"all": float(correct.mean())}
    for p in PHENOMENA:
        m = tagged[p].values
        out[p] = float(correct[m].mean()) if m.any() else float("nan")
        out[f"{p}_n"] = int(m.sum())
    return out


def misclassified(tagged: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    df = tagged.copy()
    df["pred"] = [POLARITIES[i] for i in pred]
    df = df[df.pred != df.polarity]
    df["phenomena"] = df[PHENOMENA].apply(lambda r: ",".join(p for p in PHENOMENA if r[p]), axis=1)
    return df[["sid", "text", "aspect", "polarity", "pred", "phenomena"]]
