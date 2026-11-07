"""Traditional baselines.

ASC : TF-IDF (sentence n-grams + aspect-local context window + aspect term) -> Logistic Regression / Linear SVM
ATE : hand-crafted token-window features -> DictVectorizer -> Logistic Regression BIO tagger

Hyper-parameter C is tuned on the validation split; the grid is evaluated in parallel on all cores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .config import N_CPU, SEED
from .data import parallel_map
from .metrics import bio_to_spans, classification_metrics, span_prf

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


# --------------------------------------------------------------------------- ASC
def _asc_features():
    return ColumnTransformer([
        ("sent", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True), "text"),
        ("ctx", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False, token_pattern=r"\S+"), "context"),
        ("asp", TfidfVectorizer(ngram_range=(1, 1), lowercase=True), "aspect"),
    ], sparse_threshold=1.0)


def _make_clf(kind: str, c: float):
    if kind == "lr":
        return LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=SEED)
    return LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=20000)


def _fit_eval(Xtr, ytr, Xva, yva, kind, c):
    clf = _make_clf(kind, c).fit(Xtr, ytr)
    return c, clf, classification_metrics(ytr, clf.predict(Xtr))["macro_f1"], classification_metrics(yva, clf.predict(Xva))["macro_f1"]


class ASCBaseline:
    def __init__(self, kind: str):
        self.kind = kind
        self.name = {"lr": "TF-IDF+LR", "svm": "TF-IDF+SVM"}[kind]

    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        self.feats = _asc_features().fit(train)
        Xtr, Xva = self.feats.transform(train), self.feats.transform(val)
        res = Parallel(n_jobs=min(N_CPU, len(C_GRID)))(
            delayed(_fit_eval)(Xtr, train.label.values, Xva, val.label.values, self.kind, c) for c in C_GRID)
        self.curve = {"C": [r[0] for r in res], "train_macro_f1": [r[2] for r in res], "val_macro_f1": [r[3] for r in res]}
        best = max(res, key=lambda r: r[3])
        self.best_c, self.clf = best[0], best[1]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.array([], dtype=int)
        return self.clf.predict(self.feats.transform(df))

    def evaluate(self, df: pd.DataFrame) -> tuple[dict, np.ndarray]:
        pred = self.predict(df)
        return classification_metrics(df.label.values, pred), pred


# --------------------------------------------------------------------------- ATE
def _word_shape(w):
    s = "".join("X" if ch.isupper() else "x" if ch.islower() else "d" if ch.isdigit() else ch for ch in w)
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _sent_features(tokens: list[str]) -> list[dict]:
    feats = []
    low = [t.lower() for t in tokens]
    n = len(tokens)
    for i, w in enumerate(tokens):
        f = {"bias": 1.0, "w": low[i], "suf3": low[i][-3:], "suf2": low[i][-2:], "pre3": low[i][:3],
             "shape": _word_shape(w), "is_title": float(w.istitle()), "is_digit": float(w.isdigit()),
             "is_alpha": float(w.isalpha()), "len": min(len(w), 12) / 12, "pos_rel": i / max(n - 1, 1)}
        for d in (-2, -1, 1, 2):
            j = i + d
            f[f"w[{d}]"] = low[j] if 0 <= j < n else ("<s>" if j < 0 else "</s>")
            if 0 <= j < n:
                f[f"suf3[{d}]"] = low[j][-3:]
        f["bigram-"] = f"{f['w[-1]']}|{low[i]}"
        f["bigram+"] = f"{low[i]}|{f['w[1]']}"
        feats.append(f)
    return feats


def _flatten_feats(sents):
    per_sent = parallel_map(_sent_features, [s["tokens"] for s in sents])   # multi-process feature extraction
    return [f for fs in per_sent for f in fs], [len(fs) for fs in per_sent]


def _fit_eval_ate(Xtr, ytr, Xva, va_lens, va_gold, c):
    clf = LogisticRegression(C=c, max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    return c, clf, _spans_from_flat(clf.predict(Xva), va_lens, va_gold)["f1"]


def _spans_from_flat(flat_pred, lens, gold):
    out, k = [], 0
    for n in lens:
        out.append(bio_to_spans(list(flat_pred[k:k + n])))
        k += n
    return span_prf(gold, out) if gold is not None else out


class ATEBaseline:
    name = "TF-IDF+LR"  # reported in the "traditional" row of ATE tables

    def fit(self, train_sents, val_sents):
        ftr, _ = _flatten_feats(train_sents)
        fva, va_lens = _flatten_feats(val_sents)
        self.vec = DictVectorizer().fit(ftr)
        Xtr, Xva = self.vec.transform(ftr), self.vec.transform(fva)
        ytr = np.array([t for s in train_sents for t in s["tag_ids"]])
        grid = C_GRID[2:]
        res = Parallel(n_jobs=min(N_CPU, len(grid)))(
            delayed(_fit_eval_ate)(Xtr, ytr, Xva, va_lens, [s["gold_spans"] for s in val_sents], c) for c in grid)
        self.curve = {"C": [r[0] for r in res], "val_f1": [r[2] for r in res]}
        best = max(res, key=lambda r: r[2])
        self.best_c, self.clf = best[0], best[1]
        return self

    def predict_spans(self, sents):
        f, lens = _flatten_feats(sents)
        return _spans_from_flat(self.clf.predict(self.vec.transform(f)), lens, None)

    def evaluate(self, sents):
        pred = self.predict_spans(sents)
        return span_prf([s["gold_spans"] for s in sents], pred), pred

