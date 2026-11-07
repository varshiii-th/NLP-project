"""Evaluation metrics for both sub-tasks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .config import POLARITIES


# --------------------------------------------------------------------------- ASC
def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(POLARITIES)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
           "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}
    for i, name in enumerate(POLARITIES):
        out[f"{name}_precision"], out[f"{name}_recall"] = float(pc[i]), float(rc[i])
        out[f"{name}_f1"], out[f"{name}_support"] = float(fc[i]), int(sc[i])
    return out


# --------------------------------------------------------------------------- ATE
def bio_to_spans(tags) -> list[tuple[int, int]]:
    """BIO tag ids/strings -> list of (start, end_exclusive). A stray I after O starts a new span."""
    spans, start = [], None
    for i, t in enumerate(tags):
        t = {0: "O", 1: "B", 2: "I"}.get(t, t)
        if t == "B" or (t == "I" and start is None):
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == "O":
            if start is not None:
                spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def span_prf(gold: list, pred: list) -> dict:
    """Exact-match span precision/recall/F1 over a corpus (lists of per-sentence span lists)."""
    tp = n_pred = n_gold = 0
    for g, p in zip(gold, pred):
        g, p = set(map(tuple, g)), set(map(tuple, p))
        tp += len(g & p)
        n_pred += len(p)
        n_gold += len(g)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


# --------------------------------------------------------------------------- end-to-end
def joint_prf(sents: list[dict], pred_pairs: list[set]) -> dict:
    """Aspect-polarity pair F1: a prediction counts only if span AND polarity are right.
    Gold aspects with the dropped 'conflict' polarity are ignored on both sides."""
    tp = n_pred = n_gold = 0
    for s, pred in zip(sents, pred_pairs):
        gold, ignore = set(), set()
        for a in s["aspects"]:
            inside = [j for j, (st, en) in enumerate(s["tok_spans"]) if st < a["to"] and en > a["from"]]
            if not inside:
                continue
            span = (inside[0], inside[-1] + 1)
            (gold.add((span, a["polarity"])) if a["polarity"] in POLARITIES else ignore.add(span))
        pred = {p for p in pred if p[0] not in ignore}
        tp += len(gold & pred)
        n_pred += len(pred)
        n_gold += len(gold)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
