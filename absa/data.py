"""Data loading & preprocessing.

* Parses SemEval-2014 Laptop and MAMS XML files (in parallel processes).
* Converts both into one common schema:
    sentence-level  : {sid, text, tokens, spans, tags, aspects:[{term, from, to, polarity}]}
    aspect-level    : DataFrame rows (sid, text, aspect, from, to, polarity, label, ...)
* BIO tagging / tokenisation is fanned out over all CPU cores with a process pool.
* Results are cached as a pickle so later runs skip preprocessing entirely.

This module deliberately does not import torch so worker processes start fast on Windows (spawn).
"""
from __future__ import annotations

import hashlib
import pickle
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .config import (CACHE_DIR, DATA_FILES, MAMS_URL, N_CPU, POL2ID, POLARITIES,
                     SEED, SEMEVAL_VAL_FRACTION, TAG2ID)

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# --------------------------------------------------------------------------- download
def ensure_mams() -> None:
    """Download the MAMS-ATSA XML files if they are not present."""
    for split, path in DATA_FILES["mams"].items():
        if path.exists() and path.stat().st_size > 0:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = MAMS_URL.format(split=split)
        print(f"[data] downloading MAMS {split} <- {url}")
        urllib.request.urlretrieve(url, path)


# --------------------------------------------------------------------------- parsing
def parse_xml(path: str | Path) -> list[dict]:
    """Parse one SemEval-style XML file (SemEval-14 and MAMS share the schema)."""
    root = ET.parse(str(path)).getroot()
    out = []
    for i, s in enumerate(root.iter("sentence")):
        text = s.findtext("text") or ""
        aspects = []
        for a in s.iter("aspectTerm"):
            term = a.get("term")
            frm, to = int(a.get("from")), int(a.get("to"))
            if text[frm:to] != term:              # repair the occasional broken offset
                k = text.find(term)
                if k < 0:
                    continue
                frm, to = k, k + len(term)
            aspects.append({"term": term, "from": frm, "to": to, "polarity": a.get("polarity")})
        out.append({"sid": s.get("id", f"{Path(path).stem}-{i}"), "text": text, "aspects": aspects})
    return out


def _tokenize_and_tag(sent: dict) -> dict:
    """Word-tokenise with char offsets and produce BIO tags from aspect char spans."""
    text = sent["text"]
    toks, spans = [], []
    for m in TOKEN_RE.finditer(text):
        toks.append(m.group())
        spans.append((m.start(), m.end()))
    tags = ["O"] * len(toks)
    for a in sorted(sent["aspects"], key=lambda x: x["from"]):
        inside = [j for j, (s, e) in enumerate(spans) if s < a["to"] and e > a["from"]]
        for k, j in enumerate(inside):
            if tags[j] == "O":
                tags[j] = "B" if k == 0 else "I"
    # gold aspect spans in token space (start, end_exclusive), used for span-level ATE metrics
    gold_tok_spans = []
    for a in sent["aspects"]:
        inside = [j for j, (s, e) in enumerate(spans) if s < a["to"] and e > a["from"]]
        if inside:
            gold_tok_spans.append((inside[0], inside[-1] + 1))
    return {**sent, "tokens": toks, "tok_spans": spans, "tags": tags,
            "tag_ids": [TAG2ID[t] for t in tags], "gold_spans": sorted(set(gold_tok_spans))}


def _process_file(args: tuple[str, str, str]) -> tuple[str, str, list[dict]]:
    name, split, path = args
    sents = [_tokenize_and_tag(s) for s in parse_xml(path)]
    return name, split, sents


def parallel_map(fn, items, chunksize: int | None = None, n_jobs: int = N_CPU):
    items = list(items)
    if not items:
        return []
    n_jobs = max(1, min(n_jobs, len(items)))
    chunksize = chunksize or max(1, len(items) // (n_jobs * 4))
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(fn, items, chunksize=chunksize))


# --------------------------------------------------------------------------- aspect-level frame
def context_window(text: str, frm: int, to: int, k: int = 6) -> str:
    """k words left and right of the aspect (aspect excluded) - local opinion context for TF-IDF."""
    left = TOKEN_RE.findall(text[:frm])[-k:]
    right = TOKEN_RE.findall(text[to:])[:k]
    return " ".join(["L_" + w.lower() for w in left] + ["R_" + w.lower() for w in right])


def _sentence_to_rows(sent: dict) -> list[dict]:
    asp = [a for a in sent["aspects"] if a["polarity"] in POL2ID]
    n_asp = len(asp)
    n_pol = len({a["polarity"] for a in asp})
    rows = []
    for a in asp:
        rows.append({
            "sid": sent["sid"], "text": sent["text"], "aspect": a["term"],
            "from": a["from"], "to": a["to"], "polarity": a["polarity"], "label": POL2ID[a["polarity"]],
            "context": context_window(sent["text"], a["from"], a["to"]),
            "n_aspects": n_asp, "n_polarities": n_pol, "n_tokens": len(sent["tokens"]),
        })
    return rows


def to_aspect_frame(sents: list[dict]) -> pd.DataFrame:
    rows = [r for rs in parallel_map(_sentence_to_rows, sents) for r in rs]
    return pd.DataFrame(rows, columns=["sid", "text", "aspect", "from", "to", "polarity", "label",
                                       "context", "n_aspects", "n_polarities", "n_tokens"])


def predicted_aspect_frame(sents: list[dict], pred_spans: list[list[tuple[int, int]]]) -> pd.DataFrame:
    """Aspect-level frame built from *predicted* token spans (for the end-to-end pipeline)."""
    rows = []
    for i, (s, spans) in enumerate(zip(sents, pred_spans)):
        for (a, b) in spans:
            frm, to = s["tok_spans"][a][0], s["tok_spans"][b - 1][1]
            rows.append({"sent_idx": i, "sid": s["sid"], "text": s["text"], "aspect": s["text"][frm:to],
                         "from": frm, "to": to, "tok_span": (a, b),
                         "context": context_window(s["text"], frm, to)})
    return pd.DataFrame(rows, columns=["sent_idx", "sid", "text", "aspect", "from", "to", "tok_span", "context"])


# --------------------------------------------------------------------------- main entry
def _cache_key() -> str:
    h = hashlib.md5()
    for name, splits in DATA_FILES.items():
        for split, p in splits.items():
            h.update(f"{name}{split}{p.stat().st_size}{p.stat().st_mtime_ns}".encode())
    h.update(f"{SEED}{SEMEVAL_VAL_FRACTION}v3".encode())
    return h.hexdigest()[:12]


def load_datasets(use_cache: bool = True) -> dict:
    """Return {dataset: {split: {"sents": [...], "asc": DataFrame}}} for semeval & mams."""
    ensure_mams()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"datasets_{_cache_key()}.pkl"
    if use_cache and cache.exists():
        print(f"[data] loading cached preprocessed data: {cache.name}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    jobs = [(n, s, str(p)) for n, sp in DATA_FILES.items() for s, p in sp.items()]
    print(f"[data] parsing + BIO-tagging {len(jobs)} files on {min(N_CPU, len(jobs))} processes")
    raw = {}
    with ProcessPoolExecutor(max_workers=min(N_CPU, len(jobs))) as ex:
        for name, split, sents in ex.map(_process_file, jobs):
            raw.setdefault(name, {})[split] = sents

    # SemEval has no validation split: hold out a random 10% of training *sentences* (no leakage).
    rng = random.Random(SEED)
    tr = raw["semeval"]["train"][:]
    rng.shuffle(tr)
    n_val = int(len(tr) * SEMEVAL_VAL_FRACTION)
    raw["semeval"]["val"], raw["semeval"]["train"] = tr[:n_val], tr[n_val:]

    data = {}
    for name in raw:
        data[name] = {}
        for split in ("train", "val", "test"):
            sents = raw[name][split]
            data[name][split] = {"sents": sents, "asc": to_aspect_frame(sents)}
    with open(cache, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def describe(data: dict) -> pd.DataFrame:
    rows = []
    for name, splits in data.items():
        for split, d in splits.items():
            df = d["asc"]
            rows.append({"dataset": name, "split": split, "sentences": len(d["sents"]),
                         "sentences_with_aspects": sum(bool(s["aspects"]) for s in d["sents"]),
                         "aspect_terms(all)": sum(len(s["aspects"]) for s in d["sents"]),
                         "asc_instances": len(df),
                         **{p: int((df.polarity == p).sum()) for p in POLARITIES}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = load_datasets(use_cache=False)
    print(describe(d).to_string(index=False))
