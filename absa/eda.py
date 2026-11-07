"""Exploratory data analysis: polarity balance, aspects/sentence, length, frequent aspects."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATASET_NAMES, POLARITIES
from .data import describe
from .plotting import C, INK2, SURFACE, _save, grouped_bars


def run_eda(data: dict, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    summary = describe(data)
    summary.to_csv(out / "dataset_summary.csv", index=False)

    # 1. polarity distribution (share) per dataset & split
    cats = [f"{DATASET_NAMES[n]}\n{s}" for n in data for s in ("train", "val", "test")]
    shares = {p: [] for p in POLARITIES}
    for n in data:
        for s in ("train", "val", "test"):
            vc = data[n][s]["asc"].polarity.value_counts(normalize=True)
            for p in POLARITIES:
                shares[p].append(float(vc.get(p, 0)))
    grouped_bars(cats, shares, "Polarity distribution (share of aspect instances)", "share", out / "polarity_distribution.png",
                 ylim=(0, 0.75), fmt="{:.2f}", colors={"negative": C[7], "neutral": "#8a8984", "positive": C[2]})

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    stats = []
    for k, n in enumerate(data):
        sents = [s for sp in ("train", "val", "test") for s in data[n][sp]["sents"]]
        asc = pd.concat([data[n][sp]["asc"] for sp in ("train", "val", "test")])
        n_asp = np.array([len(s["aspects"]) for s in sents])
        lens = np.array([len(s["tokens"]) for s in sents])
        multi_pol = asc.groupby("sid").polarity.nunique()
        stats.append({"dataset": n, "mean_aspects_per_sentence(with>=1)": n_asp[n_asp > 0].mean(),
                      "share_sentences_multi_aspect": (n_asp > 1).mean(),
                      "share_sentences_multi_polarity": float((multi_pol > 1).mean()),
                      "mean_tokens": lens.mean(), "max_tokens": lens.max(),
                      "unique_aspect_terms": asc.aspect.str.lower().nunique()})
        bins = np.arange(0, min(n_asp.max(), 10) + 2) - 0.5
        axes[0].hist(np.clip(n_asp, 0, 10), bins=bins, alpha=0.75, color=C[k], label=DATASET_NAMES[n],
                     density=True, edgecolor=SURFACE)
        axes[1].hist(lens, bins=40, alpha=0.6, color=C[k], label=DATASET_NAMES[n], density=True, edgecolor=SURFACE)
    axes[0].set_title("Aspects per sentence")
    axes[0].set_xlabel("# aspect terms (10 = 10+)")
    axes[1].set_title("Sentence length")
    axes[1].set_xlabel("# word tokens")
    for a in axes[:2]:
        a.set_ylabel("density")
        a.legend()
    st = pd.DataFrame(stats)
    x = np.arange(2)
    for k, n in enumerate(data):
        r = st[st.dataset == n].iloc[0]
        axes[2].bar(x + (k - 0.5) * 0.38, [r.share_sentences_multi_aspect, r.share_sentences_multi_polarity], 0.36,
                    color=C[k], label=DATASET_NAMES[n], edgecolor=SURFACE)
    axes[2].set_xticks(x, ["multi-aspect\nsentences", "multi-polarity\nsentences"])
    axes[2].set_title("Aspect density")
    axes[2].set_ylabel("share of sentences")
    axes[2].legend()
    fig.tight_layout()
    _save(fig, out / "aspects_and_lengths.png")
    st.to_csv(out / "dataset_statistics.csv", index=False)

    # 3. most frequent aspect terms
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k, n in enumerate(data):
        cnt = Counter(data[n]["train"]["asc"].aspect.str.lower())
        top = cnt.most_common(15)[::-1]
        axes[k].barh([t for t, _ in top], [c for _, c in top], color=C[k], edgecolor=SURFACE)
        axes[k].set_title(f"Top aspect terms - {DATASET_NAMES[n]} (train)")
        axes[k].grid(axis="y", visible=False)
        axes[k].tick_params(axis="y", colors=INK2)
    fig.tight_layout()
    _save(fig, out / "top_aspects.png")

    # aspect-vocabulary overlap between domains (motivates the cross-dataset study)
    va = set(data["semeval"]["train"]["asc"].aspect.str.lower())
    vb = set(data["mams"]["train"]["asc"].aspect.str.lower())
    with open(out / "vocab_overlap.txt", "w") as f:
        f.write(f"unique aspect terms SemEval={len(va)}, MAMS={len(vb)}, shared={len(va & vb)} "
                f"(Jaccard={len(va & vb) / len(va | vb):.3f})\n")
    print(summary.to_string(index=False))
    print(st.round(3).to_string(index=False))
    return summary
