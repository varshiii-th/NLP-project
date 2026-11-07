"""Version 2 training round - informed improvements over v1, results in results_v2/.

Diagnosis of v1 (see results/ and results_v2/IMPROVEMENTS.md):
  * strong over-fitting: train macro-F1 -> 0.94-0.98 while val plateaus ~epoch 2 and val loss rises
  * the [CLS]-only head does not know *which* aspect of a multi-aspect sentence it is judging
  * neutral is the weakest class (most errors are neutral <-> pos/neg)
  * noisy checkpoint selection on SemEval's small validation split (221 aspects)
  * base-size encoders

v2 recipe
  ASC  aspect markers in the sentence ("# battery life #") + aspect-pooled head ([CLS] ++ mean(aspect)),
       label smoothing 0.1, FGM adversarial training (eps 0.5), layer-wise LR decay, smaller batches,
       4 validation checks / epoch
  ATE  BIO-constrained Viterbi decoding, FGM, layer-wise LR decay, 4 validation checks / epoch
  Backbones: RoBERTa-base (isolates the recipe effect vs v1) and RoBERTa-large (adds capacity)

Usage:  python run_v2.py            (resumable; --fresh to retrain; --quick for a smoke test)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_all
from absa.config import DATASET_NAMES, ROOT
from absa.data import load_datasets

DSETS = ["semeval", "mams"]
V1_DIR = ROOT / "results"
V2_DIR = ROOT / "results_v2"
V1_NAMES = {"BERT": "BERT-base v1", "RoBERTa": "RoBERTa-base v1"}

_ASC_COMMON = dict(aspect_markers=True, aspect_pool=True, label_smoothing=0.1, fgm_eps=0.5,
                   evals_per_epoch=4, patience=2, warmup_ratio=0.1, batch_size=16, train_eval_max=3000)
_ATE_COMMON = dict(viterbi=True, fgm_eps=0.5, evals_per_epoch=4, patience=3, batch_size=16, train_eval_max=3000)

VARIANTS = {
    "roberta-v2": {
        "model_key": "roberta", "label": "RoBERTa-base v2",
        "asc": dict(_ASC_COMMON, epochs=5, lr=3e-5, llrd=0.9, head_lr=1e-4),
        "ate": dict(_ATE_COMMON, epochs=8, lr=5e-5, llrd=0.9, head_lr=1e-4),
    },
    "roberta-large-v2": {
        "model_key": "roberta-large", "label": "RoBERTa-large v2",
        "asc": dict(_ASC_COMMON, epochs=4, lr=1.5e-5, llrd=0.95, head_lr=5e-5, grad_ckpt=True),
        "ate": dict(_ATE_COMMON, epochs=6, lr=2e-5, llrd=0.95, head_lr=5e-5, grad_ckpt=True),
    },
}


def load_v1():
    """v1 transformer results (metrics + predictions) so v1 and v2 are reported side by side."""
    asc_rows, ate_rows, e2e_rows, preds, hist = [], [], [], {}, {}
    for task in ("ate", "asc"):
        for f in sorted((V1_DIR / "transformers" / task).glob("*/test_results.json")):
            r = json.load(open(f))
            name = V1_NAMES.get(r["model"], r["model"] + " v1")
            hist[(name, task, r["train"])] = json.load(open(f.parent / "history.json"))["history"]
            for te, m in r["test"].items():
                row = {"model": name, "train": r["train"], "test": te, **m}
                (asc_rows if task == "asc" else ate_rows).append(row)
                if task == "asc":
                    preds[(name, r["train"], te)] = r["pred"][te]
            for te, j in r.get("e2e", {}).items():
                e2e_rows.append({"model": name, "train": r["train"], "test": te, **j})
    return asc_rows, ate_rows, e2e_rows, preds, hist


# =========================================================================== improvement report
def improvement_report(asc, ate, e2e, hist, log):
    from absa.plotting import MODEL_COLORS, _save, grouped_bars, plt

    fdir = V2_DIR / "improvements"
    fdir.mkdir(parents=True, exist_ok=True)
    ref = "RoBERTa-base v1"
    news = [m for m in ["RoBERTa-base v2", "RoBERTa-large v2"] if m in set(asc.model)]
    cfgs = [(a, b) for a in DSETS for b in DSETS]
    cat = [f"{DATASET_NAMES[a].split()[0]} -> {DATASET_NAMES[b].split()[0]}" for a, b in cfgs]

    def get(df, m, a, b, col):
        s = df[(df.model == m) & (df.train == a) & (df.test == b)]
        return float(s[col].iloc[0]) if len(s) else np.nan

    rows = []
    for task, df, cols in [("ASC", asc, ["accuracy", "macro_f1", "neutral_f1"]), ("ATE", ate, ["f1"]),
                           ("End-to-end", e2e, ["f1"])]:
        for a, b in cfgs:
            for col in cols:
                r = {"task": task, "metric": col, "train": a, "test": b,
                     "setting": "in-domain" if a == b else "cross-dataset", "v1 RoBERTa-base": get(df, ref, a, b, col)}
                for m in news:
                    r[m] = get(df, m, a, b, col)
                    r[f"delta {m}"] = r[m] - r["v1 RoBERTa-base"]
                rows.append(r)
    imp = pd.DataFrame(rows)
    imp.to_csv(fdir / "v1_vs_v2.csv", index=False)

    # bar charts: v1 models vs v2 models per configuration
    models = [m for m in ["BERT-base v1", "RoBERTa-base v1"] + news if m in set(asc.model)]
    for df, col, name, lab in [(asc, "macro_f1", "asc_macro_f1", "ASC macro-F1"), (asc, "accuracy", "asc_accuracy", "ASC accuracy"),
                               (ate, "f1", "ate_f1", "ATE span F1"), (e2e, "f1", "end2end_f1", "End-to-end pair F1")]:
        grouped_bars(cat, {m: [get(df, m, a, b, col) for a, b in cfgs] for m in models},
                     f"{lab}: v1 vs v2", lab, fdir / f"v1_vs_v2_{name}.png", colors=MODEL_COLORS)

    # validation-curve overlays: how the learning dynamics changed
    for task, key, lab in [("asc", "val_macro_f1", "validation macro-F1"), ("ate", "val_f1", "validation span F1")]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
        for i, tr in enumerate(DSETS):
            for m in models:
                h = hist.get((m, task, tr))
                if not h:
                    continue
                ep = [r["epoch"] for r in h]
                axes[0, i].plot(ep, [r[key] for r in h], "-o", ms=3.5, color=MODEL_COLORS[m], label=m)
                axes[1, i].plot(ep, [r["val_loss"] for r in h], "-o", ms=3.5, color=MODEL_COLORS[m], label=m)
            axes[0, i].set_title(f"{lab} - train {DATASET_NAMES[tr]}")
            axes[1, i].set_title(f"validation loss (plain CE) - train {DATASET_NAMES[tr]}")
            for ax in axes[:, i]:
                ax.set_xlabel("epoch")
                ax.legend(loc="best")
        fig.suptitle(f"{task.upper()}: learning curves v1 vs v2", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
        fig.tight_layout()
        _save(fig, fdir / f"{task}_val_curves_v1_vs_v2.png")

    # overfitting gap at the selected checkpoint
    gap_rows = []
    for (m, task, tr), h in hist.items():
        k = "macro_f1" if task == "asc" else "f1"
        best = max(h, key=lambda r: r[f"val_{k}"])
        gap_rows.append({"model": m, "task": task, "train": tr, "best_epoch": best["epoch"],
                         "train_metric": best[f"train_{k}"], "val_metric": best[f"val_{k}"],
                         "gap": best[f"train_{k}"] - best[f"val_{k}"]})
    gap = pd.DataFrame(gap_rows).sort_values(["task", "train", "model"])
    gap.to_csv(fdir / "generalisation_gap.csv", index=False)

    def md(df, cols):
        fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df[cols].itertuples(index=False)]
        return "\n".join(out)

    def table(task, metric):
        sub = imp[(imp.task == task) & (imp.metric == metric)].copy()
        cols = ["train", "test", "setting", "v1 RoBERTa-base"]
        for m in news:
            sub[f"Δ {m}"] = sub[f"delta {m}"].map(lambda v: f"{v:+.4f}")
            cols += [m, f"Δ {m}"]
        return md(sub, cols)

    lines = ["# v2 improvements over v1", "",
             "Reference = v1 RoBERTa-base (the best v1 model). Δ = v2 − v1 (absolute points on the 0–1 scale).", "",
             "## What changed and why", "",
             "| v1 observation | v2 change |", "|---|---|",
             "| Train macro-F1 → 0.94–0.98 while val plateaus at epoch ~2 and val loss rises (over-fitting) | label smoothing 0.1, FGM adversarial training on embeddings (ε=0.5), layer-wise LR decay so lower layers move less |",
             "| [CLS]-only head cannot tell which aspect it is judging in multi-aspect sentences (MAMS: every sentence) | aspect wrapped with `#` markers in the sentence + head on [CLS] ⊕ mean(aspect sub-word states) |",
             "| Val F1 jumps ±3 pts between checks on SemEval's 221-aspect val split | 4 validation checks per epoch, batch 16 (more updates), patience in epochs |",
             "| ATE decoding could emit invalid `O → I` sequences | BIO-constrained Viterbi decoding |",
             "| Base-size encoder | RoBERTa-large (bf16, gradient checkpointing to fit the 8 GB GPU) |", "",
             "## ASC macro-F1", "", table("ASC", "macro_f1"), "",
             "## ASC accuracy", "", table("ASC", "accuracy"), "",
             "## ASC neutral-class F1 (weakest class in v1)", "", table("ASC", "neutral_f1"), "",
             "## ATE span F1", "", table("ATE", "f1"), "",
             "## End-to-end ATE → ASC pair F1", "", table("End-to-end", "f1"), "",
             "## Generalisation gap at the selected checkpoint (train − val main metric)", "",
             md(gap, ["model", "task", "train", "best_epoch", "train_metric", "val_metric", "gap"]), "",
             "Figures: `improvements/v1_vs_v2_*.png`, `improvements/*_val_curves_v1_vs_v2.png`; "
             "per-run training curves: `transformers/<task>/<run>/training_curves.png`; full tables: `summary.md`.", ""]
    (V2_DIR / "IMPROVEMENTS.md").write_text("\n".join(lines), encoding="utf-8")
    log("\n".join(lines))


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--tasks", nargs="+", default=["ate", "asc"], choices=["ate", "asc"])
    ap.add_argument("--train-on", nargs="+", default=DSETS, choices=DSETS)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke test (1 epoch, 10%% data) -> results_v2_quick/")
    ap.add_argument("--report-only", action="store_true", help="no training: rebuild reports from finished runs only")
    args = ap.parse_args()
    args.epochs, args.batch_size = None, None

    global V2_DIR
    if args.quick:
        V2_DIR = ROOT / "results_v2_quick"
        args.epochs, args.fresh = 1, True
    run_all.RESULTS_DIR = V2_DIR
    log = run_all.Logger(V2_DIR / "run_log.txt")
    log(f"command: {' '.join(sys.argv)}")
    t0 = time.time()

    data = load_datasets()
    if args.quick:
        data = run_all.subsample(data, 0.1)

    # baselines are re-run (deterministic, ~1 min) so the v2 report is self-contained
    asc_rows, ate_rows, e2e_rows, asc_preds = run_all.run_baselines(data, V2_DIR / "baselines", log)
    v1 = load_v1()
    if not args.quick:
        asc_rows += v1[0]
        ate_rows += v1[1]
        e2e_rows += v1[2]
        asc_preds.update(v1[3])
    hist = dict(v1[4]) if not args.quick else {}

    train_times = {}
    for tr in args.train_on:
        for vname in args.variants:
            v = VARIANTS[vname]
            ate_res = None
            done = lambda t: (V2_DIR / "transformers" / t / f"{vname}_train-{tr}" / "test_results.json").exists()
            if args.report_only and not (done("ate") and done("asc")):
                log(f"[report-only] skipping unfinished {vname}/{tr}")
                continue
            if "ate" in args.tasks:
                ate_res = run_all.run_transformer("ate", v["model_key"], tr, data, args, log, hp=v["ate"],
                                                  label=v["label"], out_root=V2_DIR / "transformers", run_name=vname)
                train_times[f"ate/{vname}/{tr}"] = round(ate_res["train_time_s"], 1)
                for te, m in ate_res["test"].items():
                    ate_rows.append({"model": v["label"], "train": tr, "test": te, **m})
                hist[(v["label"], "ate", tr)] = json.load(open(V2_DIR / "transformers" / "ate" / f"{vname}_train-{tr}" / "history.json"))["history"]
            if "asc" in args.tasks:
                r = run_all.run_transformer("asc", v["model_key"], tr, data, args, log, hp=v["asc"], label=v["label"],
                                            out_root=V2_DIR / "transformers", run_name=vname,
                                            ate_spans=ate_res["pred"] if ate_res else None)
                train_times[f"asc/{vname}/{tr}"] = round(r["train_time_s"], 1)
                for te, m in r["test"].items():
                    asc_rows.append({"model": v["label"], "train": tr, "test": te, **m})
                    asc_preds[(v["label"], tr, te)] = r["pred"][te]
                for te, j in r.get("e2e", {}).items():
                    e2e_rows.append({"model": v["label"], "train": tr, "test": te, **j})
                hist[(v["label"], "asc", tr)] = json.load(open(V2_DIR / "transformers" / "asc" / f"{vname}_train-{tr}" / "history.json"))["history"]

    run_all.report(asc_rows, ate_rows, e2e_rows, asc_preds, data, log, train_times, title="ABSA results summary - v2")
    asc, ate, e2e = pd.DataFrame(asc_rows), pd.DataFrame(ate_rows), pd.DataFrame(e2e_rows)
    if not args.quick:
        improvement_report(asc, ate, e2e, hist, log)
    log(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min - results in {V2_DIR}")


if __name__ == "__main__":
    main()
