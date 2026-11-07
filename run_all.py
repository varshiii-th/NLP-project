"""End-to-end ABSA experiment runner.

Stages
  1. data      download MAMS, parse + BIO-tag both corpora in parallel, cache
  2. eda       dataset statistics & figures
  3. baseline  TF-IDF + LR / SVM (ASC), feature-based LR tagger (ATE)
  4. transformers  BERT & RoBERTa for ATE and ASC, trained on the GPU on each corpus
  5. evaluate  4 train/test configurations per model (2 in-domain + 2 cross-dataset),
               end-to-end (ATE -> ASC) pair F1, confusion matrices, error analysis, summary report

Usage
  python run_all.py                     # everything (resumes finished runs)
  python run_all.py --quick             # 1-epoch smoke test on a data subset
  python run_all.py --models roberta --tasks asc --fresh
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from absa.config import DATASET_NAMES, DEFAULT_HPARAMS, POLARITIES, RESULTS_DIR, TrainConfig
from absa.data import load_datasets, predicted_aspect_frame

DSETS = ["semeval", "mams"]


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def subsample(data, frac):
    """--quick: shrink every split (sentence level) for a fast smoke test."""
    from absa.data import to_aspect_frame
    out = {}
    for n, splits in data.items():
        out[n] = {}
        for s, d in splits.items():
            sents = d["sents"][: max(20, int(len(d["sents"]) * frac))]
            out[n][s] = {"sents": sents, "asc": to_aspect_frame(sents)}
    return out


def pairs_from(frame: pd.DataFrame, pred, n_sents: int) -> list[set]:
    pairs = [set() for _ in range(n_sents)]
    for (_, r), p in zip(frame.iterrows(), pred):
        pairs[r.sent_idx].add((tuple(r.tok_span), POLARITIES[int(p)]))
    return pairs


# =========================================================================== baselines
def run_baselines(data, out: Path, log) -> tuple[list, list, list, dict]:
    from absa.baselines import ASCBaseline, ATEBaseline
    from absa.metrics import joint_prf
    from absa.plotting import plot_confusion_matrix, plot_validation_curve

    asc_rows, ate_rows, e2e_rows, preds = [], [], [], {}
    for tr in DSETS:
        t0 = time.time()
        ate = ATEBaseline().fit(data[tr]["train"]["sents"], data[tr]["val"]["sents"])
        log(f"[baseline] ATE LR tagger trained on {tr} (best C={ate.best_c}) in {time.time() - t0:.1f}s")
        plot_validation_curve(ate.curve["C"], {"validation span F1": ate.curve["val_f1"]},
                              f"ATE LR tagger - C sweep (train: {DATASET_NAMES[tr]})", out / f"ate_lr_{tr}_c_sweep.png")
        ate_spans = {}
        for te in DSETS:
            m, sp = ate.evaluate(data[te]["test"]["sents"])
            ate_spans[te] = sp
            ate_rows.append({"model": "TF-IDF+LR", "task": "ate", "train": tr, "test": te, **m})
            log(f"  ATE  TF-IDF+LR  {tr}->{te}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

        for kind in ("lr", "svm"):
            t0 = time.time()
            clf = ASCBaseline(kind).fit(data[tr]["train"]["asc"], data[tr]["val"]["asc"])
            log(f"[baseline] ASC {clf.name} trained on {tr} (best C={clf.best_c}) in {time.time() - t0:.1f}s")
            plot_validation_curve(clf.curve["C"], {"train macro-F1": clf.curve["train_macro_f1"],
                                                   "validation macro-F1": clf.curve["val_macro_f1"]},
                                  f"{clf.name} - C sweep (train: {DATASET_NAMES[tr]})", out / f"asc_{kind}_{tr}_c_sweep.png")
            for te in DSETS:
                m, p = clf.evaluate(data[te]["test"]["asc"])
                preds[(clf.name, tr, te)] = p
                asc_rows.append({"model": clf.name, "train": tr, "test": te, **m})
                plot_confusion_matrix(m["confusion_matrix"], POLARITIES, f"{clf.name} trained on {DATASET_NAMES[tr]}"
                                      f"\ntest: {DATASET_NAMES[te]}", out / f"confusion_{kind}_{tr}_to_{te}.png")
                log(f"  ASC  {clf.name:10s} {tr}->{te}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
                # end-to-end: baseline ATE spans -> baseline ASC
                sents = data[te]["test"]["sents"]
                fr = predicted_aspect_frame(sents, ate_spans[te])
                j = joint_prf(sents, pairs_from(fr, clf.predict(fr), len(sents)))
                e2e_rows.append({"model": clf.name, "train": tr, "test": te, **j})
    return asc_rows, ate_rows, e2e_rows, preds


# =========================================================================== transformers
def run_transformer(task, model_key, tr, data, args, log, ate_spans=None, hp=None, label=None,
                    out_root=None, run_name=None) -> dict:
    from absa.metrics import joint_prf
    from absa.transformer_trainer import MODEL_LABEL, Trainer

    out = (out_root or RESULTS_DIR / "transformers") / task / f"{run_name or model_key}_train-{tr}"
    res_file = out / "test_results.json"
    if res_file.exists() and not args.fresh:
        log(f"[skip] {task}/{model_key}/{tr} already finished (use --fresh to retrain)")
        return json.load(open(res_file))

    hp = dict(hp if hp is not None else DEFAULT_HPARAMS[task])
    label = label or MODEL_LABEL[model_key]
    if args.epochs:
        hp["epochs"] = args.epochs
    if args.batch_size:
        hp["batch_size"] = args.batch_size
    cfg = TrainConfig(task=task, model_key=model_key, **hp)
    title = f"{label} {task.upper()} - train: {DATASET_NAMES[tr]}"
    log(f"\n=== {title} | epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size}")
    trainer = Trainer(cfg, out, log)
    key = "asc" if task == "asc" else "sents"
    info = trainer.fit(data[tr]["train"][key], data[tr]["val"][key], title)

    result = {"model": label, "task": task, "train": tr, "best_val": info["best"],
              "train_time_s": info["train_time_s"], "test": {}, "pred": {}, "e2e": {}}
    for te in DSETS:
        m, pred = trainer.test(data[te]["test"][key], te, f"{label} trained on {DATASET_NAMES[tr]}")
        if task == "asc":
            result["pred"][te] = pred[0].tolist()
            log(f"  TEST {tr}->{te}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
            if ate_spans and te in ate_spans:  # end-to-end with this model family's own extracted aspects
                sents = data[te]["test"]["sents"]
                fr = predicted_aspect_frame(sents, [list(map(tuple, s)) for s in ate_spans[te]])
                j = joint_prf(sents, pairs_from(fr, trainer.predict_asc(fr), len(sents)))
                result["e2e"][te] = j
                log(f"  E2E  {tr}->{te}: pair P={j['precision']:.4f} R={j['recall']:.4f} F1={j['f1']:.4f}")
        else:
            from absa.metrics import bio_to_spans
            result["pred"][te] = [bio_to_spans(t) for t in pred]
            log(f"  TEST {tr}->{te}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        result["test"][te] = m
    if args.save_models:
        trainer.save()
    trainer.close()
    jdump(result, res_file)
    return result


# =========================================================================== reporting
def report(asc_rows, ate_rows, e2e_rows, asc_preds, data, log, train_times, title="ABSA results summary"):
    from absa.error_analysis import PHENOMENA, accuracy_by_phenomenon, misclassified, tag_phenomena
    from absa.plotting import MODEL_COLORS, grouped_bars, plot_cross_heatmaps

    tdir, fdir = RESULTS_DIR / "tables", RESULTS_DIR / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    keep = ["model", "train", "test", "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "negative_f1", "neutral_f1", "positive_f1"]
    asc = pd.DataFrame(asc_rows)[keep]
    ate = pd.DataFrame(ate_rows)[["model", "train", "test", "precision", "recall", "f1", "n_gold", "n_pred"]]
    e2e = pd.DataFrame(e2e_rows)
    for df in (asc, ate, e2e):
        df["setting"] = np.where(df.train == df.test, "in-domain", "cross-dataset")
    asc.to_csv(tdir / "asc_results.csv", index=False)
    ate.to_csv(tdir / "ate_results.csv", index=False)
    e2e.to_csv(tdir / "end2end_results.csv", index=False)

    asc_models = list(dict.fromkeys(asc.model))  # order of first appearance
    ate_models = list(dict.fromkeys(ate.model))
    plot_cross_heatmaps(asc, "macro_f1", asc_models, "ASC: in-domain vs cross-dataset", fdir / "asc_cross_dataset_macro_f1.png")
    plot_cross_heatmaps(asc, "accuracy", asc_models, "ASC: in-domain vs cross-dataset", fdir / "asc_cross_dataset_accuracy.png", "Accuracy")
    plot_cross_heatmaps(ate, "f1", ate_models, "ATE: in-domain vs cross-dataset", fdir / "ate_cross_dataset_f1.png", "span F1")
    if len(e2e):
        e2e_models = [m for m in asc_models if m in set(e2e.model)]
        plot_cross_heatmaps(e2e, "f1", e2e_models, "End-to-end ATE->ASC", fdir / "end2end_pair_f1.png", "aspect-polarity pair F1")

    cfgs = [(a, b) for a in DSETS for b in DSETS]
    cat = [f"{DATASET_NAMES[a].split()[0]} -> {DATASET_NAMES[b].split()[0]}" for a, b in cfgs]

    def series(df, models, col):
        return {m: [float(df[(df.model == m) & (df.train == a) & (df.test == b)][col].iloc[0])
                    if len(df[(df.model == m) & (df.train == a) & (df.test == b)]) else np.nan for a, b in cfgs]
                for m in models}
    grouped_bars(cat, series(asc, asc_models, "macro_f1"), "ASC macro-F1 by train -> test configuration", "macro-F1",
                 fdir / "asc_model_comparison_macro_f1.png", colors=MODEL_COLORS)
    grouped_bars(cat, series(asc, asc_models, "accuracy"), "ASC accuracy by train -> test configuration", "accuracy",
                 fdir / "asc_model_comparison_accuracy.png", colors=MODEL_COLORS)
    grouped_bars(cat, series(ate, ate_models, "f1"), "ATE span F1 by train -> test configuration", "span F1",
                 fdir / "ate_model_comparison_f1.png", colors=MODEL_COLORS)
    # per-class F1 in-domain
    for d in DSETS:
        sub = asc[(asc.train == d) & (asc.test == d)]
        grouped_bars(POLARITIES, {m: [float(sub[sub.model == m][f"{p}_f1"].iloc[0]) for p in POLARITIES] for m in asc_models},
                     f"ASC per-class F1 (in-domain, {DATASET_NAMES[d]})", "F1", fdir / f"asc_per_class_f1_{d}.png",
                     colors=MODEL_COLORS)

    # error analysis
    edir = RESULTS_DIR / "error_analysis"
    edir.mkdir(parents=True, exist_ok=True)
    tagged = {d: tag_phenomena(data[d]["test"]["asc"]) for d in DSETS}
    ea_rows = []
    for (m, tr, te), pred in asc_preds.items():
        ea_rows.append({"model": m, "train": tr, "test": te, **accuracy_by_phenomenon(tagged[te], np.asarray(pred))})
        if not m.startswith("TF-IDF"):
            misclassified(tagged[te], np.asarray(pred)).to_csv(edir / f"errors_{m}_{tr}_to_{te}.csv", index=False)
    ea = pd.DataFrame(ea_rows)
    ea.to_csv(edir / "accuracy_by_phenomenon.csv", index=False)
    for tr in DSETS:
        for te in DSETS:
            sub = ea[(ea.train == tr) & (ea.test == te)]
            if not len(sub):
                continue
            ns = sub.iloc[0]
            cats = ["all"] + [f"{p}\n(n={ns[p + '_n']})" for p in PHENOMENA]
            grouped_bars(cats, {m: [float(sub[sub.model == m][c].iloc[0]) for c in ["all"] + PHENOMENA] for m in asc_models},
                         f"ASC accuracy by linguistic phenomenon - train {DATASET_NAMES[tr]}, test {DATASET_NAMES[te]}",
                         "accuracy", edir / f"phenomena_{tr}_to_{te}.png", colors=MODEL_COLORS, figsize=(13, 4.2))
    phen_counts = {d: {p: int(tagged[d][p].sum()) for p in PHENOMENA} for d in DSETS}

    # markdown summary
    def md(df, cols):  # dependency-free markdown table
        fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
        rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df[cols].itertuples(index=False)]
        return "\n".join(rows)
    lines = [f"# {title}", "",
             "Settings: in-domain = trained & tested on the same corpus; cross-dataset = transfer.", "",
             "## Aspect-level sentiment classification (test sets)", "",
             md(asc.sort_values(["train", "test", "model"]), ["model", "train", "test", "setting", "accuracy", "macro_precision", "macro_recall", "macro_f1"]),
             "", "## Aspect term extraction (exact-match spans, test sets)", "",
             md(ate.sort_values(["train", "test", "model"]), ["model", "train", "test", "setting", "precision", "recall", "f1"]),
             "", "## End-to-end ATE -> ASC (aspect-polarity pair F1)", "",
             md(e2e.sort_values(["train", "test", "model"]), ["model", "train", "test", "setting", "precision", "recall", "f1"]) if len(e2e) else "n/a",
             "", "## ASC accuracy by linguistic phenomenon", "",
             md(ea, ["model", "train", "test", "all"] + PHENOMENA),
             "", f"Phenomenon counts in the test sets: {json.dumps(phen_counts)}", "",
             "## Transformer training time (s)", "", json.dumps(train_times, indent=1), ""]
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines[3:]))


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["bert", "roberta"], choices=["bert", "roberta"])
    ap.add_argument("--tasks", nargs="+", default=["ate", "asc"], choices=["ate", "asc"])
    ap.add_argument("--train-on", nargs="+", default=DSETS, choices=DSETS)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for all transformer runs")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="retrain even if results exist")
    ap.add_argument("--save-models", action="store_true", help="save best checkpoints (~0.5 GB each)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 epoch on 10%% of the data")
    ap.add_argument("--no-cache", action="store_true", help="re-run preprocessing")
    args = ap.parse_args()

    global RESULTS_DIR
    if args.quick:
        import absa.config as C
        RESULTS_DIR = C.RESULTS_DIR = RESULTS_DIR.parent / "results_quick"
        args.epochs, args.fresh = 1, True
    log = Logger(RESULTS_DIR / "run_log.txt")
    log(f"command: {' '.join(sys.argv)}")

    t0 = time.time()
    data = load_datasets(use_cache=not args.no_cache)
    log(f"[data] ready in {time.time() - t0:.1f}s")
    if args.quick:
        data = subsample(data, 0.1)

    from absa.eda import run_eda
    run_eda(data, RESULTS_DIR / "eda")

    asc_rows, ate_rows, e2e_rows, asc_preds = [], [], [], {}
    if not args.skip_baselines:
        a, b, c, p = run_baselines(data, RESULTS_DIR / "baselines", log)
        asc_rows += a
        ate_rows += b
        e2e_rows += c
        asc_preds.update(p)

    train_times = {}
    for tr in args.train_on:
        for mk in args.models:
            ate_res = None
            if "ate" in args.tasks:
                ate_res = run_transformer("ate", mk, tr, data, args, log)
                train_times[f"ate/{mk}/{tr}"] = round(ate_res["train_time_s"], 1)
                for te, m in ate_res["test"].items():
                    ate_rows.append({"model": ate_res["model"], "task": "ate", "train": tr, "test": te, **m})
            if "asc" in args.tasks:
                r = run_transformer("asc", mk, tr, data, args, log, ate_spans=ate_res["pred"] if ate_res else None)
                train_times[f"asc/{mk}/{tr}"] = round(r["train_time_s"], 1)
                for te, m in r["test"].items():
                    asc_rows.append({"model": r["model"], "train": tr, "test": te, **m})
                    asc_preds[(r["model"], tr, te)] = r["pred"][te]
                for te, j in r.get("e2e", {}).items():
                    e2e_rows.append({"model": r["model"], "train": tr, "test": te, **j})

    if asc_rows and ate_rows:
        report(asc_rows, ate_rows, e2e_rows, asc_preds, data, log, train_times)
    log(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min - results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
