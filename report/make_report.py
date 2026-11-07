"""Builds the project report from the experiment outputs, in two formats from ONE source text:

    python report/make_report.py

  report/ABSA_Report.docx  ACL-style two-column Word document (Times, A4)  -> exported to PDF with Word
  report/ABSA_Report.tex   the same report as a single LaTeX file (ACL 2023 style, needs acl2023.sty + figures/)

All numbers come from results/ (v1) and results_v2/ (v2). Wherever the text states a direction
("improves", "drops", "within one point"), an assertion checks it against the data first.

Inline markup used in the text below:  **bold**  *italic*  [@key] = (Author, Year)  [@@key] = Author (Year)
{tab:x} / {fig:x} = cross-references,  @@name@@ = number from the results.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R1, R2, OUT = ROOT / "results", ROOT / "results_v2", ROOT / "report"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

V1B, V1R, V2B, V2L = "BERT-base v1", "RoBERTa-base v1", "RoBERTa-base v2", "RoBERTa-large v2"
LR, SVM = "TF-IDF+LR", "TF-IDF+SVM"
CFG = [("semeval", "semeval"), ("mams", "mams"), ("semeval", "mams"), ("mams", "semeval")]
CFG_K = ["ll", "mm", "lm", "ml"]
CFG_HDR = ["L→L", "M→M", "L→M", "M→L"]

asc = pd.read_csv(R2 / "tables" / "asc_results.csv")
ate = pd.read_csv(R2 / "tables" / "ate_results.csv")
e2e = pd.read_csv(R2 / "tables" / "end2end_results.csv")
ea = pd.read_csv(R2 / "error_analysis" / "accuracy_by_phenomenon.csv")
ds = pd.read_csv(R1 / "eda" / "dataset_summary.csv")
st = pd.read_csv(R1 / "eda" / "dataset_statistics.csv").set_index("dataset")
gap = pd.read_csv(R2 / "improvements" / "generalisation_gap.csv")
vocab = (R1 / "eda" / "vocab_overlap.txt").read_text()


def v(df, m, a, b, col):
    s = df[(df.model == m) & (df.train == a) & (df.test == b)]
    return float(s[col].iloc[0]) if len(s) else float("nan")


def p(x, d=1):
    return f"{100 * x:.{d}f}"


# =========================================================================== numbers
N = {}
sem, mam = ds[ds.dataset == "semeval"].set_index("split"), ds[ds.dataset == "mams"].set_index("split")
for d, t in [("sem", sem), ("mam", mam)]:
    for sp in ("train", "val", "test"):
        N[f"{d}_{sp}_sent"] = f"{int(t.loc[sp, 'sentences']):,}"
        N[f"{d}_{sp}_asp"] = f"{int(t.loc[sp, 'asc_instances']):,}"
N["sem_tr_total"] = f"{int(sem.loc['train', 'sentences'] + sem.loc['val', 'sentences']):,}"
N["sem_multi_asp"] = p(st.loc["semeval", "share_sentences_multi_aspect"], 0)
N["sem_multi_pol"] = p(st.loc["semeval", "share_sentences_multi_polarity"], 0)
N["sem_len"] = f"{st.loc['semeval', 'mean_tokens']:.1f}"
N["mam_len"] = f"{st.loc['mams', 'mean_tokens']:.1f}"
N["sem_aps"] = f"{st.loc['semeval', 'mean_aspects_per_sentence(with>=1)']:.1f}"
N["mam_aps"] = f"{st.loc['mams', 'mean_aspects_per_sentence(with>=1)']:.1f}"
m = re.search(r"SemEval=(\d+), MAMS=(\d+), shared=(\d+) \(Jaccard=([\d.]+)\)", vocab)
N["voc_sem"], N["voc_mam"], N["voc_shared"], N["jaccard"] = m.groups()
sem_tr = ds[(ds.dataset == "semeval") & (ds.split == "train")].iloc[0]
mam_tr = ds[(ds.dataset == "mams") & (ds.split == "train")].iloc[0]
N["sem_neu_share"] = p(sem_tr.neutral / sem_tr.asc_instances, 0)
N["mam_neu_share"] = p(mam_tr.neutral / mam_tr.asc_instances, 0)
N["sem_pos_share"] = p(sem_tr.positive / sem_tr.asc_instances, 0)

f1 = lambda mdl, a, b: v(asc, mdl, a, b, "macro_f1")  # noqa: E731
acc = lambda mdl, a, b: v(asc, mdl, a, b, "accuracy")  # noqa: E731
TAGS = [(LR, "lr"), (SVM, "svm"), (V1B, "v1b"), (V1R, "v1r"), (V2B, "v2b"), (V2L, "v2l")]
for (a, b), k in zip(CFG, CFG_K):
    best = asc[(asc.train == a) & (asc.test == b)].sort_values("macro_f1").iloc[-1]
    N[f"best_{k}"], N[f"bestf1_{k}"], N[f"bestacc_{k}"] = best.model, p(best.macro_f1), p(best.accuracy)
    for mdl, tag in TAGS:
        N[f"{tag}_{k}"] = p(f1(mdl, a, b))
        N[f"{tag}_acc_{k}"] = p(acc(mdl, a, b))
        N[f"ate_{tag}_{k}"] = p(v(ate, mdl, a, b, "f1"))
        N[f"e2e_{tag}_{k}"] = p(v(e2e, mdl, a, b, "f1"))
    for mdl, tag in [(V2B, "v2b"), (V2L, "v2l")]:
        N[f"dd_{tag}_{k}"] = f"{abs(100 * (f1(mdl, a, b) - f1(V1R, a, b))):.1f}"
        N[f"dd_ate_{tag}_{k}"] = f"{abs(100 * (v(ate, mdl, a, b, 'f1') - v(ate, V1R, a, b, 'f1'))):.1f}"
        N[f"dd_e2e_{tag}_{k}"] = f"{abs(100 * (v(e2e, mdl, a, b, 'f1') - v(e2e, V1R, a, b, 'f1'))):.1f}"


def delta(df, mdl, k, col="macro_f1"):
    a, b = CFG[CFG_K.index(k)]
    return v(df, mdl, a, b, col) - v(df, V1R, a, b, col)


# directions stated in the text -> checked here
for k in CFG_K:
    assert delta(asc, V2L, k) > 0, k                      # large v2 beats v1 in all 4 settings
    assert delta(e2e, V2B, k, "f1") > 0 and delta(e2e, V2L, k, "f1") > 0, k   # e2e improves everywhere
assert delta(asc, V2B, "ll") > 0 and delta(asc, V2B, "mm") > 0 and delta(asc, V2B, "lm") > 0
assert delta(asc, V2B, "ml") < 0
assert delta(ate, V2B, "ll", "f1") < 0 and delta(ate, V2B, "mm", "f1") > 0 and delta(ate, V2B, "lm", "f1") < 0
assert abs(delta(ate, V2L, "ll", "f1")) < 0.01 and abs(delta(ate, V2L, "mm", "f1")) < 0.01
assert f1(V1R, "semeval", "semeval") > f1(V1B, "semeval", "semeval")
for (a, b) in CFG:
    assert min(f1(V1B, a, b), f1(V1R, a, b)) > max(f1(LR, a, b), f1(SVM, a, b))   # Transformers beat TF-IDF everywhere

N["gain_tfidf_ll"] = f"{100 * (f1(V1R, 'semeval', 'semeval') - f1(LR, 'semeval', 'semeval')):.1f}"
N["gain_tfidf_mm"] = f"{100 * (f1(V1R, 'mams', 'mams') - f1(LR, 'mams', 'mams')):.1f}"
for tag, mdl in [("v1r", V1R), ("v2l", V2L), ("v2b", V2B)]:
    for k, (a, b) in zip(CFG_K, CFG):
        N[f"neu_{tag}_{k}"] = p(v(asc, mdl, a, b, "neutral_f1"))
ate_lm = ate[(ate.model == V1R) & (ate.train == "semeval") & (ate.test == "mams")].iloc[0]
N["ate_lm_p"], N["ate_lm_r"] = p(ate_lm.precision, 0), p(ate_lm.recall, 0)
ate_ml = ate[(ate.model == V1R) & (ate.train == "mams") & (ate.test == "semeval")].iloc[0]
N["ate_ml_p"], N["ate_ml_r"] = p(ate_ml.precision, 0), p(ate_ml.recall, 0)
N["e2e_cap"] = p(max(v(e2e, mm, a, b, "f1") for mm in set(e2e.model) for a, b in CFG[2:]) + 0.005, 0)

gg = gap[gap.task == "asc"].set_index(["model", "train"])
for mdl, tag in [(V1R, "v1r"), (V2B, "v2b"), (V2L, "v2l")]:
    for d, k in [("semeval", "ll"), ("mams", "mm")]:
        N[f"gap_{tag}_{k}"] = p(gg.loc[(mdl, d), "gap"])
        N[f"ep_{tag}_{k}"] = f"{gg.loc[(mdl, d), 'best_epoch']:.1f}"
assert gg.loc[(V2B, "mams"), "gap"] < gg.loc[(V1R, "mams"), "gap"] and gg.loc[(V2L, "mams"), "gap"] < gg.loc[(V1R, "mams"), "gap"]
assert gg.loc[(V2B, "semeval"), "gap"] > gg.loc[(V1R, "semeval"), "gap"]

h1 = json.load(open(R1 / "transformers" / "asc" / "roberta_train-mams" / "history.json"))["history"]
N["v1_curve_train_end"] = p(h1[-1]["train_macro_f1"], 0)
N["v1_curve_val_best"] = p(max(r["val_macro_f1"] for r in h1))
N["v1_curve_valloss_min_ep"] = f"{min(h1, key=lambda r: r['val_loss'])['epoch']:.1f}"
N["v1_curve_valloss_min"] = f"{min(r['val_loss'] for r in h1):.2f}"
N["v1_curve_valloss_end"] = f"{h1[-1]['val_loss']:.2f}"
hs = json.load(open(R1 / "transformers" / "asc" / "roberta_train-semeval" / "history.json"))["history"]
vals = [r["val_macro_f1"] for r in hs[2:]]
N["sem_val_jump"] = p(max(abs(a - b) for a, b in zip(vals, vals[1:])))
times = {}
for f in list((R1 / "transformers").glob("*/*/history.json")) + list((R2 / "transformers").glob("*/*/history.json")):
    times[f"{f.parent.parent.name}/{f.parent.name}/{'v2' if R2 in f.parents else 'v1'}"] = json.load(open(f))["train_time_s"]
N["t_v1_total"] = f"{sum(t for k, t in times.items() if k.endswith('v1')) / 60:.0f}"
N["t_v2_total"] = f"{sum(t for k, t in times.items() if k.endswith('v2')) / 60:.0f}"
N["t_large_mams"] = f"{times['asc/roberta-large-v2_train-mams/v2'] / 60:.0f}"

r_ll = ea[(ea.model == V2L) & (ea.train == "semeval") & (ea.test == "semeval")].iloc[0]
r_mm = ea[(ea.model == V2L) & (ea.train == "mams") & (ea.test == "mams")].iloc[0]
lr_ll = ea[(ea.model == LR) & (ea.train == "semeval") & (ea.test == "semeval")].iloc[0]
v1_ll = ea[(ea.model == V1R) & (ea.train == "semeval") & (ea.test == "semeval")].iloc[0]
PH = ["negation", "contrast", "multi_aspect", "multi_polarity", "implicit", "long_range"]
for tag, r in [("ll", r_ll), ("mm", r_mm)]:
    N[f"ph_{tag}_all"] = p(r["all"])
    for c in PH:
        N[f"ph_{tag}_{c}"] = p(r[c])
N["ph_v1_ll_mpol"], N["v1_ll_all"] = p(v1_ll.multi_polarity), p(v1_ll["all"])
N["ph_lr_ll_mpol"], N["ph_lr_ll_neg"] = p(lr_ll.multi_polarity), p(lr_ll.negation)
N["mpol_gain"] = f"{100 * (r_ll.multi_polarity - v1_ll.multi_polarity):.1f}"
N["all_gain"] = f"{100 * (r_ll['all'] - v1_ll['all']):.1f}"
N["n_mpol_ll"] = str(int(r_ll.multi_polarity_n))
assert min(PH, key=lambda c: r_ll[c]) == "negation"                   # "negation is now the weakest group"
assert min(PH, key=lambda c: v1_ll[c]) == "multi_polarity" and min(PH, key=lambda c: lr_ll[c]) == "multi_polarity"
cm = np.array(json.load(open(R2 / "transformers" / "asc" / "roberta-large-v2_train-semeval" / "test_results.json"))
              ["test"]["semeval"]["confusion_matrix"])
off = cm.sum() - np.trace(cm)
N["cm_ll_neu_share"] = p((cm[1].sum() - cm[1, 1] + cm[:, 1].sum() - cm[1, 1]) / off, 0)
N["cm_ll_err"] = str(int(off))

# misclassified examples of the best Laptop model
from absa.config import POLARITIES  # noqa: E402
from absa.data import load_datasets  # noqa: E402
from absa.error_analysis import _tag_row  # noqa: E402

data = load_datasets()
pred = json.load(open(R2 / "transformers" / "asc" / "roberta-large-v2_train-semeval" / "test_results.json"))["pred"]["semeval"]
_t = data["semeval"]["test"]["asc"].reset_index(drop=True)
tg = pd.concat([_t, pd.DataFrame([_tag_row((r.text, r["from"], r.to, r.n_aspects, r.n_polarities))
                                  for _, r in _t.iterrows()])], axis=1)
tg["pred"] = [POLARITIES[i] for i in pred]
wrong = tg[(tg.pred != tg.polarity) & (tg.text.str.len() < 125)]
examples = []
for ph in ["negation", "contrast", "multi_polarity", "implicit"]:
    cand = wrong[wrong[ph] & ~wrong.sid.isin([e.sid for e in examples])]
    if len(cand):
        examples.append(cand.iloc[0])
assert examples[0].text.startswith("Did not enjoy") and "OSX" in examples[1].text   # referenced in the text

# =========================================================================== tables (format-neutral)
# cell types: float -> x100, 1 decimal;  ("D", x) -> signed delta;  str -> as is.  Row "MID" = rule, ("GROUP", s) = label row
TABLES = {}


def model_short(k, ver):
    return {"bert": "BERT-b", "roberta": "RoBERTa-b", "roberta-large": "RoBERTa-L"}[k] + f" {ver}"


TABLES["datasets"] = dict(
    header=["Corpus", "Split", "Sentences", "Aspects", "Pos", "Neg", "Neu"],
    rows=[[{"semeval": "Laptops", "mams": "MAMS"}[r.dataset], r.split, f"{int(r.sentences):,}", f"{int(r.asc_instances):,}",
           f"{int(r.positive):,}", f"{int(r.negative):,}", f"{int(r.neutral):,}"] for r in ds.itertuples()],
    caption="Size of the two datasets after preprocessing. “Aspects” counts sentence–aspect pairs labelled positive, "
            "negative or neutral (the SemEval conflict label is removed). The Laptops validation split is a random 10% "
            "of the original training sentences.")
TABLES["datasets"]["rows"].insert(3, "MID")

hp = []
for vdir, ver in [(R1, "v1"), (R2, "v2")]:
    for f in sorted((vdir / "transformers").glob("*/*_train-semeval/history.json")):
        c = json.load(open(f))["config"]
        hp.append([model_short(c["model_key"], ver), c["task"].upper(), str(c["epochs"]), str(c["batch_size"]),
                   f"{c['lr']:g}".replace("e-0", "e-"), f"{c.get('llrd', 1.0):g}", f"{c.get('label_smoothing', 0):g}",
                   f"{c.get('fgm_eps', 0):g}"])
TABLES["hparams"] = dict(header=["Model", "Task", "Ep.", "BS", "LR", "LLRD", "LS", "FGM"], rows=hp, small=True,
                         caption="Fine-tuning settings (b = base, L = large). Ep. is the maximum number of epochs "
                                 "(early stopping usually ends training sooner), BS the batch size, LLRD the layer-wise "
                                 "learning-rate decay, LS label smoothing and FGM the size ε of the adversarial nudge.")

models = list(dict.fromkeys(asc.model))
rows = []
for mdl in models:
    if mdl == V2B:
        rows.append("MID")
    rows.append([mdl] + [f1(mdl, a, b) for a, b in CFG] + [acc(mdl, a, b) for a, b in CFG])
TABLES["asc"] = dict(super=[("", 1), ("Macro-F1", 4), ("Accuracy", 4)], header=["Model"] + CFG_HDR + CFG_HDR,
                     rows=rows, best=range(1, 9), wide=True,
                     caption="Aspect sentiment results on the test sets (%). L = Laptops, M = MAMS, and “L→M” means "
                             "trained on Laptops and tested on MAMS. Rows below the line use the v2 recipe. The best "
                             "value in each column is in bold.")
TABLES["perclass"] = dict(super=[("", 1), ("Laptops", 3), ("MAMS", 3)], header=["Model", "Pos", "Neg", "Neu", "Pos", "Neg", "Neu"],
                          rows=[[mdl] + [v(asc, mdl, d, d, f"{c}_f1") for d in ("semeval", "mams")
                                         for c in ("positive", "negative", "neutral")] for mdl in models],
                          best=range(1, 7), small=True, caption="F1 for each class (%), in-domain.")
ate_models = list(dict.fromkeys(ate.model))
rows = []
for mdl in ate_models:
    if mdl == V2B:
        rows.append("MID")
    rows.append([mdl] + [v(ate, mdl, "semeval", "semeval", c) for c in ("precision", "recall", "f1")]
                + [v(ate, mdl, "mams", "mams", c) for c in ("precision", "recall", "f1")]
                + [v(ate, mdl, "semeval", "mams", "f1"), v(ate, mdl, "mams", "semeval", "f1")])
TABLES["ate"] = dict(super=[("", 1), ("Laptops (in-domain)", 3), ("MAMS (in-domain)", 3), ("Cross-dataset F1", 2)],
                     header=["Model", "P", "R", "F1", "P", "R", "F1", "L→M", "M→L"], rows=rows, best=range(1, 9), wide=True,
                     caption="Aspect term extraction with exact span matching (%). The traditional tagger is listed as "
                             "TF-IDF+LR for short.")
TABLES["e2e"] = dict(header=["Model"] + CFG_HDR, rows=[[mdl] + [v(e2e, mdl, a, b, "f1") for a, b in CFG]
                                                       for mdl in dict.fromkeys(e2e.model)],
                     best=range(1, 5), small=True,
                     caption="Full pipeline (extract, then classify): F1 over (aspect, polarity) pairs (%). A pair "
                             "counts only if both the span and its polarity are right.")
rows = []
for lab, df, col in [("Sentiment macro-F1", asc, "macro_f1"), ("Sentiment accuracy", asc, "accuracy"),
                     ("Extraction span F1", ate, "f1"), ("Full pipeline F1", e2e, "f1")]:
    rows.append(("GROUP", lab))
    for mdl in (V2B, V2L):
        rows.append([mdl.replace(" v2", "")] + [("D", delta(df, mdl, k, col)) for k in CFG_K])
TABLES["delta"] = dict(header=["v2 model"] + CFG_HDR, rows=rows, small=True,
                       caption="Change in points of each v2 model compared with RoBERTa-base v1, our best v1 model. "
                               "Positive numbers mean v2 is better.")
g = gap[(gap.task == "asc") & gap.model.isin([V1B, V1R, V2B, V2L])].copy()
g["o"] = g.model.map({V1B: 0, V1R: 1, V2B: 2, V2L: 3})
g = g.sort_values(["train", "o"])
TABLES["gap"] = dict(header=["Model", "Data", "Epoch", "Train", "Val", "Gap"],
                     rows=[[r.model, {"semeval": "L", "mams": "M"}[r.train], f"{r.best_epoch:.1f}", r.train_metric,
                            r.val_metric, r.gap] for r in g.itertuples()], small=True,
                     caption="Sentiment macro-F1 on the training and validation sets (%) at the checkpoint each run "
                             "kept, and the gap between the two.")
rows = []
for d, lab in [("semeval", "Laptops"), ("mams", "MAMS")]:
    sub = ea[(ea.train == d) & (ea.test == d)]
    n0 = sub.iloc[0]
    rows.append(("GROUP", f"{lab} test set (group sizes: " + ", ".join(str(int(n0[c + '_n'])) for c in PH) + ")"))
    for mdl in models:
        r = sub[sub.model == mdl]
        if len(r):
            rows.append([mdl] + [float(r[c].iloc[0]) for c in ["all"] + PH])
TABLES["phen"] = dict(header=["Model", "All", "Negation", "Contrast", "Multi-asp.", "Multi-pol.", "Implicit", "Long-range"],
                      rows=rows, wide=True, small=True,
                      caption="In-domain sentiment accuracy (%) on groups of test examples that show a given "
                              "phenomenon. In MAMS every sentence has several aspects with different labels, so the "
                              "two middle columns equal “All”.")
TABLES["examples"] = dict(header=["Sentence", "Aspect", "Gold", "Predicted", "Tags"], wide=True, small=True,
                          rows=[[e.text, f"*{e.aspect}*", e.polarity, e.pred,
                                 ", ".join(x.replace("_", "-") for x in PH if e[x])] for e in examples],
                          widths=[3.3, 0.9, 0.6, 0.7, 1.3], align="left",
                          caption="Some mistakes of RoBERTa-large v2 on the Laptops test set.")

# =========================================================================== figures
FIGS_SRC = {
    "polarity_distribution.png": R1 / "eda" / "polarity_distribution.png",
    "aspects_and_lengths.png": R1 / "eda" / "aspects_and_lengths.png",
    "asc_cross_dataset_macro_f1.png": R2 / "figures" / "asc_cross_dataset_macro_f1.png",
    "v1_curves_roberta_asc_mams.png": R1 / "transformers" / "asc" / "roberta_train-mams" / "training_curves.png",
    "v2_curves_large_asc_mams.png": R2 / "transformers" / "asc" / "roberta-large-v2_train-mams" / "training_curves.png",
    "asc_val_curves_v1_vs_v2.png": R2 / "improvements" / "asc_val_curves_v1_vs_v2.png",
    "phenomena_semeval.png": R2 / "error_analysis" / "phenomena_semeval_to_semeval.png",
    "phenomena_mams.png": R2 / "error_analysis" / "phenomena_mams_to_mams.png",
    "confusion_large_semeval.png": R2 / "transformers" / "asc" / "roberta-large-v2_train-semeval" / "confusion_test_semeval.png",
    "confusion_large_mams.png": R2 / "transformers" / "asc" / "roberta-large-v2_train-mams" / "confusion_test_mams.png",
}
for n, src in FIGS_SRC.items():
    shutil.copy(src, FIG / n)

FIGURES = {
    "polarity": dict(files=["polarity_distribution.png"],
                     caption="Share of positive, negative and neutral aspects in each dataset and split."),
    "eda": dict(files=["aspects_and_lengths.png"], wide=True,
                caption="Aspects per sentence, sentence length, and how many sentences have several aspects or "
                        "several polarities."),
    "cross": dict(files=["asc_cross_dataset_macro_f1.png"], wide=True,
                  caption="Sentiment macro-F1 of every model in the four settings. Rows are the training dataset, "
                          "columns the test dataset."),
    "curves": dict(files=["v1_curves_roberta_asc_mams.png", "v2_curves_large_asc_mams.png"], wide=True, stack=True,
                   caption="Training progress for sentiment on MAMS: RoBERTa-base v1 (top) and RoBERTa-large v2 "
                           "(bottom). Blue is the training set, orange the validation set, and the dotted line marks "
                           "the checkpoint that was kept. The faint line is the per-step training loss, which in v2 "
                           "includes label smoothing."),
    "v1v2": dict(files=["asc_val_curves_v1_vs_v2.png"], wide=True, scale=0.8,
                 caption="Validation macro-F1 (top) and validation loss (bottom) while training the sentiment models of "
                         "both rounds, on Laptops (left) and MAMS (right)."),
    "phen": dict(files=["phenomena_semeval.png", "phenomena_mams.png"], wide=True,
                 caption="Sentiment accuracy for each group of test examples, Laptops (left) and MAMS (right)."),
    "confusion": dict(files=["confusion_large_semeval.png", "confusion_large_mams.png"],
                      caption="Confusion matrices of RoBERTa-large v2 on Laptops (left) and MAMS (right)."),
}

# =========================================================================== references
REFS = [
    ("devlin2019bert", "Devlin et al.", "2019", "Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. "
     "BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT*, pages 4171–4186."),
    ("hu2004mining", "Hu and Liu", "2004", "Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In "
     "*Proceedings of the 10th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pages 168–177."),
    ("jiang2019mams", "Jiang et al.", "2019", "Qingnan Jiang, Lei Chen, Ruifeng Xu, Xiang Ao, and Min Yang. 2019. A challenge "
     "dataset and effective models for aspect-based sentiment analysis. In *Proceedings of EMNLP-IJCNLP*, pages 6280–6285."),
    ("liu2012sentiment", "Liu", "2012", "Bing Liu. 2012. *Sentiment Analysis and Opinion Mining*. Morgan & Claypool Publishers."),
    ("liu2015sentiment", "Liu", "2015", "Bing Liu. 2015. *Sentiment Analysis: Mining Opinions, Sentiments, and Emotions*. "
     "Cambridge University Press."),
    ("liu2019roberta", "Liu et al.", "2019", "Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, "
     "Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A robustly optimized BERT pretraining approach. "
     "*arXiv preprint arXiv:1907.11692*."),
    ("medhat2014sentiment", "Medhat et al.", "2014", "Walaa Medhat, Ahmed Hassan, and Hoda Korashy. 2014. Sentiment analysis "
     "algorithms and applications: A survey. *Ain Shams Engineering Journal*, 5(4):1093–1113."),
    ("miyato2017adversarial", "Miyato et al.", "2017", "Takeru Miyato, Andrew M. Dai, and Ian Goodfellow. 2017. Adversarial "
     "training methods for semi-supervised text classification. In *International Conference on Learning Representations*."),
    ("pang2008opinion", "Pang and Lee", "2008", "Bo Pang and Lillian Lee. 2008. Opinion mining and sentiment analysis. "
     "*Foundations and Trends in Information Retrieval*, 2(1–2):1–135."),
    ("pontiki2016semeval", "Pontiki et al.", "2016", "Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, et al. 2016. "
     "SemEval-2016 task 5: Aspect based sentiment analysis. In *Proceedings of the 10th International Workshop on Semantic "
     "Evaluation*, pages 19–30."),
    ("pontiki2014semeval", "Pontiki et al.", "2014", "Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Harris Papageorgiou, "
     "Ion Androutsopoulos, and Suresh Manandhar. 2014. SemEval-2014 task 4: Aspect based sentiment analysis. In "
     "*Proceedings of the 8th International Workshop on Semantic Evaluation*, pages 27–35."),
    ("reimers2019sentence", "Reimers and Gurevych", "2019", "Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence "
     "embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP*."),
    ("rietzler2020adapt", "Rietzler et al.", "2020", "Alexander Rietzler, Sebastian Stabinger, Paul Opitz, and Stefan Engl. 2020. "
     "Adapt or get left behind: Domain adaptation through BERT language model finetuning for aspect-target sentiment "
     "classification. In *Proceedings of the 12th Language Resources and Evaluation Conference*."),
    ("sun2019utilizing", "Sun et al.", "2019", "Chi Sun, Luyao Huang, and Xipeng Qiu. 2019. Utilizing BERT for aspect-based "
     "sentiment analysis via constructing auxiliary sentence. In *Proceedings of NAACL-HLT*."),
    ("szegedy2016rethinking", "Szegedy et al.", "2016", "Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and "
     "Zbigniew Wojna. 2016. Rethinking the inception architecture for computer vision. In *Proceedings of CVPR*."),
    ("turney2002thumbs", "Turney", "2002", "Peter D. Turney. 2002. Thumbs up or thumbs down? Semantic orientation applied to "
     "unsupervised classification of reviews. In *Proceedings of ACL*, pages 417–424."),
    ("vaswani2017attention", "Vaswani et al.", "2017", "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, "
     "Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In *Advances in Neural "
     "Information Processing Systems*."),
    ("zhou2022improved", "Zhou and Chen", "2022", "Wenxuan Zhou and Muhao Chen. 2022. An improved baseline for sentence-level "
     "relation extraction. In *Proceedings of AACL-IJCNLP*."),
]
CITE = {k: (a, y) for k, a, y, _ in REFS}

# =========================================================================== the text
TITLE = "Aspect-Based Sentiment Analysis of Product Reviews: Comparing Traditional and Transformer Models Within and Across Datasets"
AUTHORS = [("Mrigank", "230150018"), ("Varshit", "230150023"), ("Jeshwanth", "230150011")]

ABSTRACT = (
    "A single product review often praises one part of a product and complains about another. Aspect-based sentiment "
    "analysis (ABSA) handles this by finding the aspects a reviewer talks about and working out how the reviewer feels "
    "about each one. In this project we built an ABSA pipeline for both steps, aspect term extraction and aspect-level "
    "sentiment classification, and used it to compare a TF-IDF baseline with fine-tuned BERT and RoBERTa models on "
    "SemEval-2014 Laptops and MAMS. Every model was tested on its own dataset and on the other one. Our first round of "
    "fine-tuning already beat TF-IDF by @@gain_tfidf_ll@@ macro-F1 points on Laptops and @@gain_tfidf_mm@@ on MAMS, but "
    "the training curves showed heavy over-fitting and a classifier that had no idea where the aspect was in the "
    "sentence. In a second round we addressed this with aspect markers and aspect pooling, label smoothing, adversarial "
    "training, layer-wise learning rates and constrained decoding, and we also tried RoBERTa-large. Our best models reach "
    "@@bestf1_ll@@ macro-F1 on Laptops and @@bestf1_mm@@ on MAMS. Transfer between the datasets turned out to be "
    "one-sided: models trained on MAMS do reasonably well on Laptops, but not the other way round, and aspect extraction "
    "hardly transfers at all. The biggest gain from the second round came on sentences that mix positive and negative "
    "opinions, which were the hardest cases for our first models.")

B = []  # blocks: (kind, payload)


def H(t): B.append(("h1", t))
def H2(t): B.append(("h2", t))
def P(t): B.append(("p", t))
def UL(items): B.append(("ul", items))
def OL(items): B.append(("ol", items))
def TAB(k): B.append(("table", k))
def FIGR(k): B.append(("figure", k))
def EQ(tex, plain): B.append(("eq", (tex, plain)))
def HU(t): B.append(("h1u", t))   # unnumbered heading


H("Introduction")
P("If you read reviews before buying a laptop, you have probably come across sentences like *“The laptop has an "
  "excellent display, but the battery life is disappointing.”* Is that a positive review or a negative one? It is both, "
  "and neither. A standard sentiment classifier has to choose one label for the whole sentence, and whichever label it "
  "chooses, the useful part of the message is lost. What a buyer, or the company that made the laptop, actually wants "
  "to know is that the display is good and the battery is not.")
P("Aspect-based sentiment analysis works at this finer level [@hu2004mining; @pontiki2014semeval]. It is usually split "
  "into two steps. **Aspect term extraction (ATE)** finds the words that name a part or a property of the product, such "
  "as *battery life* or *keyboard*. We treat it as sequence labelling with BIO tags. **Aspect-level sentiment "
  "classification (ASC)** then takes the sentence together with one of its aspects and decides whether the opinion about "
  "that aspect is positive, negative or neutral. The same sentence comes back once for every aspect it contains, and the "
  "model has to give a different answer depending on which aspect it is asked about.")
P("We set out to answer three questions:")
OL(["How much do pre-trained Transformer models gain over a TF-IDF model with a linear classifier, on both steps?",
    "Does a model trained on one ABSA dataset still work on a different one?",
    "Which sentences are still hard, and can we fix some of those errors by changing the way we train?"])
P("We used two public datasets. SemEval-2014 Task 4 Laptops [@pontiki2014semeval] is the standard benchmark for "
  "product reviews. MAMS [@jiang2019mams] contains restaurant reviews and was built so that every sentence has at least "
  "two aspects with different sentiments. We trained each model on one dataset and tested it on both, which gives two "
  "in-domain and two cross-dataset settings.")
P("The work happened in two rounds. In the first round (v1) we trained the baselines and a standard Transformer set-up. "
  "Looking at the training curves and the errors showed some clear problems, so we ran a second round (v2) in which "
  "each change targets one of them. In short, this report covers:")
UL(["a reproducible pipeline that puts both datasets into one format, trains and evaluates every model, and also "
    "scores the full extract-then-classify system;",
    "a fair comparison of TF-IDF with logistic regression and SVM against BERT-base and RoBERTa-base, on the same splits "
    "and metrics, in four train/test settings;",
    "a second training round based on what the v1 curves showed, run on the same backbone as v1 so that the effect of "
    "the new recipe can be separated from the effect of a bigger model;",
    "an error analysis that groups test examples by negation, contrast, several aspects, mixed polarities, implicit "
    "sentiment and distant opinion words."])

H("Related Work")
P("**Lexicons and hand-made features.** Early opinion mining relied on lists of sentiment words. [@@turney2002thumbs] "
  "estimated whether a phrase is positive or negative from how often it appears near words such as “excellent” and "
  "“poor”. [@@hu2004mining] mined frequent noun phrases from customer reviews as product features and gave each one the "
  "polarity of the opinion words around it. The surveys by [@@pang2008opinion], [@@liu2012sentiment], "
  "[@@liu2015sentiment] and [@@medhat2014sentiment] describe how these ideas grew into supervised classifiers over "
  "bag-of-words features. Such models are fast and easy to understand, but they cannot tell which opinion word belongs "
  "to which aspect, and that is exactly what ABSA needs.")
P("**Benchmarks.** The SemEval-2014 shared task [@pontiki2014semeval] released laptop and restaurant reviews annotated "
  "with aspect terms, their positions and their polarity, and SemEval-2016 [@pontiki2016semeval] added more languages "
  "and domains. [@@jiang2019mams] noticed that most SemEval sentences have only one aspect, or several aspects with the "
  "same polarity, so a model can score well just by reading the overall tone of the sentence. Their MAMS dataset removes "
  "that shortcut: every sentence has several aspects, and they disagree.")
P("**Transformers.** The Transformer [@vaswani2017attention] and pre-trained encoders such as BERT [@devlin2019bert] and "
  "RoBERTa [@liu2019roberta] changed most of NLP, and ABSA was no exception. [@@sun2019utilizing] turned aspect "
  "sentiment into a sentence-pair task by giving BERT the sentence together with a short auxiliary sentence about the "
  "aspect, and our input format follows the same idea. [@@rietzler2020adapt] showed that further pre-training on "
  "in-domain reviews helps, and they also looked at transfer between laptops and restaurants. Sentence-BERT "
  "[@reimers2019sentence] trains BERT to produce embeddings for comparing sentences. We did not use it, because in our "
  "task the aspect and the sentence need to be read together by a single encoder.")
P("**Training techniques.** Marking a target span in the input text is a simple way to tell an encoder what a "
  "prediction is about. [@@zhou2022improved] found that plain punctuation marks work well for this in relation "
  "extraction, and we borrowed that idea. Adversarial training on word embeddings [@miyato2017adversarial] and label "
  "smoothing [@szegedy2016rethinking] are general ways of reducing over-fitting. None of these techniques is new. What "
  "we wanted to find out was which of them actually help in our setting.")

H("Data")
P("**SemEval-2014 Laptops.** The training file has @@sem_tr_total@@ sentences and the test file @@sem_test_sent@@. "
  "Each aspect term comes with its character positions and one of four labels: positive, negative, neutral or "
  "conflict. We kept sentences without any aspect for extraction, because the tagger also has to learn when there is "
  "nothing to extract. The conflict label is rare (45 training cases) and does not exist in MAMS, so we removed those "
  "cases from the sentiment task. That way every experiment uses the same three classes, which the cross-dataset tests "
  "need. SemEval has no validation set, so we held out a random 10% of the training *sentences* (@@sem_val_sent@@ of "
  "them). Splitting by sentence rather than by aspect makes sure that no sentence ends up in both training and "
  "validation.")
P("**MAMS.** We used the aspect-term version of MAMS from the authors' GitHub repository. It comes with its own "
  "training, validation and test splits of @@mam_train_sent@@, @@mam_val_sent@@ and @@mam_test_sent@@ sentences, all "
  "taken from restaurant reviews.")
TAB("datasets")
P("**Preprocessing.** Both datasets are XML files with the same basic structure, so one parser handles both. We split "
  "each sentence into word tokens with a regular expression that keeps track of character positions, and then mapped "
  "each aspect's character span onto the tokens it covers to get the BIO tags. The same records also give one "
  "sentence–aspect pair per labelled aspect for the sentiment task. Parsing and tagging run in parallel on all CPU "
  "cores and the result is cached, so this step takes a few seconds the first time and no time at all after that.")
FIGR("polarity")
P("**What the data looks like.** {tab:datasets} gives the sizes, and {fig:polarity} and {fig:eda} show how the two "
  "datasets differ. They differ in almost every way that matters for this task:")
UL(["**Aspects per sentence.** Only @@sem_multi_asp@@% of Laptops sentences have more than one aspect, and only "
    "@@sem_multi_pol@@% of the sentences with aspects mix polarities. In MAMS both numbers are 100%. A MAMS sentence "
    "has @@mam_aps@@ aspects on average, compared with @@sem_aps@@ for Laptops.",
    "**Labels.** Laptops is mostly positive (@@sem_pos_share@@% of training aspects, with only @@sem_neu_share@@% "
    "neutral). MAMS leans the other way: @@mam_neu_share@@% of its aspects are neutral.",
    "**Length.** MAMS sentences are longer, @@mam_len@@ words on average against @@sem_len@@.",
    "**Vocabulary.** The two training sets contain @@voc_sem@@ and @@voc_mam@@ different aspect terms, and only "
    "@@voc_shared@@ of them appear in both (a Jaccard overlap of @@jaccard@@). Laptop reviews talk about the *battery*, "
    "*screen* and *keyboard*, restaurant reviews about the *food*, *staff* and *service*."])
P("Even before training anything, these numbers suggested that sentiment would transfer unevenly between the two "
  "datasets and that extraction would hardly transfer at all.")
FIGR("eda")

H("Methods")
H2("Task set-up")
P("For extraction, each word gets one of three tags: B starts an aspect term, I continues it, and O marks everything "
  "else. We turn the predicted tags back into spans and count a span as correct only if it matches a gold span "
  "exactly. For sentiment, the input is a sentence plus one aspect from it, and the output is positive, negative or "
  "neutral.")
H2("Traditional baselines")
P("**Sentiment.** A bag-of-words model sees the whole sentence but has no way of knowing which aspect we are asking "
  "about. To give it a fair chance, we built three groups of TF-IDF features and joined them: unigrams and bigrams of "
  "the full sentence; unigrams and bigrams of a window of six words on either side of the aspect, with each word marked "
  "by the side it comes from (so “not” before the aspect and “not” after it are different features); and the words of "
  "the aspect itself. On top of these features we trained logistic regression (LR) and a linear SVM, both with class "
  "weights to balance the labels. We picked the regularisation constant C from eight values between 0.01 and 30 using "
  "validation macro-F1.")
P("**Extraction.** The traditional tagger is a logistic regression classifier that labels one word at a time. Its "
  "features are the lower-cased word, its prefixes and suffixes, a word-shape pattern, capitalisation and digit flags, "
  "the position of the word, the two words on each side, and word bigrams. To keep the names short we list it as "
  "TF-IDF+LR in the extraction tables, even though its features are simple indicators rather than TF-IDF weights.")
H2("Transformer models (v1)")
P("We fine-tuned BERT-base-uncased [@devlin2019bert] and RoBERTa-base [@liu2019roberta]. For sentiment, the input is "
  "the sentence and the aspect as a pair, “[CLS] sentence [SEP] aspect [SEP]”, as in [@@sun2019utilizing], and a "
  "linear layer on the final [CLS] vector predicts the label. For extraction, a linear layer on every token predicts a "
  "BIO tag. When a word is split into several sub-word pieces, only the first piece is labelled and the rest are left "
  "out of the loss. Both tasks use cross-entropy loss and the AdamW optimiser with weight decay 0.01, a learning rate "
  "that warms up over the first 10% of the steps and then decays linearly, and gradient clipping at 1.0.")
H2("Improved recipe (v2)")
P("The v1 results, described in the next sections, showed four problems. The models over-fitted badly, the sentiment "
  "classifier was never told where the aspect was, the neutral class was weak, and choosing the best checkpoint on the "
  "small Laptops validation set was noisy. The second round makes the following changes.")
P("**Aspect markers and aspect pooling.** We wrap the aspect inside the sentence with plain # characters, for example "
  "“the # battery life # is disappointing”, following the idea of [@@zhou2022improved]. The aspect is still passed as "
  "the second segment as well. Instead of using only the [CLS] vector, the classifier now reads")
EQ(r"\mathbf{z} = \big[\,\mathbf{h}_{\texttt{CLS}} \,;\, \tfrac{1}{|A|}\textstyle\sum_{t \in A} \mathbf{h}_t\,\big]",
   "z = [ h_CLS ; mean of h_t over t ∈ A ]")
P("where A holds the positions of the aspect's sub-word pieces inside the sentence, which we find from the "
  "tokenizer's character offsets. The vector z goes through a small two-layer head with dropout. Put simply, the head "
  "is now told which part of the sentence it has to judge, and that matters most when a sentence contains several "
  "aspects.")
P("**Label smoothing.** Instead of a hard target of 1 for the correct class, we use 0.9 and share the remaining 0.1 "
  "between the other two classes [@szegedy2016rethinking]. This keeps the model from becoming over-confident on the "
  "training examples.")
P("**Adversarial training.** We use the fast gradient method [@miyato2017adversarial]. After the normal backward pass "
  "we push the word-embedding matrix a small step (ε = 0.5) in the direction that increases the loss, run a second "
  "forward and backward pass on the shifted embeddings, add up the gradients, and then put the embeddings back. The "
  "model learns to give the same answer when its input is slightly disturbed.")
P("**Layer-wise learning rates.** Each layer of the encoder trains with a slightly smaller learning rate than the layer "
  "above it (a factor of 0.9 per layer for the 12-layer base model and 0.95 for the 24-layer large model). The lower "
  "layers, which hold general knowledge about language, therefore change less than the top layers, while the new "
  "classification head gets a higher rate.")
P("**Constrained decoding for extraction.** Instead of picking the best tag for each word on its own, we run the "
  "Viterbi algorithm over the word-level probabilities and forbid the transitions O→I and start→I. The output is always "
  "a valid BIO sequence.")
P("**Checkpoint selection and model size.** We now check the validation set four times per epoch instead of twice, "
  "use batches of 16 instead of 32, and stop after two epochs without improvement (three for extraction). Finally, we "
  "trained RoBERTa-large with the same recipe. It only fits on our 8 GB GPU with gradient checkpointing.")
TAB("hparams")
H2("Implementation")
P("Everything is written in Python with PyTorch, Hugging Face Transformers, scikit-learn and pandas, and all neural "
  "models were trained on one laptop GPU, an NVIDIA RTX 4060 with 8 GB of memory. To keep the GPU busy we load the "
  "whole tokenised dataset into GPU memory, cut each batch to the length of its longest sentence, and train in "
  "bfloat16 mixed precision. The eight v1 fine-tuning runs took about @@t_v1_total@@ minutes of training in total and "
  "the eight v2 runs about @@t_v2_total@@ minutes, @@t_large_mams@@ of which went to RoBERTa-large on MAMS sentiment. "
  "The scripts can be resumed if they stop, and a Jupyter notebook reproduces every table and figure in this report.")

H("Experimental Setup")
P("**Settings.** Each model is trained on one dataset, tuned on that dataset's validation split, and then tested on "
  "the test sets of both datasets. We write the four settings as L→L and M→M (in-domain) and L→M and M→L "
  "(cross-dataset), where L stands for Laptops and M for MAMS. The test sets were used only once, after training had "
  "finished.")
P("**Metrics.** For sentiment we report accuracy, macro-averaged precision, recall and F1, the F1 of each class, and "
  "confusion matrices. Macro-F1 is our main number, because with uneven labels accuracy can look fine while a small "
  "class is handled badly. For extraction we report exact-match span precision, recall and F1. For the full pipeline "
  "we pass the aspects found by a tagger to the sentiment model of the same family and compute F1 over (aspect, "
  "polarity) pairs, where a pair only counts if both parts are right.")
P("**Training and model selection.** During training we measured the loss and the main metric on the training and "
  "validation sets several times per epoch, and kept the checkpoint with the best validation score (macro-F1 for "
  "sentiment, span F1 for extraction). That is the checkpoint we tested. {tab:hparams} lists the settings. Every run "
  "uses the same random seed, 42.")

H("Results")
TAB("asc")
H2("Aspect sentiment within a dataset")
P("{tab:asc} has the main results. In the first round, both Transformers beat both TF-IDF models in every setting. On "
  "Laptops, RoBERTa-base reaches @@v1r_ll@@ macro-F1 against @@lr_ll@@ for TF-IDF+LR, a difference of "
  "@@gain_tfidf_ll@@ points. On MAMS the difference is @@gain_tfidf_mm@@ points (@@v1r_mm@@ against @@lr_mm@@). "
  "RoBERTa is clearly better than BERT on Laptops (@@v1r_ll@@ against @@v1b_ll@@) and slightly better on MAMS "
  "(@@v1r_mm@@ against @@v1b_mm@@). The two TF-IDF classifiers are almost tied.")
P("The second round pushed the in-domain numbers further. With the same backbone, RoBERTa-base v2 gets @@v2b_ll@@ on "
  "Laptops and @@v2b_mm@@ on MAMS, only @@dd_v2b_ll@@ and @@dd_v2b_mm@@ points above v1. RoBERTa-large v2 does more: "
  "@@v2l_ll@@ on Laptops (@@dd_v2l_ll@@ points above v1, with @@v2l_acc_ll@@% accuracy) and @@v2l_mm@@ on MAMS "
  "(@@dd_v2l_mm@@ points above v1). These are our best in-domain results.")
TAB("perclass")
P("{tab:perclass} breaks the scores down by class. On Laptops, neutral is the weakest class for every model: "
  "RoBERTa-base v1 reaches @@neu_v1r_ll@@ neutral F1 and RoBERTa-large v2 @@neu_v2l_ll@@, while positive F1 is around "
  "90. Neutral aspects are often mentioned in passing, without any opinion words, and the line between a mildly "
  "positive remark and a neutral one is blurry even for people. On MAMS, where neutral is the most common label, it is "
  "actually the easiest class.")
H2("Aspect sentiment across datasets")
P("The cross-dataset columns of {tab:asc}, also drawn as heat maps in {fig:cross}, show that transfer only works well "
  "in one direction. A RoBERTa-base model trained on MAMS scores @@v1r_ml@@ on Laptops, not far from the @@v1r_ll@@ "
  "of a model trained on Laptops itself. The same model trained on Laptops scores only @@v1r_lm@@ on MAMS, far below "
  "the @@v1r_mm@@ of the in-domain MAMS model. Every model shows the same pattern, TF-IDF included, which tells us that "
  "the cause lies in the data rather than in the architecture.")
FIGR("cross")
P("We see two reasons. First, about nine out of ten Laptops sentences with aspects express only one sentiment, so a "
  "model trained on them can do well by reading the general tone of the sentence. On MAMS, where the aspects of a "
  "sentence always disagree, that shortcut breaks. A model trained on MAMS has had to learn to keep aspects apart, and "
  "that skill still works on the easier Laptops sentences. Second, the label mix is different. The Laptops model rarely "
  "predicts neutral, and in v1 its neutral F1 on MAMS is only @@neu_v1r_lm@@, even though neutral is the most common "
  "label there.")
P("The second round helps a lot in the hard direction. RoBERTa-base v2 reaches @@v2b_lm@@ on L→M, @@dd_v2b_lm@@ "
  "points above v1, and its neutral F1 on MAMS rises to @@neu_v2b_lm@@. In the other direction the picture is mixed: "
  "RoBERTa-base v2 loses @@dd_v2b_ml@@ points on M→L, while RoBERTa-large v2 gains @@dd_v2l_ml@@. Our best "
  "cross-dataset results are @@bestf1_lm@@ for L→M (@@best_lm@@) and @@bestf1_ml@@ for M→L (@@best_ml@@).")
TAB("ate")
H2("Aspect term extraction")
P("{tab:ate} shows the extraction results. Within a domain, the fine-tuned taggers are far ahead of the "
  "feature-based one on Laptops (RoBERTa-base v1 reaches @@ate_v1r_ll@@ span F1, against @@ate_lr_ll@@) and well ahead "
  "on MAMS (@@ate_v1r_mm@@ against @@ate_lr_mm@@). The v2 changes made little difference here. Compared with "
  "RoBERTa-base v1, the base v2 tagger is @@dd_ate_v2b_ll@@ points lower on Laptops and @@dd_ate_v2b_mm@@ points "
  "higher on MAMS, and the large tagger is within one point of v1 on both. Extraction was already strong in v1, so "
  "there was not much room left.")
P("Across domains, extraction mostly fails. Trained on Laptops and tested on MAMS, RoBERTa-base v1 reaches "
  "@@ate_lm_p@@% precision but only @@ate_lm_r@@% recall. The aspects it finds are usually right, but most restaurant "
  "aspects (dishes, staff, decor) never appear in laptop reviews, so it simply does not recognise them. The other way "
  "round, precision drops to @@ate_ml_p@@% at @@ate_ml_r@@% recall, because the MAMS tagger has learned to tag many "
  "nouns and over-tags laptop text. With an aspect vocabulary overlap of only @@jaccard@@ this is what we expected. "
  "Getting extraction to work in a new domain needs at least some text from that domain [@rietzler2020adapt].")
H2("The full pipeline")
TAB("e2e")
P("{tab:e2e} chains the two steps together. The pair F1 is always lower than either step on its own, since an aspect "
  "that the tagger misses can never get the right sentiment. Within a domain, RoBERTa-large v2 reaches @@e2e_v2l_ll@@ "
  "on Laptops and @@e2e_v2l_mm@@ on MAMS, @@dd_e2e_v2l_ll@@ and @@dd_e2e_v2l_mm@@ points above RoBERTa-base v1, while "
  "the TF-IDF pipeline gets @@e2e_lr_ll@@ and @@e2e_lr_mm@@. Across datasets every pipeline stays below @@e2e_cap@@, "
  "held back by extraction. Both v2 models improve the pair F1 in all four settings.")
H2("How the models learned")
P("{fig:curves} shows how RoBERTa learned MAMS sentiment in each round. In v1, training macro-F1 keeps climbing to "
  "@@v1_curve_train_end@@, while validation macro-F1 stops improving early and peaks at @@v1_curve_val_best@@. The "
  "validation loss is lowest (@@v1_curve_valloss_min@@) around epoch @@v1_curve_valloss_min_ep@@ and then climbs back "
  "to @@v1_curve_valloss_end@@. In other words, the model keeps getting more confident about examples it has already "
  "seen without getting any better on new ones. On the Laptops validation set, which has only @@sem_val_asp@@ aspects, "
  "the score jumped by up to @@sem_val_jump@@ points between two checks, so which checkpoint we ended up with was "
  "partly luck. These two observations are what led to the changes in v2.")
FIGR("curves")

H("From v1 to v2")
TAB("delta")
P("{tab:delta} compares each v2 model with RoBERTa-base v1, our best first-round model, and {fig:v1v2} puts their "
  "validation curves side by side. RoBERTa-base v2 uses the same encoder as v1, so its row shows what the new recipe "
  "does on its own. The RoBERTa-large row adds a bigger model on top of that.")
P("**What worked.** For sentiment, the recipe alone gives only small in-domain gains (@@dd_v2b_ll@@ points on Laptops "
  "and @@dd_v2b_mm@@ on MAMS) but a large one in the hard L→M setting (@@dd_v2b_lm@@ points). The larger encoder adds "
  "more in-domain (@@dd_v2l_ll@@ points on Laptops and @@dd_v2l_mm@@ on MAMS over v1). RoBERTa-large v2 beats the best "
  "v1 model in all four sentiment settings, and both v2 models improve the full-pipeline F1 everywhere. The "
  "aspect-pooled head was designed for sentences with several aspects, so it makes sense that its biggest effect shows "
  "up in L→M, where every test sentence mixes polarities.")
FIGR("v1v2")
P("**Over-fitting.** {tab:gap} compares training and validation macro-F1 at the checkpoint each run kept. On MAMS "
  "the regularisation did what we hoped. The gap between the two falls from @@gap_v1r_mm@@ points in v1 to "
  "@@gap_v2b_mm@@ for base v2 and @@gap_v2l_mm@@ for large v2, and the best checkpoint comes earlier (epoch "
  "@@ep_v2b_mm@@ instead of @@ep_v1r_mm@@). On Laptops it went the other way: the gap grows from @@gap_v1r_ll@@ to "
  "@@gap_v2b_ll@@ for base v2 and @@gap_v2l_ll@@ for large v2. Most of this comes down to when the checkpoint was "
  "taken. The v1 model was picked at epoch @@ep_v1r_ll@@, before it had memorised the training set, while base v2 "
  "peaked at epoch @@ep_v2b_ll@@. With only about two thousand training aspects, label smoothing and adversarial "
  "training slow memorisation down but cannot stop it, and with @@sem_val_asp@@ validation aspects the choice of "
  "epoch stays noisy.")
TAB("gap")
P("**What did not work.** Not every number went up. For extraction from Laptops to MAMS, base v2 is "
  "@@dd_ate_v2b_lm@@ points below v1, and for sentiment from MAMS to Laptops it is @@dd_v2b_ml@@ points below. Our best "
  "guess for the second drop is that a model which focuses on the aspect and the words around it picks up "
  "restaurant-specific ways of expressing opinions, while the v1 model relied more on the general tone of the sentence, "
  "which happens to work well on the mostly single-sentiment Laptops sentences. Since every run used a single seed, we "
  "would not read too much into differences of one or two points.")

H("Error Analysis")
P("To find out which sentences are still hard, we tagged every test example with six simple rules. A sentence counts "
  "as *negation* if it contains a word like not, never, n't or without, and as *contrast* if it contains but, however, "
  "although or a similar word. *Multi-aspect* and *multi-polarity* mean that the sentence has more than one aspect, or "
  "aspects with different labels. *Implicit* means that no word from the opinion lexicon of [@@hu2004mining] appears "
  "in the sentence, and *long-range* means that the closest opinion word is five or more tokens away from the aspect. "
  "{tab:phen} gives the accuracy for each group and {fig:phen} plots the same numbers.")
TAB("phen")
P("**Mixed polarities.** On Laptops, the @@n_mpol_ll@@ test aspects that sit in sentences with mixed polarities were "
  "the hardest group for the TF-IDF models (@@ph_lr_ll_mpol@@% for TF-IDF+LR) and for the v1 Transformers "
  "(@@ph_v1_ll_mpol@@% for RoBERTa-base v1, against @@v1_ll_all@@% overall). These are exactly the sentences where the "
  "general tone points the wrong way for at least one aspect. It is also where v2 helped most. RoBERTa-large v2 gets "
  "@@ph_ll_multi_polarity@@% of them right, @@mpol_gain@@ points more than v1, while its overall accuracy goes up by only "
  "@@all_gain@@ points to @@ph_ll_all@@%. For us this is the clearest sign that marking and pooling the aspect does "
  "what it was meant to do.")
P("**Negation and contrast.** For RoBERTa-large v2, negation is now the weakest group on Laptops "
  "(@@ph_ll_negation@@%), and contrast (@@ph_ll_contrast@@%) is also below the overall accuracy. Negation hurts the "
  "TF-IDF model much more (@@ph_lr_ll_neg@@%), since a bag of n-grams cannot tell what a “not” applies to. On MAMS "
  "these groups are much closer to the overall accuracy (@@ph_mm_negation@@% for negation and @@ph_mm_contrast@@% for "
  "contrast, against @@ph_mm_all@@%), probably because the model saw many sentences like them during training.")
FIGR("phen")
P("**Implicit and long-range sentiment.** Aspects in sentences with no opinion words reach @@ph_ll_implicit@@% on "
  "Laptops, and aspects far away from their opinion word @@ph_ll_long_range@@%. Implicit sentiment is often stated as "
  "a fact, like “it shuts down after an hour”, and realising that this is bad needs some knowledge of the world.")
P("**Confusions.** {fig:confusion} shows the confusion matrices of RoBERTa-large v2. On Laptops, @@cm_ll_neu_share@@% "
  "of its @@cm_ll_err@@ mistakes involve the neutral class, either as the true label or as the prediction, while mixing "
  "up positive and negative is much rarer. {tab:examples} lists some typical mistakes. The first two show what still "
  "goes wrong: in “Did not enjoy the new Windows 8…” the model misses the negation, and in “I can refute that OSX is "
  "FAST” it misses that the writer is disagreeing with a claim.")
FIGR("confusion")
TAB("examples")

H("Conclusion")
P("We built an ABSA pipeline for aspect extraction and aspect sentiment and used it to compare TF-IDF models with "
  "fine-tuned Transformers on SemEval-2014 Laptops and MAMS. Three things stood out. First, pre-trained encoders are "
  "clearly better than bag-of-words models on both steps, and our best models reach @@bestf1_ll@@ macro-F1 on Laptops "
  "and @@bestf1_mm@@ on MAMS. Second, how well a model transfers depends mostly on what it was trained on. Training on "
  "MAMS, where every sentence mixes sentiments, gives a model that also works on Laptops. Training on Laptops does not "
  "prepare a model for MAMS, and aspect extraction does not carry over between the two domains at all. Third, a second "
  "round of training based on what the first-round curves showed improved most in-domain results and the hard "
  "Laptops-to-MAMS setting, and a larger encoder added a little more on top. The biggest single gain came on Laptops "
  "sentences that mix positive and negative opinions, the group our first models handled worst. Negation is now the "
  "main weak spot.")
P("If we continued this work, we would average our results over several seeds, continue pre-training the encoders on "
  "unlabelled reviews from both domains, train one model on both datasets together, and replace the two-step pipeline "
  "with a single model that finds aspects and their sentiment at the same time.")
HU("Limitations")
P("All results come from one random seed per setting, so differences of one or two points may not be meaningful. The "
  "phenomenon tags in the error analysis come from simple rules and a word list, not from manual annotation, and will "
  "be wrong for some sentences. Dropping the SemEval conflict class makes the task a little easier than the original "
  "four-class version. We set the v2 hyper-parameters by hand from the v1 curves rather than with a proper search. "
  "Finally, both datasets are in English and cover only two domains, so our conclusions about transfer may not hold "
  "for other languages or product types.")
HU("Use of AI Assistants")
P("We used an AI assistant to help write the experiment code and to help draft and edit this report. The project idea "
  "and the choice of datasets and methods come from our proposal, and every number in this report comes from running "
  "our code.")

# =========================================================================== resolve numbers / references
def fill(t: str) -> str:
    t = re.sub(r"@@(\w+)@@", lambda m: N[m.group(1)], t)
    return t


tab_no, fig_no = {}, {}
for kind, pl in B:
    if kind == "table":
        tab_no[pl] = len(tab_no) + 1
    elif kind == "figure":
        fig_no[pl] = len(fig_no) + 1


def cite_text(keys, narrative):
    parts = []
    for k in keys:
        a, y = CITE[k]
        parts.append(f"{a} ({y})" if narrative else f"{a}, {y}")
    return "; ".join(parts) if narrative else "(" + "; ".join(parts) + ")"


# =========================================================================== DOCX renderer
def build_docx(path: Path):
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Inches, Pt

    doc = Document()
    st_ = doc.styles["Normal"]
    st_.font.name, st_.font.size = "Times New Roman", Pt(11)
    st_.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    pf = st_.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(5), 1.0

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.5))
    sec.top_margin, sec.bottom_margin = Cm(2.5), Cm(2.5)
    COL_W, PAGE_W = Inches(3.0), Cm(16.0)

    def set_cols(section, n):
        sp = section._sectPr
        cols = sp.find(qn("w:cols"))
        if cols is None:
            cols = OxmlElement("w:cols")
            sp.append(cols)
        cols.set(qn("w:num"), str(n))
        cols.set(qn("w:space"), "340")

    def new_section(ncols):
        s = doc.add_section(WD_SECTION.CONTINUOUS)
        set_cols(s, ncols)

    # page numbers in the footer
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    for tag, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        run._r.append(el)
    run.font.size = Pt(10)

    def rich(par, text, size=None, italic_all=False):
        text = fill(text)
        text = re.sub(r"\{tab:(\w+)\}", lambda m: f"Table {tab_no[m.group(1)]}", text)
        text = re.sub(r"\{fig:(\w+)\}", lambda m: f"Figure {fig_no[m.group(1)]}", text)
        text = re.sub(r"\[@@([\w; @]+)\]", lambda m: cite_text([k.strip(" @") for k in m.group(1).split(";")], True), text)
        text = re.sub(r"\[@([\w; @]+)\]", lambda m: cite_text([k.strip(" @") for k in m.group(1).split(";")], False), text)
        for tok in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                r = par.add_run(tok[2:-2])
                r.bold = True
            elif tok.startswith("*"):
                r = par.add_run(tok[1:-1])
                r.italic = True
            else:
                r = par.add_run(tok)
                r.italic = italic_all or None
            if size:
                r.font.size = Pt(size)

    def para(text, size=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=5, indent=None):
        par = doc.add_paragraph()
        par.alignment = align
        par.paragraph_format.space_after = Pt(after)
        if indent:
            par.paragraph_format.left_indent = par.paragraph_format.right_indent = indent
        rich(par, text, size)
        return par

    # ---- title block (one column)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(14)
    r = t.add_run(TITLE)
    r.bold, r.font.size = True, Pt(15)
    at = doc.add_table(rows=2, cols=3)
    at.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, (name, roll) in enumerate(AUTHORS):
        for i, (txt, bold) in enumerate([(name, True), (f"Roll No. {roll}", False)]):
            c = at.cell(i, j).paragraphs[0]
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            c.paragraph_format.space_after = Pt(0)
            rr = c.add_run(txt)
            rr.bold, rr.font.size = bold, Pt(11)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    new_section(2)

    a = doc.add_paragraph()
    a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    a.paragraph_format.space_after = Pt(4)
    ra = a.add_run("Abstract")
    ra.bold, ra.font.size = True, Pt(12)
    para(ABSTRACT, size=10, indent=Cm(0.6), after=10)

    def table_block(key):
        spec = TABLES[key]
        hdr, rows = spec["header"], spec["rows"]
        fs = 8 if not spec.get("wide") else 8.5
        best = {}
        for j in spec.get("best", []):
            vals = [r[j] for r in rows if isinstance(r, list) and isinstance(r[j], float) and r[j] == r[j]]
            best[j] = max(vals) if vals else None
        body = [r for r in rows if r != "MID"]
        n_hdr = 2 if spec.get("super") else 1
        tbl = doc.add_table(rows=len(body) + n_hdr, cols=len(hdr))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = tbl._tbl.tblPr
        w = OxmlElement("w:tblW")
        w.set(qn("w:w"), "5000")
        w.set(qn("w:type"), "pct")
        tblPr.append(w)
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "10")
            e.set(qn("w:color"), "000000")
            borders.append(e)
        tblPr.append(borders)
        mar = OxmlElement("w:tblCellMar")          # tighter cell padding
        for side in ("left", "right"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), "45")
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        tblPr.append(mar)

        def cell_border(cell, edge, sz="4"):
            tcPr = cell._tc.get_or_add_tcPr()
            b = tcPr.find(qn("w:tcBorders"))
            if b is None:
                b = OxmlElement("w:tcBorders")
                tcPr.append(b)
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), sz)
            e.set(qn("w:color"), "000000")
            b.append(e)

        def put(cell, txt, bold=False, align="center", size=fs):
            cp = cell.paragraphs[0]
            cp.alignment = {"center": WD_ALIGN_PARAGRAPH.CENTER, "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
            cp.paragraph_format.space_after = Pt(0)
            cp.paragraph_format.keep_with_next = True      # keep the whole table (and its caption) together
            for tok in re.split(r"(\*[^*]+?\*)", str(txt)):
                if not tok:
                    continue
                rr = cp.add_run(tok[1:-1] if tok.startswith("*") else tok)
                rr.italic = tok.startswith("*") or None
                rr.bold = bold or None
                rr.font.size = Pt(size)

        r0 = 0
        if spec.get("super"):
            j = 0
            for txt, span in spec["super"]:
                c = tbl.cell(0, j)
                if span > 1:
                    c = c.merge(tbl.cell(0, j + span - 1))
                    cell_border(c, "bottom", "4")
                put(c, txt, bold=True)
                j += span
            r0 = 1
        for j, h in enumerate(hdr):
            put(tbl.cell(r0, j), h, bold=True, align="left" if j == 0 else "center")
            cell_border(tbl.cell(r0, j), "bottom", "6")
        ri = n_hdr
        rule_next = False
        for r in rows:
            if r == "MID":
                rule_next = True
                continue
            cells = tbl.rows[ri].cells
            if isinstance(r, tuple) and r[0] == "GROUP":
                c = cells[0].merge(cells[-1])
                put(c, f"*{r[1]}*", align="left")
            else:
                for j, x in enumerate(r):
                    if isinstance(x, tuple) and x[0] == "D":
                        txt = f"{100 * x[1]:+.1f}"
                    elif isinstance(x, float):
                        txt = "–" if x != x else p(x)
                    else:
                        txt = x
                    isb = isinstance(x, float) and best.get(j) is not None and abs(x - best[j]) < 1e-12
                    put(cells[j], txt, bold=isb, align="left" if (j == 0 or spec.get("align") == "left") else "center")
            if rule_next:
                for c in cells:
                    cell_border(c, "top", "4")
                rule_next = False
            ri += 1
        # column widths proportional to the longest text in each column (python-docx otherwise splits evenly)
        total = int((PAGE_W if spec.get("wide") else COL_W))
        if spec.get("widths"):
            wts = list(spec["widths"])
        else:
            wts = [max([len(str(h)) for h in [hdr[j]]] +
                       [len(f"{100 * x[1]:+.1f}" if isinstance(x, tuple) else p(x) if isinstance(x, float) else str(x))
                        for r in body if isinstance(r, list) for x in [r[j]]]) for j in range(len(hdr))]
            wts = [min(max(w_, 4), 13) + 1 for w_ in wts]   # cap long labels, keep numbers on one line
        tbl.autofit = False
        for j, wt in enumerate(wts):
            wj = int(total * wt / sum(wts))
            tbl.columns[j].width = wj
            for row in tbl.rows:
                row.cells[j].width = wj
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        cap.paragraph_format.space_before = Pt(4)
        cap.paragraph_format.space_after = Pt(10)
        lab = cap.add_run(f"Table {tab_no[key]}: ")
        lab.bold, lab.font.size = True, Pt(9.5)
        rich(cap, spec["caption"], size=9.5)

    def figure_block(key):
        spec = FIGURES[key]
        wide = spec.get("wide")
        total = (PAGE_W if wide else COL_W) * spec.get("scale", 1.0)
        files = spec["files"]
        if spec.get("stack") or len(files) == 1:
            for f in files:
                fp_ = doc.add_paragraph()
                fp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                fp_.paragraph_format.space_after = Pt(2)
                fp_.paragraph_format.keep_with_next = True
                fp_.add_run().add_picture(str(FIG / f), width=int(total))
        else:
            fp_ = doc.add_paragraph()
            fp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fp_.paragraph_format.space_after = Pt(2)
            fp_.paragraph_format.keep_with_next = True
            for f in files:
                fp_.add_run().add_picture(str(FIG / f), width=int(total / len(files) * 0.98))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        cap.paragraph_format.space_after = Pt(10)
        lab = cap.add_run(f"Figure {fig_no[key]}: ")
        lab.bold, lab.font.size = True, Pt(9.5)
        rich(cap, spec["caption"], size=9.5)

    sec_no, sub_no = 0, 0
    for kind, pl in B:
        if kind == "h1":
            sec_no, sub_no = sec_no + 1, 0
            hp_ = doc.add_paragraph()
            hp_.paragraph_format.space_before, hp_.paragraph_format.space_after = Pt(10), Pt(5)
            hp_.paragraph_format.keep_with_next = True
            rr = hp_.add_run(f"{sec_no}   {pl}")
            rr.bold, rr.font.size = True, Pt(12)
        elif kind == "h1u":
            hp_ = doc.add_paragraph()
            hp_.paragraph_format.space_before, hp_.paragraph_format.space_after = Pt(10), Pt(5)
            hp_.paragraph_format.keep_with_next = True
            rr = hp_.add_run(pl)
            rr.bold, rr.font.size = True, Pt(12)
        elif kind == "h2":
            sub_no += 1
            hp_ = doc.add_paragraph()
            hp_.paragraph_format.space_before, hp_.paragraph_format.space_after = Pt(6), Pt(4)
            hp_.paragraph_format.keep_with_next = True
            rr = hp_.add_run(f"{sec_no}.{sub_no}   {pl}")
            rr.bold, rr.font.size = True, Pt(11)
        elif kind == "p":
            para(pl)
        elif kind in ("ul", "ol"):
            for i, it in enumerate(pl, 1):
                par = doc.add_paragraph()
                par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                par.paragraph_format.left_indent = Cm(0.6)
                par.paragraph_format.first_line_indent = Cm(-0.4)
                par.paragraph_format.space_after = Pt(3)
                par.add_run("•  " if kind == "ul" else f"{i}.  ")
                rich(par, it)
        elif kind == "eq":
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = par.add_run(pl[1])
            rr.italic = True
        elif kind == "table":
            if TABLES[pl].get("wide"):
                new_section(1)
                table_block(pl)
                new_section(2)
            else:
                table_block(pl)
        elif kind == "figure":
            if FIGURES[pl].get("wide"):
                new_section(1)
                figure_block(pl)
                new_section(2)
            else:
                figure_block(pl)

    hp_ = doc.add_paragraph()
    hp_.paragraph_format.space_before, hp_.paragraph_format.space_after = Pt(10), Pt(5)
    rr = hp_.add_run("References")
    rr.bold, rr.font.size = True, Pt(12)
    for _, _, _, full in sorted(REFS, key=lambda r: (r[1].lower(), r[2])):
        par = doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.LEFT
        par.paragraph_format.left_indent = Cm(0.4)
        par.paragraph_format.first_line_indent = Cm(-0.4)
        par.paragraph_format.space_after = Pt(4)
        rich(par, full, size=9.5)
    doc.save(path)


# =========================================================================== LaTeX renderer
def tex_text(t: str) -> str:
    t = fill(t)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$")]:
        t = t.replace(a, b)
    t = re.sub(r"\{tab:(\w+)\}", r"Table~\\ref{tab:\1}", t)
    t = re.sub(r"\{fig:(\w+)\}", r"Figure~\\ref{fig:\1}", t)
    t = re.sub(r"\[@@([\w; @]+)\]", lambda m: r"\citet{" + ",".join(k.strip(" @") for k in m.group(1).split(";")) + "}", t)
    t = re.sub(r"\[@([\w; @]+)\]", lambda m: r"\citep{" + ",".join(k.strip(" @") for k in m.group(1).split(";")) + "}", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"\*([^*]+?)\*", r"\\textit{\1}", t)
    for a, b in [("“", "``"), ("”", "''"), ("’", "'"), ("→", r"$\to$"), ("–", "--"), ("…", r"\ldots{}"), ("ε", r"$\epsilon$"),
                 ("∈", r"$\in$"), ("Ł", r"{\L}")]:
        t = t.replace(a, b)
    return t


def tex_table(key):
    spec = TABLES[key]
    hdr, rows = spec["header"], spec["rows"]
    best = {}
    for j in spec.get("best", []):
        vals = [r[j] for r in rows if isinstance(r, list) and isinstance(r[j], float) and r[j] == r[j]]
        best[j] = max(vals) if vals else None
    n = len(hdr)
    if spec.get("widths"):
        colspec = "".join(f"p{{{w / 6.3:.2f}\\textwidth}}" for w in spec["widths"])
    else:
        colspec = "l" + "c" * (n - 1)
    out = [r"\begin{table*}[t]" if spec.get("wide") else r"\begin{table}[t]", r"\centering",
           r"\footnotesize" if spec.get("small") or spec.get("wide") else r"\small",
           rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    if spec.get("super"):
        cells, j, rules = [], 1, []
        for txt, span in spec["super"]:
            cells.append(rf"\multicolumn{{{span}}}{{c}}{{{tex_text(txt)}}}" if span > 1 else tex_text(txt))
            if span > 1:
                rules.append(rf"\cmidrule(lr){{{j}-{j + span - 1}}}")
            j += span
        out += [" & ".join(cells) + r" \\", " ".join(rules)]
    out += [" & ".join(tex_text(h) for h in hdr) + r" \\", r"\midrule"]
    for r in rows:
        if r == "MID":
            out.append(r"\midrule")
            continue
        if isinstance(r, tuple) and r[0] == "GROUP":
            out.append(rf"\multicolumn{{{n}}}{{l}}{{\textit{{{tex_text(r[1])}}}}} \\")
            continue
        cells = []
        for j, x in enumerate(r):
            if isinstance(x, tuple) and x[0] == "D":
                c = f"{100 * x[1]:+.1f}"
            elif isinstance(x, float):
                c = "--" if x != x else p(x)
                if best.get(j) is not None and abs(x - best[j]) < 1e-12:
                    c = rf"\textbf{{{c}}}"
            else:
                c = tex_text(x)
            cells.append(c)
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{tex_text(spec['caption'])}}}", rf"\label{{tab:{key}}}",
            r"\end{table*}" if spec.get("wide") else r"\end{table}"]
    return "\n".join(out)


def tex_figure(key):
    spec = FIGURES[key]
    wide = spec.get("wide")
    files = spec["files"]
    base = (r"\textwidth" if wide else r"\columnwidth")
    if spec.get("stack") or len(files) == 1:
        w = f"{spec.get('scale', 1.0):g}{base}"
        imgs = "\n".join(rf"\includegraphics[width={w}]{{figures/{f}}}" for f in files)
    else:
        imgs = "\n".join(rf"\includegraphics[width={0.98 / len(files):.2f}{base}]{{figures/{f}}}" for f in files)
    env = "figure*" if wide else "figure"
    return (rf"\begin{{{env}}}[t]" + "\n\\centering\n" + imgs + "\n" + rf"\caption{{{tex_text(spec['caption'])}}}" + "\n"
            + rf"\label{{fig:{key}}}" + "\n" + rf"\end{{{env}}}")


def build_tex(path: Path):
    o = [r"""% ABSA project report - ACL 2023 format, single LaTeX file.
% Compile with pdfLaTeX (e.g. Overleaf: upload this file, acl2023.sty and the figures/ folder).
% Generated by report/make_report.py (the same text is also written to ABSA_Report.docx).
\pdfoutput=1
\documentclass[11pt]{article}
\usepackage[final]{acl2023}
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
""",
         rf"\title{{{tex_text(TITLE)}}}",
         r"\author{" + r" \And ".join(f"{a} \\\\ Roll No. {b}" for a, b in AUTHORS) + "}",
         r"\begin{document}", r"\maketitle", r"\begin{abstract}", tex_text(ABSTRACT), r"\end{abstract}", ""]
    for kind, pl in B:
        if kind == "h1":
            o.append(rf"\section{{{pl}}}")
        elif kind == "h1u":
            o.append(rf"\section*{{{pl}}}")
        elif kind == "h2":
            o.append(rf"\subsection{{{pl}}}")
        elif kind == "p":
            o.append(tex_text(pl) + "\n")
        elif kind in ("ul", "ol"):
            env = "itemize" if kind == "ul" else "enumerate"
            o += [rf"\begin{{{env}}}", r"\setlength\itemsep{0em}"] + [rf"\item {tex_text(it)}" for it in pl] + [rf"\end{{{env}}}", ""]
        elif kind == "eq":
            o += [r"\begin{equation}", pl[0], r"\end{equation}"]
        elif kind == "table":
            o += [tex_table(pl), ""]
        elif kind == "figure":
            o += [tex_figure(pl), ""]
    o += [r"\begin{thebibliography}{99}", r"\expandafter\ifx\csname natexlab\endcsname\relax\def\natexlab#1{#1}\fi"]
    for k, a, y, full in sorted(REFS, key=lambda r: (r[1].lower(), r[2])):
        o += [rf"\bibitem[{{{tex_text(a).replace(' et al.', ' et~al.')}({y})}}]{{{k}}}", tex_text(full), ""]
    o += [r"\end{thebibliography}", r"\end{document}"]
    path.write_text("\n".join(o), encoding="utf-8")


if __name__ == "__main__":
    for f in ("acl2023.sty", "acl_natbib.bst"):
        shutil.copy(OUT / "acl2023_template" / f, OUT / f)
    build_docx(OUT / "ABSA_Report.docx")
    build_tex(OUT / "ABSA_Report.tex")
    words = sum(len(fill(x).split()) for k, x in B if k == "p") + sum(len(fill(i).split()) for k, x in B if k in ("ul", "ol") for i in x)
    print("written report/ABSA_Report.docx and report/ABSA_Report.tex  |  body words:", words + len(fill(ABSTRACT).split()))
