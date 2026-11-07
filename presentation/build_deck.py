"""Builds the final project presentation (ABSA_Final_Presentation.pptx) from the experiment outputs.

    python presentation/build_deck.py

All numbers are read from results/ (v1) and results_v2/ (v2); figures are the pipeline's own PNGs.
Visual style follows the proposal deck (navy / teal palette, Cambria headings, Calibri body).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
R1, R2 = ROOT / "results", ROOT / "results_v2"
OUT = ROOT / "presentation" / "ABSA_Final_Presentation.pptx"

NAVY, NAVY2, TEAL, SLATE, MUTED = "13232F", "0B1922", "1C7293", "5C7182", "8FA9B8"
LIGHT, LIGHT2, WHITE, GREEN, RED, TEAL_LT = "EFF4F7", "DCE5EB", "FFFFFF", "0E8A5F", "C4442A", "7FB6C9"
HEAD, BODY = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)

CFG = [("semeval", "semeval"), ("mams", "mams"), ("semeval", "mams"), ("mams", "semeval")]
CFG_LAB = ["Laptop → Laptop", "MAMS → MAMS", "Laptop → MAMS", "MAMS → Laptop"]

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
_n = [0]


# --------------------------------------------------------------------------- primitives
def rgb(h):
    return RGBColor.from_string(h)


def rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    r = s.shapes.add_shape(shape, x, y, w, h)
    r.fill.solid()
    r.fill.fore_color.rgb = rgb(fill)
    if line:
        r.line.color.rgb = rgb(line)
        r.line.width = Pt(1)
    else:
        r.line.fill.background()
    r.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        r.adjustments[0] = 0.08
    return r


def text(s, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, margin=0.05):
    """paras: list of str or dict(text|runs, size, bold, color, font, bullet, space, align)."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    for i, p in enumerate(paras):
        p = {"text": p} if isinstance(p, str) else p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", align)
        para.space_after = Pt(p.get("space", 4))
        runs = p.get("runs") or [(p.get("text", ""), {})]
        if p.get("bullet"):
            runs = [("•  ", {"color": p.get("bullet_color", TEAL), "bold": True})] + list(runs)
        for t, kw in runs:
            r = para.add_run()
            r.text = t
            f = r.font
            f.size = Pt(kw.get("size", p.get("size", 16)))
            f.bold = kw.get("bold", p.get("bold", False))
            f.italic = kw.get("italic", p.get("italic", False))
            f.name = kw.get("font", p.get("font", BODY))
            f.color.rgb = rgb(kw.get("color", p.get("color", NAVY)))
    return tb


def picture(s, path, x, y, w, h):
    """Fit image inside the box, centred."""
    path = Path(path)
    if not path.exists():
        text(s, x, y, w, h, [{"text": f"[missing figure: {path.name}]", "color": RED}])
        return
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(str(path), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


def slide(kicker, title, sub=None):
    s = prs.slides.add_slide(BLANK)
    _n[0] += 1
    rect(s, 0, 0, W, H, WHITE)
    rect(s, 0, 0, Inches(0.12), H, TEAL)
    text(s, Inches(0.55), Inches(0.32), Inches(12), Inches(0.35),
         [{"text": kicker.upper(), "size": 12, "bold": True, "color": TEAL}])
    text(s, Inches(0.55), Inches(0.62), Inches(12.3), Inches(0.7),
         [{"text": title, "size": 28, "font": HEAD, "bold": True, "color": NAVY}])
    if sub:
        text(s, Inches(0.55), Inches(1.28), Inches(12.3), Inches(0.45), [{"text": sub, "size": 15, "color": SLATE}])
    rect(s, Inches(0.55), Inches(7.08), Inches(12.25), Emu(9525), LIGHT2)
    text(s, Inches(0.55), Inches(7.1), Inches(9), Inches(0.3),
         [{"text": "Aspect-Based Sentiment Analysis · Final Presentation", "size": 10, "color": MUTED}])
    text(s, Inches(11.8), Inches(7.1), Inches(1), Inches(0.3), [{"text": str(_n[0]), "size": 10, "color": MUTED}],
         align=PP_ALIGN.RIGHT)
    return s


def card(s, x, y, w, h, label, title, lines, accent=TEAL, size=13.5):
    rect(s, x, y, w, h, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    rect(s, x, y + Inches(0.15), Inches(0.07), h - Inches(0.3), accent)
    paras = [{"text": label.upper(), "size": 10.5, "bold": True, "color": accent, "space": 2},
             {"text": title, "size": 17, "bold": True, "font": HEAD, "color": NAVY, "space": 8}]
    paras += [{"text": l, "size": size, "color": NAVY2, "bullet": True, "bullet_color": accent, "space": 5} for l in lines]
    text(s, x + Inches(0.22), y + Inches(0.12), w - Inches(0.35), h - Inches(0.2), paras)


def stat(s, x, y, w, value, label, color=TEAL, size=34):
    text(s, x, y, w, Inches(0.7), [{"text": value, "size": size, "bold": True, "font": HEAD, "color": color}])
    text(s, x, y + Inches(0.68), w, Inches(0.7), [{"text": label, "size": 12.5, "color": SLATE}])


def table(s, x, y, w, header, rows, col_w=None, font=13, row_h=0.36, bold_best=None, delta_cols=(), first_col_bold=True,
          highlight_rows=()):
    """rows: list of lists; floats are shown x100 with 1 decimal. delta_cols are shown with sign and coloured."""
    nr, nc = len(rows) + 1, len(header)
    gt = s.shapes.add_table(nr, nc, x, y, w, Inches(row_h * nr)).table
    col_w = col_w or [1 / nc] * nc
    for j, f in enumerate(col_w):
        gt.columns[j].width = int(w * f / sum(col_w))
    best = {}
    for j in (bold_best or []):
        vals = [r[j] for r in rows if isinstance(r[j], float) and r[j] == r[j]]
        best[j] = max(vals) if vals else None

    def fill(cell, color, t, bold=False, fc=NAVY, align=PP_ALIGN.CENTER, size=font):
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb(color)
        cell.margin_left = cell.margin_right = Inches(0.06)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.text = ""
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = t
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, BODY
        r.font.color.rgb = rgb(fc)

    for j, hname in enumerate(header):
        fill(gt.cell(0, j), NAVY, hname, True, WHITE, PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER, font - 0.5)
    for i, r in enumerate(rows, start=1):
        bg = "E3EEF3" if (i - 1) in highlight_rows else (WHITE if i % 2 else LIGHT)
        for j, v in enumerate(r):
            fc, bold = NAVY2, (j == 0 and first_col_bold)
            if isinstance(v, float):
                if v != v:
                    t = "–"
                elif j in delta_cols:
                    t = f"{100 * v:+.1f}"
                    fc = GREEN if v > 0.0005 else RED if v < -0.0005 else SLATE
                    bold = True
                else:
                    t = f"{100 * v:.1f}"
                    if best.get(j) is not None and abs(v - best[j]) < 1e-12:
                        bold, fc = True, TEAL
            else:
                t = str(v)
            fill(gt.cell(i, j), bg, t, bold, fc, PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
    return gt


def arrow(s, x, y, w):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, Inches(0.32))
    a.fill.solid()
    a.fill.fore_color.rgb = rgb(TEAL_LT)
    a.line.fill.background()


# --------------------------------------------------------------------------- data
asc = pd.read_csv(R2 / "tables" / "asc_results.csv")
ate = pd.read_csv(R2 / "tables" / "ate_results.csv")
e2e = pd.read_csv(R2 / "tables" / "end2end_results.csv")
ea = pd.read_csv(R2 / "error_analysis" / "accuracy_by_phenomenon.csv")
ds = pd.read_csv(R1 / "eda" / "dataset_summary.csv")
st = pd.read_csv(R1 / "eda" / "dataset_statistics.csv").set_index("dataset")
gap = pd.read_csv(R2 / "improvements" / "generalisation_gap.csv")
vocab = (R1 / "eda" / "vocab_overlap.txt").read_text().strip()
demo_path = R2 / "demo_predictions.csv"
demo = pd.read_csv(demo_path) if demo_path.exists() else None
times = json.loads((R2 / "summary.md").read_text(encoding="utf-8").split("## Transformer training time (s)")[-1].strip() or "{}") \
    if "## Transformer training time (s)" in (R2 / "summary.md").read_text(encoding="utf-8") else {}

V1B, V1R, V2B, V2L = "BERT-base v1", "RoBERTa-base v1", "RoBERTa-base v2", "RoBERTa-large v2"


def v(df, m, a, b, col):
    s = df[(df.model == m) & (df.train == a) & (df.test == b)]
    return float(s[col].iloc[0]) if len(s) else float("nan")


def pts(x):
    return f"{100 * x:+.1f} pts"


# =========================================================================== slides
# 1 ---- title
s = prs.slides.add_slide(BLANK)
_n[0] += 1
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, Inches(0.12), H, TEAL)
rect(s, Inches(8.9), 0, Inches(4.433), H, NAVY2)
text(s, Inches(0.8), Inches(0.8), Inches(8), Inches(0.4),
     [{"text": "DEPARTMENT ELECTIVE PROJECT  ·  FINAL PRESENTATION", "size": 13, "bold": True, "color": TEAL_LT}])
text(s, Inches(0.8), Inches(1.5), Inches(7.9), Inches(2.2),
     [{"text": "Aspect-Based Sentiment Analysis of Product Reviews", "size": 40, "bold": True, "font": HEAD, "color": WHITE}])
text(s, Inches(0.8), Inches(3.75), Inches(7.6), Inches(1.0),
     [{"text": "What we built, what the numbers say, and how a second training round (v2) improved on the first (v1).",
       "size": 17, "color": LIGHT2}])
text(s, Inches(0.8), Inches(4.85), Inches(7.9), Inches(0.4),
     [{"text": "TF-IDF  ·  BERT  ·  RoBERTa-base  ·  RoBERTa-large  ·  SemEval-2014 Laptops  ·  MAMS", "size": 13, "color": MUTED}])
text(s, Inches(0.8), Inches(5.6), Inches(7.5), Inches(1.3),
     [{"text": "Mrigank · 230150018", "size": 15, "color": WHITE},
      {"text": "Varshit · 230150023", "size": 15, "color": WHITE},
      {"text": "Jeshwanth · 230150011", "size": 15, "color": WHITE}])
best_l = asc[(asc.train == "semeval") & (asc.test == "semeval")].sort_values("macro_f1").iloc[-1]
best_m = asc[(asc.train == "mams") & (asc.test == "mams")].sort_values("macro_f1").iloc[-1]
text(s, Inches(9.3), Inches(1.0), Inches(3.7), Inches(0.4), [{"text": "HEADLINE RESULTS", "size": 12, "bold": True, "color": TEAL_LT}])
for k, (val, lab) in enumerate([(f"{100 * best_l.macro_f1:.1f}", f"macro-F1 on SemEval Laptops\n({best_l.model})"),
                                (f"{100 * best_m.macro_f1:.1f}", f"macro-F1 on MAMS\n({best_m.model})"),
                                (f"{100 * v(ate, V2B, 'semeval', 'semeval', 'f1'):.1f}" if V2B in set(ate.model) else "–",
                                 "aspect-extraction span F1\non Laptops (RoBERTa-base v2)")]):
    text(s, Inches(9.3), Inches(1.55 + 1.75 * k), Inches(3.7), Inches(0.8),
         [{"text": val, "size": 44, "bold": True, "font": HEAD, "color": WHITE}])
    text(s, Inches(9.3), Inches(2.35 + 1.75 * k), Inches(3.7), Inches(0.8), [{"text": lab, "size": 13, "color": LIGHT2}])
notes(s, "Introduce the team and the goal: compare a traditional TF-IDF approach with fine-tuned Transformers for "
         "aspect-based sentiment analysis, in-domain and across datasets, then show how a second training round improved results.")

# 2 ---- recap
s = slide("Recap", "One sentence, several opinions",
          "ABSA splits the problem into two sub-tasks, and we evaluate every model in four train/test configurations.")
rect(s, Inches(0.55), Inches(1.95), Inches(12.25), Inches(0.85), LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, Inches(0.8), Inches(2.02), Inches(11.8), Inches(0.75),
     [{"runs": [("“The laptop has an excellent ", {}), ("display", {"bold": True, "color": GREEN}),
                (", but the ", {}), ("battery life", {"bold": True, "color": RED}), (" is disappointing.”", {})],
       "size": 21, "font": HEAD, "italic": True}], anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
card(s, Inches(0.55), Inches(3.05), Inches(4.0), Inches(3.8), "Sub-task 1", "Aspect Term Extraction",
     ["Token-level BIO tagging", "the/O battery/B life/I is/O ...", "Scored by exact-match span precision, recall, F1"])
card(s, Inches(4.7), Inches(3.05), Inches(4.0), Inches(3.8), "Sub-task 2", "Aspect Sentiment Classification",
     ["Input: sentence + one target aspect", "Output: positive / negative / neutral",
      "Scored by accuracy and macro-F1 (class imbalance)"])
card(s, Inches(8.85), Inches(3.05), Inches(3.95), Inches(3.8), "Protocol", "Four configurations",
     ["Train Laptop → test Laptop", "Train MAMS → test MAMS", "Train Laptop → test MAMS", "Train MAMS → test Laptop",
      "Plus an end-to-end run: extract, then classify"], accent=SLATE)
notes(s, "Quick reminder of the problem framing from the proposal. Sub-task 1 finds the aspects, sub-task 2 assigns a polarity "
         "to each. Two configurations are in-domain, two test generalisation to the other corpus.")

# 3 ---- data
s = slide("Data", "Two corpora with very different shapes",
          "SemEval-2014 Laptops (product reviews) and MAMS (restaurant reviews, built so every sentence mixes polarities).")
rows = []
for _, r in ds.iterrows():
    rows.append([{"semeval": "SemEval Laptop", "mams": "MAMS"}[r.dataset], r.split, f"{int(r.sentences):,}",
                 f"{int(r.asc_instances):,}", f"{int(r.positive):,}", f"{int(r.negative):,}", f"{int(r.neutral):,}"])
table(s, Inches(0.55), Inches(1.95), Inches(7.0), ["Dataset", "Split", "Sentences", "Aspects", "Pos", "Neg", "Neu"], rows,
      col_w=[2.1, 1, 1.3, 1.2, 0.9, 0.9, 0.9], font=12.5, row_h=0.4)
stat(s, Inches(0.6), Inches(5.0), Inches(2.4), f"{100 * st.loc['semeval', 'share_sentences_multi_polarity']:.0f}% / 100%",
     "sentences with mixed polarities\n(Laptop / MAMS)", size=28)
stat(s, Inches(3.05), Inches(5.0), Inches(2.2), f"{st.loc['mams', 'mean_aspects_per_sentence(with>=1)']:.1f} vs "
                                                f"{st.loc['semeval', 'mean_aspects_per_sentence(with>=1)']:.1f}",
     "aspects per sentence\n(MAMS vs Laptop)")
jac = vocab.split("Jaccard=")[-1].rstrip(")")
stat(s, Inches(5.35), Inches(5.0), Inches(2.3), jac, "Jaccard overlap of aspect\nvocabularies (train)", color=RED)
picture(s, R1 / "eda" / "polarity_distribution.png", Inches(7.75), Inches(1.9), Inches(5.1), Inches(4.9))
notes(s, "SemEval has no validation split, so we hold out 10 percent of training sentences. The 45 'conflict' labels are "
         "dropped so both datasets share three classes. MAMS is neutral-heavy, Laptops is positive-heavy, and the aspect "
         "vocabularies barely overlap, which matters for the cross-dataset results.")

# 4 ---- pipeline
s = slide("What we built", "A reproducible pipeline, end to end",
          "One command runs everything: python run_all.py (v1) and python run_v2.py (v2). Runs are cached and resumable.")
steps = [("Raw data", "SemEval XML\nMAMS auto-download"), ("Preprocess", "16-process pool:\nparse, tokenise, BIO"),
         ("Two views", "token BIO tags\nsentence–aspect pairs"), ("Models", "TF-IDF + LR / SVM\nBERT, RoBERTa"),
         ("Evaluate", "4 configurations\nend-to-end pairs"), ("Analyse", "curves, confusion,\nerror phenomena")]
bw, gapw, x0 = Inches(1.78), Inches(0.31), Inches(0.55)
for k, (t1, t2) in enumerate(steps):
    x = x0 + k * (bw + gapw)
    rect(s, x, Inches(2.05), bw, Inches(1.55), NAVY if k in (3,) else LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x, Inches(2.12), bw, Inches(1.45),
         [{"text": t1, "size": 16, "bold": True, "font": HEAD, "color": WHITE if k == 3 else NAVY, "align": PP_ALIGN.CENTER},
          {"text": t2, "size": 12, "color": LIGHT2 if k == 3 else SLATE, "align": PP_ALIGN.CENTER}], anchor=MSO_ANCHOR.MIDDLE)
    if k < len(steps) - 1:
        arrow(s, x + bw + Inches(0.04), Inches(2.67), gapw - Inches(0.08))
card(s, Inches(0.55), Inches(4.0), Inches(6.0), Inches(2.85), "Engineering", "Fast on a laptop GPU (RTX 4060, 8 GB)",
     ["Whole tokenised dataset kept on the GPU, batches trimmed to their longest sentence",
      "bf16 mixed precision, TF32 matmuls, fused AdamW",
      "Preprocessing in ~8 s and cached; full v1 study in ~17 min"])
card(s, Inches(6.8), Inches(4.0), Inches(6.0), Inches(2.85), "Tracking", "Every run leaves a trail",
     ["Train and validation loss + metrics several times per epoch",
      "Early stopping on validation F1, best checkpoint restored",
      "Per-run curves, LR schedule, confusion matrices, live history.csv"], accent=SLATE)
notes(s, "Walk through the pipeline left to right. Emphasise that every number in this talk is produced by the scripts and "
         "that the Jupyter notebook reproduces them with execution instructions.")

# 5 ---- v1 setup
s = slide("Round 1 (v1)", "Models compared under identical conditions")
card(s, Inches(0.55), Inches(1.55), Inches(4.0), Inches(5.3), "Baseline", "TF-IDF + LR / SVM",
     ["Sentence 1–2-gram TF-IDF", "Context window: 6 words left/right of the aspect, side-tagged", "Aspect term TF-IDF",
      "C tuned on validation, grid run in parallel", "ATE: logistic-regression BIO tagger on word-window features"], accent=SLATE)
card(s, Inches(4.7), Inches(1.55), Inches(4.0), Inches(5.3), "Transformers", "BERT-base & RoBERTa-base",
     ["ASC input: [CLS] sentence [SEP] aspect [SEP]", "Classifier on the [CLS] vector", "ATE: token classifier, label on first sub-word",
      "AdamW, linear warm-up/decay, LR 2e-5 (ASC) / 5e-5 (ATE)", "6 (ASC) / 8 (ATE) epochs, batch 32"])
card(s, Inches(8.85), Inches(1.55), Inches(3.95), Inches(5.3), "Evaluation", "Same splits, same metrics",
     ["ASC: accuracy, macro-P/R/F1, per-class F1", "ATE: exact-match span P/R/F1", "End-to-end: aspect span and polarity both right",
      "Checkpoint chosen on validation only; test touched once"], accent=GREEN)
notes(s, "The baseline has no positional link between opinion words and the target, so we gave it a context window feature "
         "to make it a fair comparison. The Transformers read the sentence and the aspect together.")

# 6 ---- v1 ASC results
s = slide("Round 1 results", "Transformers beat TF-IDF everywhere, but transfer is lopsided",
          "Aspect sentiment classification, macro-F1 (%) on the test sets.")
m1 = ["TF-IDF+LR", "TF-IDF+SVM", V1B, V1R]
rows = [[m] + [v(asc, m, a, b, "macro_f1") for a, b in CFG] for m in m1]
table(s, Inches(0.55), Inches(1.95), Inches(7.4), ["Model"] + CFG_LAB, rows, col_w=[2.2, 1.3, 1.3, 1.3, 1.3],
      bold_best=[1, 2, 3, 4], row_h=0.5, font=14)
gain = v(asc, V1R, "semeval", "semeval", "macro_f1") - v(asc, "TF-IDF+LR", "semeval", "semeval", "macro_f1")
gain_m = v(asc, V1R, "mams", "mams", "macro_f1") - v(asc, "TF-IDF+LR", "mams", "mams", "macro_f1")
fwd, bwd = v(asc, V1R, "mams", "semeval", "macro_f1"), v(asc, V1R, "semeval", "mams", "macro_f1")
card(s, Inches(8.2), Inches(1.95), Inches(4.6), Inches(4.9), "Reading the table", "Three observations",
     [f"Contextual models add {100 * gain:.0f} macro-F1 points over TF-IDF on Laptops",
      f"MAMS → Laptop keeps {100 * fwd:.0f} F1, Laptop → MAMS only {100 * bwd:.0f}",
      "A model trained on mostly single-polarity sentences never learns to separate aspects that disagree",
      "On Laptops, neutral is the weakest class for every model"])
picture(s, R1 / "figures" / "asc_cross_dataset_macro_f1.png", Inches(0.55), Inches(4.65), Inches(7.4), Inches(2.3))
notes(s, "Highlight the asymmetry: training on MAMS transfers to laptops reasonably well, the reverse does not.")

# 7 ---- v1 ATE + e2e
s = slide("Round 1 results", "Aspect extraction is domain-bound",
          "Span F1 (%) for aspect term extraction and for the full extract-then-classify pipeline.")
rows = [[m] + [v(ate, m, a, b, "f1") for a, b in CFG] for m in ["TF-IDF+LR", V1B, V1R]]
text(s, Inches(0.55), Inches(1.85), Inches(6), Inches(0.4), [{"text": "Aspect term extraction (span F1)", "size": 15, "bold": True}])
table(s, Inches(0.55), Inches(2.3), Inches(7.4), ["Model"] + CFG_LAB, rows, col_w=[2.2, 1.3, 1.3, 1.3, 1.3], bold_best=[1, 2, 3, 4],
      row_h=0.48, font=14)
rows = [[m] + [v(e2e, m, a, b, "f1") for a, b in CFG] for m in ["TF-IDF+LR", V1B, V1R]]
text(s, Inches(0.55), Inches(4.45), Inches(6), Inches(0.4), [{"text": "End-to-end (aspect and polarity both correct)", "size": 15, "bold": True}])
table(s, Inches(0.55), Inches(4.9), Inches(7.4), ["Model"] + CFG_LAB, rows, col_w=[2.2, 1.3, 1.3, 1.3, 1.3], bold_best=[1, 2, 3, 4],
      row_h=0.48, font=14)
lm = ate[(ate.model == V1R) & (ate.train == "semeval") & (ate.test == "mams")].iloc[0]
card(s, Inches(8.2), Inches(1.85), Inches(4.6), Inches(5.0), "Why", "What limits extraction",
     ["In-domain RoBERTa reaches the mid-80s on Laptops",
      f"Laptop → MAMS: precision {100 * lm.precision:.0f}% but recall {100 * lm.recall:.0f}% — restaurant aspects are simply unseen",
      "End-to-end scores sit well below ASC alone: every missed aspect is also a missed sentiment"])
notes(s, "Extraction depends on domain vocabulary; the model is precise but misses most restaurant aspects when trained on laptops.")

# 8 ---- diagnosis
s = slide("Diagnosis", "Why were the v1 numbers limited?",
          "Reading the training curves and per-class scores pointed to four causes rather than one bug.")
picture(s, R1 / "transformers" / "asc" / "roberta_train-mams" / "training_curves.png", Inches(0.45), Inches(1.85),
        Inches(7.9), Inches(2.4))
h1 = json.load(open(R1 / "transformers" / "asc" / "roberta_train-mams" / "history.json"))["history"]
last = h1[-1]
neu = v(asc, V1R, "semeval", "semeval", "neutral_f1")
items = [("1", "Over-fitting", f"train macro-F1 reaches {100 * last['train_macro_f1']:.0f} while validation stalls near "
                               f"{100 * max(r['val_macro_f1'] for r in h1):.0f}; validation loss rises after ~1.5 epochs"),
         ("2", "Aspect localisation", "the [CLS] head is never told where the aspect sits — a problem when MAMS sentences carry 2–3 aspects with different labels"),
         ("3", "Neutral class", f"neutral F1 only {100 * neu:.0f} on Laptops; most errors are neutral ↔ positive/negative"),
         ("4", "Noisy selection", "SemEval validation has 221 aspects, so validation F1 jumps ±3 points between checks")]
for k, (num, head, body) in enumerate(items):
    x = Inches(0.55) + (k % 2) * Inches(6.2)
    y = Inches(4.45) + (k // 2) * Inches(1.3)
    rect(s, x, y, Inches(0.55), Inches(0.55), TEAL, shape=MSO_SHAPE.OVAL)
    text(s, x, y, Inches(0.55), Inches(0.55), [{"text": num, "size": 18, "bold": True, "color": WHITE, "align": PP_ALIGN.CENTER}],
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + Inches(0.7), y - Inches(0.05), Inches(5.3), Inches(1.25),
         [{"text": head, "size": 16, "bold": True, "font": HEAD, "space": 2}, {"text": body, "size": 13, "color": SLATE}])
text(s, Inches(8.55), Inches(1.9), Inches(4.3), Inches(2.3),
     [{"text": "v1 RoBERTa-base, trained on MAMS", "size": 13, "bold": True, "color": TEAL},
      {"text": "Blue = train, orange = validation. The gap opens after the first epoch and keeps widening.", "size": 13, "color": SLATE}])
notes(s, "This slide motivates every change in v2. Point at the widening gap between train and validation curves.")

# 9 ---- v2 changes
s = slide("Round 2 (v2)", "Each change answers one diagnosis")
rows = [["Over-fitting", "Label smoothing 0.1 · FGM adversarial training on embeddings (ε = 0.5) · layer-wise LR decay", "ASC + ATE"],
        ["Aspect localisation", "Aspect wrapped in the sentence: “… the # battery life # is …”; head reads [CLS] ⊕ mean of aspect tokens", "ASC"],
        ["Noisy selection", "4 validation checks per epoch, batch 16, patience counted in epochs", "ASC + ATE"],
        ["Invalid BIO output", "Viterbi decoding that forbids O → I transitions", "ATE"],
        ["Model capacity", "RoBERTa-large (355M params) with gradient checkpointing to fit 8 GB", "ASC + ATE"]]
table(s, Inches(0.55), Inches(1.6), Inches(12.25), ["v1 problem", "v2 change", "Task"], rows, col_w=[2.3, 8.2, 1.4],
      row_h=0.62, font=14)
text(s, Inches(0.55), Inches(5.6), Inches(12.2), Inches(1.3),
     [{"runs": [("Controlled comparison: ", {"bold": True, "color": TEAL}),
                ("RoBERTa-base v2 uses the same backbone as v1, so its gain is due to the recipe alone. "
                 "RoBERTa-large v2 adds model capacity on top.", {})], "size": 15},
      {"runs": [("Unchanged: ", {"bold": True, "color": TEAL}), ("data splits, metrics, test sets and the four configurations.", {})],
       "size": 15}])
notes(s, "Go row by row: problem, then the change. Mention that markers follow Zhou and Chen's typed-marker idea and FGM is from Miyato et al.")

# 10 ---- v2 ASC results
s = slide("v1 → v2", "Aspect sentiment: where the second round helped",
          "Macro-F1 (%) and change versus the best v1 model (RoBERTa-base v1).")
rows = []
for a, b in CFG:
    base = v(asc, V1R, a, b, "macro_f1")
    rows.append([CFG_LAB[CFG.index((a, b))], base, v(asc, V2B, a, b, "macro_f1"), v(asc, V2B, a, b, "macro_f1") - base,
                 v(asc, V2L, a, b, "macro_f1"), v(asc, V2L, a, b, "macro_f1") - base])
table(s, Inches(0.55), Inches(1.95), Inches(6.6), ["Configuration", "v1 base", "v2 base", "Δ", "v2 large", "Δ"], rows,
      col_w=[2.2, 1.1, 1.1, 0.9, 1.1, 0.9], delta_cols=(3, 5), row_h=0.52, font=14)
acc_rows = []
for a, b in CFG:
    base = v(asc, V1R, a, b, "accuracy")
    acc_rows.append([CFG_LAB[CFG.index((a, b))], base, v(asc, V2L, a, b, "accuracy"), v(asc, V2L, a, b, "accuracy") - base])
text(s, Inches(0.55), Inches(4.75), Inches(6), Inches(0.4), [{"text": "Accuracy (%)", "size": 14, "bold": True}])
table(s, Inches(0.55), Inches(5.15), Inches(6.6), ["Configuration", "v1 base", "v2 large", "Δ"], acc_rows,
      col_w=[2.2, 1.4, 1.4, 1.2], delta_cols=(3,), row_h=0.36, font=12.5)
picture(s, R2 / "improvements" / "v1_vs_v2_asc_macro_f1.png", Inches(7.3), Inches(1.85), Inches(5.55), Inches(5.0))
notes(s, "Read the deltas: green is an improvement over the best v1 model. Comment separately on in-domain and cross-dataset rows.")

# 11 ---- v2 ATE + e2e
s = slide("v1 → v2", "Extraction and the full pipeline",
          "Span F1 (%) — change versus RoBERTa-base v1.")
for blk, (df, lab, y) in enumerate([(ate, "Aspect term extraction", Inches(1.9)), (e2e, "End-to-end pair F1", Inches(4.45))]):
    rows = []
    for a, b in CFG:
        base = v(df, V1R, a, b, "f1")
        rows.append([CFG_LAB[CFG.index((a, b))], base, v(df, V2B, a, b, "f1"), v(df, V2B, a, b, "f1") - base,
                     v(df, V2L, a, b, "f1"), v(df, V2L, a, b, "f1") - base])
    text(s, Inches(0.55), y, Inches(6), Inches(0.4), [{"text": lab, "size": 15, "bold": True}])
    table(s, Inches(0.55), y + Inches(0.42), Inches(6.6), ["Configuration", "v1 base", "v2 base", "Δ", "v2 large", "Δ"], rows,
          col_w=[2.2, 1.1, 1.1, 0.9, 1.1, 0.9], delta_cols=(3, 5), row_h=0.4, font=13)
picture(s, R2 / "improvements" / "v1_vs_v2_end2end_f1.png", Inches(7.3), Inches(1.85), Inches(5.55), Inches(5.0))
notes(s, "Constrained decoding and regularisation change extraction; the end-to-end score inherits gains from both stages.")

# 12 ---- learning dynamics
s = slide("v1 → v2", "Learning dynamics: less over-fitting on MAMS, noisy on Laptops",
          "Validation macro-F1 and validation loss per epoch (top: metric, bottom: plain cross-entropy).")
picture(s, R2 / "improvements" / "asc_val_curves_v1_vs_v2.png", Inches(0.45), Inches(1.8), Inches(7.6), Inches(5.15))
g = gap[gap.task == "asc"].copy()
order = {V1B: 0, V1R: 1, V2B: 2, V2L: 3}
g = g[g.model.isin(order)].sort_values(["train", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
rows = [[f"{r.model}", {"semeval": "Laptop", "mams": "MAMS"}[r.train], r.train_metric, r.val_metric, r.gap] for r in g.itertuples()]
text(s, Inches(8.3), Inches(1.8), Inches(4.5), Inches(0.4), [{"text": "Gap at the selected checkpoint (ASC)", "size": 14, "bold": True}])
table(s, Inches(8.3), Inches(2.2), Inches(4.55), ["Model", "Train", "Train F1", "Val F1", "Gap"], rows,
      col_w=[2.1, 0.9, 0.9, 0.9, 0.8], row_h=0.4, font=11.5)
notes(s, "On MAMS the train-validation gap roughly halves and the best epoch comes earlier. On Laptops the v1 checkpoint happened to be "
         "picked early, so its gap looks small; with only 221 validation aspects the selected epoch is noisy.")

# 13 ---- cross dataset
s = slide("Generalisation", "Cross-dataset transfer, all models",
          "Rows: training corpus. Columns: test corpus. Macro-F1 for aspect sentiment classification.")
picture(s, R2 / "figures" / "asc_cross_dataset_macro_f1.png", Inches(0.45), Inches(1.85), Inches(12.4), Inches(3.4))
best_x = asc[asc.train != asc.test].sort_values("macro_f1").groupby(["train", "test"]).tail(1)
lines = [f"Best {('Laptop → MAMS' if r.train == 'semeval' else 'MAMS → Laptop')}: {r.model}, {100 * r.macro_f1:.1f} macro-F1"
         for r in best_x.itertuples()]
card(s, Inches(0.55), Inches(5.35), Inches(12.25), Inches(1.6), "Takeaway", "Transfer is asymmetric for every model family",
     lines + ["Training on multi-polarity text (MAMS) produces a model that also works on laptops; the reverse needs data, not tricks"],
     size=13)
notes(s, "Every row of heatmaps shows the same pattern, from TF-IDF to RoBERTa-large: training data shape matters more than architecture for transfer.")

# 14 ---- error analysis
s = slide("Error analysis", "Which sentences are still hard?",
          "Accuracy of aspect sentiment by linguistic phenomenon, in-domain test sets.")
picture(s, R2 / "error_analysis" / "phenomena_semeval_to_semeval.png", Inches(0.45), Inches(1.8), Inches(8.2), Inches(2.6))
picture(s, R2 / "error_analysis" / "phenomena_mams_to_mams.png", Inches(0.45), Inches(4.4), Inches(8.2), Inches(2.6))
ph = ["negation", "contrast", "multi_aspect", "multi_polarity", "implicit", "long_range"]
bm = V2L if V2L in set(ea.model) else V1R
lines = []
for d, lab in [("semeval", "Laptop"), ("mams", "MAMS")]:
    r = ea[(ea.model == bm) & (ea.train == d) & (ea.test == d)].iloc[0]
    worst = min(ph, key=lambda p: r[p])
    lines.append(f"{lab}: hardest = {worst.replace('_', '-')} ({100 * r[worst]:.0f}% vs {100 * r['all']:.0f}% overall)")
r_v1 = ea[(ea.model == V1R) & (ea.train == "semeval") & (ea.test == "semeval")].iloc[0]
r_v2 = ea[(ea.model == bm) & (ea.train == "semeval") & (ea.test == "semeval")].iloc[0]
lines.insert(0, f"Mixed-polarity Laptop sentences: {100 * r_v1.multi_polarity:.0f}% (v1) → {100 * r_v2.multi_polarity:.0f}% (v2), "
                f"the largest gain of any group")
card(s, Inches(8.85), Inches(1.85), Inches(3.95), Inches(5.1), "Best model: " + bm, "What changed, what remains",
     lines + ["Tags are rule-based: negators, contrast words, Liu's opinion lexicon, distance to nearest opinion word",
              "Misclassified examples per model are saved for inspection"], size=12.5)
notes(s, "Read the hardest category per dataset and give one concrete misclassified example from the error CSVs.")

# 15 ---- demo
s = slide("Demo", "From raw text to aspect-level opinions",
          "RoBERTa-base v2 extractor + classifier trained on Laptops, run on new sentences (notebook section 9).")
if demo is not None and len(demo):
    rows = [[r.sentence if len(r.sentence) < 78 else r.sentence[:75] + "…", r.aspect, r.polarity, f"{r.confidence:.2f}"]
            for r in demo.head(11).itertuples()]
    gt = table(s, Inches(0.55), Inches(1.9), Inches(12.25), ["Sentence", "Aspect", "Polarity", "Conf."], rows,
               col_w=[8.0, 2.0, 1.3, 0.9], row_h=0.42, font=12.5, first_col_bold=False)
    for i, r in enumerate(rows, start=1):
        c = gt.cell(i, 2).text_frame.paragraphs[0].runs[0].font
        c.color.rgb = rgb({"positive": GREEN, "negative": RED}.get(r[2], SLATE))
        c.bold = True
else:
    text(s, Inches(0.55), Inches(2), Inches(12), Inches(1), [{"text": "Run the notebook's demo section to fill this slide.", "color": RED}])
notes(s, "Show the live notebook demo in the video. The proposal's motivating example is the first row.")

# 16 ---- conclusions
s = slide("Conclusions", "What we learned")
card(s, Inches(0.55), Inches(1.55), Inches(6.0), Inches(5.3), "Findings", "Results",
     [f"Fine-tuned Transformers beat TF-IDF by {100 * gain_m:.0f}–{100 * gain:.0f} macro-F1 points in-domain",
      f"Best in-domain macro-F1: {100 * best_l.macro_f1:.1f} (Laptop), {100 * best_m.macro_f1:.1f} (MAMS)",
      "RoBERTa-large v2 beats the best v1 model in all 4 sentiment configurations; v2 improves end-to-end F1 everywhere",
      f"Biggest v2 gain: mixed-polarity Laptop sentences, {100 * r_v1.multi_polarity:.0f}% → {100 * r_v2.multi_polarity:.0f}%",
      "Transfer depends on the shape of the training data more than on the model",
      "Aspect extraction does not survive a domain change"], size=13)
card(s, Inches(6.8), Inches(1.55), Inches(6.0), Inches(5.3), "Limitations & next steps", "Honest caveats",
     ["One random seed per run: differences of 1–2 points may not be significant",
      f"v2 hurt Laptop → MAMS extraction ({pts(v(ate, V2B, 'semeval', 'mams', 'f1') - v(ate, V1R, 'semeval', 'mams', 'f1'))}) "
      f"and base-v2 MAMS → Laptop sentiment ({pts(v(asc, V2B, 'mams', 'semeval', 'macro_f1') - v(asc, V1R, 'mams', 'semeval', 'macro_f1'))})",
      "Small Laptop validation set (221 aspects) makes checkpoint choice noisy; the train–val gap grew there",
      "Phenomenon tags are heuristic, not hand-annotated",
      "Next: multi-seed averages, domain-adaptive pre-training on unlabeled reviews, joint ATE+ASC model, "
      "training on both corpora together"], accent=SLATE)
notes(s, "Summarise in three sentences, then state limitations clearly.")

# 17 ---- close
s = prs.slides.add_slide(BLANK)
_n[0] += 1
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, 0, Inches(0.12), H, TEAL)
text(s, Inches(0.8), Inches(1.6), Inches(11.5), Inches(1.2),
     [{"text": "Thank you", "size": 48, "bold": True, "font": HEAD, "color": WHITE}])
text(s, Inches(0.8), Inches(2.8), Inches(11.5), Inches(0.6), [{"text": "Questions welcome", "size": 20, "color": LIGHT2}])
text(s, Inches(0.8), Inches(4.0), Inches(11.5), Inches(2.5),
     [{"text": "REPRODUCE", "size": 12, "bold": True, "color": TEAL_LT},
      {"text": "pip install -r requirements.txt", "size": 15, "font": "Consolas", "color": WHITE},
      {"text": "python run_all.py      # v1 study", "size": 15, "font": "Consolas", "color": WHITE},
      {"text": "python run_v2.py       # v2 study", "size": 15, "font": "Consolas", "color": WHITE},
      {"text": "ABSA_Notebook.ipynb    # results, figures and live demo", "size": 15, "font": "Consolas", "color": WHITE}])
text(s, Inches(0.8), Inches(6.5), Inches(11.5), Inches(0.5),
     [{"text": "Mrigank · Varshit · Jeshwanth", "size": 14, "color": MUTED}])

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
