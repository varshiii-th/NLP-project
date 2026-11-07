# Aspect-Based Sentiment Analysis — TF-IDF vs BERT vs RoBERTa

Aspect term extraction (ATE, BIO sequence labelling) and aspect-level
sentiment classification (ASC, sentence + aspect pair input) on **SemEval-2014 Task 4 Laptops** and **MAMS**,
with in-domain and cross-dataset evaluation, an end-to-end ATE→ASC pipeline, and error analysis.




## Run

```bash
pip install -r requirements.txt          # needs the CUDA build of PyTorch
python run_all.py                        # full pipeline (MAMS is downloaded automatically)
python run_all.py --quick                # 1-epoch smoke test on 10% of the data -> results_quick/
python run_all.py --models roberta --tasks asc --fresh --epochs 8   # selective re-runs
python run_v2.py                         # v2 round: improved recipe, RoBERTa-base + RoBERTa-large -> results_v2/
python run_v2.py --report-only           # rebuild v2 tables/figures from finished runs (no training)
```

Measured on an RTX 4060 Laptop GPU (8 GB): v1 ≈ 17 min, v2 ≈ 75 min of training.


Finished transformer runs are skipped on re-run (resumable); pass `--fresh` to retrain, `--save-models` to keep checkpoints.

## Layout

| path | purpose |
|---|---|
| `absa/data.py` | parallel XML parsing (process pool), word tokenisation with offsets, BIO tagging, unified schema, caching; SemEval val split = 10% of train sentences |
| `absa/eda.py` | polarity balance, aspects/sentence, sentence length, top aspects, vocabulary overlap |
| `absa/baselines.py` | TF-IDF (sentence + aspect-local context + aspect) → LR / Linear SVM; feature-based LR BIO tagger; C tuned on validation in parallel |
| `absa/transformer_trainer.py` | GPU fine-tuning (bf16 autocast, TF32, fused AdamW, GPU-resident data, dynamic padding), evaluation twice per epoch on train & val, early stopping on val macro-F1 / span-F1, best-checkpoint restore |
| `absa/metrics.py` | accuracy, macro P/R/F1, per-class, confusion matrix; exact-match span P/R/F1; aspect-polarity pair F1 |
| `absa/error_analysis.py` | per-instance tags: negation, contrast, multi-aspect, multi-polarity, implicit (no opinion-lexicon word), long-range (opinion word ≥5 tokens away) |
| `run_all.py` | orchestrates everything and writes the report |

## Outputs (`results/`)

* `summary.md` — all result tables; `tables/*.csv` — ASC, ATE, end-to-end results
* `transformers/<task>/<model>_train-<dataset>/training_curves.png` — train/val loss (per-step + per-eval) and train/val metrics with the best checkpoint marked; `history.csv` (updated live during training), `lr_schedule.png`, `confusion_test_*.png`
* `figures/` — cross-dataset heatmaps, model comparison bars, per-class F1
* `baselines/` — C-sweep validation curves, confusion matrices
* `eda/` — dataset figures and statistics
* `error_analysis/` — accuracy by phenomenon (CSV + figures), misclassified examples per model/config
* `run_log.txt` — full progress log

Notes: SemEval's `conflict` polarity (45 train instances) is dropped for ASC — MAMS has no such class, so
cross-dataset evaluation needs a shared 3-class label set. Aspect extraction keeps all annotated aspects.
