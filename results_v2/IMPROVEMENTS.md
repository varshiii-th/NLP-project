# v2 improvements over v1

Reference = v1 RoBERTa-base (the best v1 model). Δ = v2 − v1 (absolute points on the 0–1 scale).

## What changed and why

| v1 observation | v2 change |
|---|---|
| Train macro-F1 → 0.94–0.98 while val plateaus at epoch ~2 and val loss rises (over-fitting) | label smoothing 0.1, FGM adversarial training on embeddings (ε=0.5), layer-wise LR decay so lower layers move less |
| [CLS]-only head cannot tell which aspect it is judging in multi-aspect sentences (MAMS: every sentence) | aspect wrapped with `#` markers in the sentence + head on [CLS] ⊕ mean(aspect sub-word states) |
| Val F1 jumps ±3 pts between checks on SemEval's 221-aspect val split | 4 validation checks per epoch, batch 16 (more updates), patience in epochs |
| ATE decoding could emit invalid `O → I` sequences | BIO-constrained Viterbi decoding |
| Base-size encoder | RoBERTa-large (bf16, gradient checkpointing to fit the 8 GB GPU) |

## ASC macro-F1

| train | test | setting | v1 RoBERTa-base | RoBERTa-base v2 | Δ RoBERTa-base v2 | RoBERTa-large v2 | Δ RoBERTa-large v2 |
|---|---|---|---|---|---|---|---|
| semeval | semeval | in-domain | 0.7748 | 0.7790 | +0.0043 | 0.7988 | +0.0240 |
| semeval | mams | cross-dataset | 0.4824 | 0.5739 | +0.0915 | 0.5416 | +0.0592 |
| mams | semeval | cross-dataset | 0.7208 | 0.6911 | -0.0297 | 0.7321 | +0.0113 |
| mams | mams | in-domain | 0.8403 | 0.8449 | +0.0046 | 0.8506 | +0.0103 |

## ASC accuracy

| train | test | setting | v1 RoBERTa-base | RoBERTa-base v2 | Δ RoBERTa-base v2 | RoBERTa-large v2 | Δ RoBERTa-large v2 |
|---|---|---|---|---|---|---|---|
| semeval | semeval | in-domain | 0.8041 | 0.8197 | +0.0157 | 0.8323 | +0.0282 |
| semeval | mams | cross-dataset | 0.4873 | 0.5831 | +0.0958 | 0.5561 | +0.0689 |
| mams | semeval | cross-dataset | 0.7539 | 0.7273 | -0.0266 | 0.7618 | +0.0078 |
| mams | mams | in-domain | 0.8451 | 0.8503 | +0.0052 | 0.8563 | +0.0112 |

## ASC neutral-class F1 (weakest class in v1)

| train | test | setting | v1 RoBERTa-base | RoBERTa-base v2 | Δ RoBERTa-base v2 | RoBERTa-large v2 | Δ RoBERTa-large v2 |
|---|---|---|---|---|---|---|---|
| semeval | semeval | in-domain | 0.6809 | 0.6272 | -0.0537 | 0.6906 | +0.0097 |
| semeval | mams | cross-dataset | 0.3313 | 0.4080 | +0.0766 | 0.3157 | -0.0157 |
| mams | semeval | cross-dataset | 0.6528 | 0.6104 | -0.0424 | 0.6715 | +0.0187 |
| mams | mams | in-domain | 0.8645 | 0.8724 | +0.0079 | 0.8765 | +0.0120 |

## ATE span F1

| train | test | setting | v1 RoBERTa-base | RoBERTa-base v2 | Δ RoBERTa-base v2 | RoBERTa-large v2 | Δ RoBERTa-large v2 |
|---|---|---|---|---|---|---|---|
| semeval | semeval | in-domain | 0.8591 | 0.8456 | -0.0135 | 0.8607 | +0.0016 |
| semeval | mams | cross-dataset | 0.4396 | 0.3825 | -0.0571 | 0.3808 | -0.0588 |
| mams | semeval | cross-dataset | 0.3948 | 0.4149 | +0.0201 | 0.4032 | +0.0084 |
| mams | mams | in-domain | 0.7620 | 0.7801 | +0.0181 | 0.7683 | +0.0063 |

## End-to-end ATE → ASC pair F1

| train | test | setting | v1 RoBERTa-base | RoBERTa-base v2 | Δ RoBERTa-base v2 | RoBERTa-large v2 | Δ RoBERTa-large v2 |
|---|---|---|---|---|---|---|---|
| semeval | semeval | in-domain | 0.6914 | 0.7017 | +0.0103 | 0.7095 | +0.0181 |
| semeval | mams | cross-dataset | 0.2757 | 0.2883 | +0.0126 | 0.2950 | +0.0193 |
| mams | semeval | cross-dataset | 0.2986 | 0.3079 | +0.0093 | 0.3257 | +0.0271 |
| mams | mams | in-domain | 0.6550 | 0.6713 | +0.0163 | 0.6619 | +0.0069 |

## Generalisation gap at the selected checkpoint (train − val main metric)

| model | task | train | best_epoch | train_metric | val_metric | gap |
|---|---|---|---|---|---|---|
| BERT-base v1 | asc | mams | 5.5000 | 0.9876 | 0.8243 | 0.1634 |
| RoBERTa-base v1 | asc | mams | 5.0000 | 0.9822 | 0.8306 | 0.1516 |
| RoBERTa-base v2 | asc | mams | 1.5000 | 0.9104 | 0.8322 | 0.0782 |
| RoBERTa-large v2 | asc | mams | 2.7500 | 0.9602 | 0.8315 | 0.1287 |
| BERT-base v1 | asc | semeval | 3.5000 | 0.8926 | 0.7095 | 0.1830 |
| RoBERTa-base v1 | asc | semeval | 2.5000 | 0.8374 | 0.7877 | 0.0497 |
| RoBERTa-base v2 | asc | semeval | 4.5040 | 0.9944 | 0.7758 | 0.2186 |
| RoBERTa-large v2 | asc | semeval | 1.7480 | 0.9186 | 0.7910 | 0.1275 |
| BERT-base v1 | ate | mams | 5.0000 | 0.9728 | 0.7761 | 0.1967 |
| RoBERTa-base v1 | ate | mams | 6.5040 | 0.9591 | 0.7840 | 0.1751 |
| RoBERTa-base v2 | ate | mams | 3.0000 | 0.8742 | 0.7893 | 0.0849 |
| RoBERTa-large v2 | ate | mams | 5.0000 | 0.9296 | 0.7802 | 0.1494 |
| BERT-base v1 | ate | semeval | 5.5000 | 0.9977 | 0.8294 | 0.1683 |
| RoBERTa-base v1 | ate | semeval | 8.0000 | 0.9993 | 0.8657 | 0.1336 |
| RoBERTa-base v2 | ate | semeval | 3.7500 | 0.9886 | 0.8756 | 0.1130 |
| RoBERTa-large v2 | ate | semeval | 4.0000 | 0.9948 | 0.8596 | 0.1352 |

Figures: `improvements/v1_vs_v2_*.png`, `improvements/*_val_curves_v1_vs_v2.png`; per-run training curves: `transformers/<task>/<run>/training_curves.png`; full tables: `summary.md`.
