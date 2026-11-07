# ABSA results summary - v2

Settings: in-domain = trained & tested on the same corpus; cross-dataset = transfer.

## Aspect-level sentiment classification (test sets)

| model | train | test | setting | accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|---|---|
| BERT-base v1 | mams | mams | in-domain | 0.8368 | 0.8297 | 0.8360 | 0.8326 |
| RoBERTa-base v1 | mams | mams | in-domain | 0.8451 | 0.8400 | 0.8416 | 0.8403 |
| RoBERTa-base v2 | mams | mams | in-domain | 0.8503 | 0.8448 | 0.8451 | 0.8449 |
| RoBERTa-large v2 | mams | mams | in-domain | 0.8563 | 0.8517 | 0.8496 | 0.8506 |
| TF-IDF+LR | mams | mams | in-domain | 0.7148 | 0.7035 | 0.7025 | 0.7030 |
| TF-IDF+SVM | mams | mams | in-domain | 0.7253 | 0.7160 | 0.7100 | 0.7127 |
| BERT-base v1 | mams | semeval | cross-dataset | 0.6991 | 0.6837 | 0.6937 | 0.6769 |
| RoBERTa-base v1 | mams | semeval | cross-dataset | 0.7539 | 0.7238 | 0.7273 | 0.7208 |
| RoBERTa-base v2 | mams | semeval | cross-dataset | 0.7273 | 0.7003 | 0.6979 | 0.6911 |
| RoBERTa-large v2 | mams | semeval | cross-dataset | 0.7618 | 0.7453 | 0.7409 | 0.7321 |
| TF-IDF+LR | mams | semeval | cross-dataset | 0.5925 | 0.5503 | 0.5303 | 0.5234 |
| TF-IDF+SVM | mams | semeval | cross-dataset | 0.5784 | 0.5350 | 0.5198 | 0.5112 |
| BERT-base v1 | semeval | mams | cross-dataset | 0.5165 | 0.5205 | 0.5466 | 0.5180 |
| RoBERTa-base v1 | semeval | mams | cross-dataset | 0.4873 | 0.5135 | 0.5479 | 0.4824 |
| RoBERTa-base v2 | semeval | mams | cross-dataset | 0.5831 | 0.6365 | 0.6522 | 0.5739 |
| RoBERTa-large v2 | semeval | mams | cross-dataset | 0.5561 | 0.6028 | 0.6374 | 0.5416 |
| TF-IDF+LR | semeval | mams | cross-dataset | 0.4476 | 0.4923 | 0.4965 | 0.4465 |
| TF-IDF+SVM | semeval | mams | cross-dataset | 0.4274 | 0.4875 | 0.4871 | 0.4189 |
| BERT-base v1 | semeval | semeval | in-domain | 0.7759 | 0.7362 | 0.7481 | 0.7321 |
| RoBERTa-base v1 | semeval | semeval | in-domain | 0.8041 | 0.7660 | 0.7951 | 0.7748 |
| RoBERTa-base v2 | semeval | semeval | in-domain | 0.8197 | 0.7871 | 0.7960 | 0.7790 |
| RoBERTa-large v2 | semeval | semeval | in-domain | 0.8323 | 0.7971 | 0.8165 | 0.7988 |
| TF-IDF+LR | semeval | semeval | in-domain | 0.6708 | 0.6171 | 0.6123 | 0.6067 |
| TF-IDF+SVM | semeval | semeval | in-domain | 0.6646 | 0.6178 | 0.6058 | 0.5918 |

## Aspect term extraction (exact-match spans, test sets)

| model | train | test | setting | precision | recall | f1 |
|---|---|---|---|---|---|---|
| BERT-base v1 | mams | mams | in-domain | 0.7177 | 0.7859 | 0.7503 |
| RoBERTa-base v1 | mams | mams | in-domain | 0.7415 | 0.7837 | 0.7620 |
| RoBERTa-base v2 | mams | mams | in-domain | 0.7299 | 0.8376 | 0.7801 |
| RoBERTa-large v2 | mams | mams | in-domain | 0.7345 | 0.8054 | 0.7683 |
| TF-IDF+LR | mams | mams | in-domain | 0.6678 | 0.7058 | 0.6863 |
| BERT-base v1 | mams | semeval | cross-dataset | 0.2850 | 0.3547 | 0.3161 |
| RoBERTa-base v1 | mams | semeval | cross-dataset | 0.3337 | 0.4832 | 0.3948 |
| RoBERTa-base v2 | mams | semeval | cross-dataset | 0.3568 | 0.4954 | 0.4149 |
| RoBERTa-large v2 | mams | semeval | cross-dataset | 0.4228 | 0.3853 | 0.4032 |
| TF-IDF+LR | mams | semeval | cross-dataset | 0.2398 | 0.0719 | 0.1106 |
| BERT-base v1 | semeval | mams | cross-dataset | 0.7239 | 0.1452 | 0.2419 |
| RoBERTa-base v1 | semeval | mams | cross-dataset | 0.7606 | 0.3091 | 0.4396 |
| RoBERTa-base v2 | semeval | mams | cross-dataset | 0.7629 | 0.2552 | 0.3825 |
| RoBERTa-large v2 | semeval | mams | cross-dataset | 0.8063 | 0.2493 | 0.3808 |
| TF-IDF+LR | semeval | mams | cross-dataset | 0.5483 | 0.1190 | 0.1956 |
| BERT-base v1 | semeval | semeval | in-domain | 0.8054 | 0.8226 | 0.8139 |
| RoBERTa-base v1 | semeval | semeval | in-domain | 0.8651 | 0.8532 | 0.8591 |
| RoBERTa-base v2 | semeval | semeval | in-domain | 0.8583 | 0.8333 | 0.8456 |
| RoBERTa-large v2 | semeval | semeval | in-domain | 0.8814 | 0.8410 | 0.8607 |
| TF-IDF+LR | semeval | semeval | in-domain | 0.6252 | 0.5382 | 0.5785 |

## End-to-end ATE -> ASC (aspect-polarity pair F1)

| model | train | test | setting | precision | recall | f1 |
|---|---|---|---|---|---|---|
| BERT-base v1 | mams | mams | in-domain | 0.6145 | 0.6729 | 0.6424 |
| RoBERTa-base v1 | mams | mams | in-domain | 0.6374 | 0.6737 | 0.6550 |
| RoBERTa-base v2 | mams | mams | in-domain | 0.6282 | 0.7208 | 0.6713 |
| RoBERTa-large v2 | mams | mams | in-domain | 0.6328 | 0.6939 | 0.6619 |
| TF-IDF+LR | mams | mams | in-domain | 0.5071 | 0.5359 | 0.5211 |
| TF-IDF+SVM | mams | mams | in-domain | 0.5149 | 0.5442 | 0.5291 |
| BERT-base v1 | mams | semeval | cross-dataset | 0.2207 | 0.2774 | 0.2458 |
| RoBERTa-base v1 | mams | semeval | cross-dataset | 0.2511 | 0.3683 | 0.2986 |
| RoBERTa-base v2 | mams | semeval | cross-dataset | 0.2637 | 0.3699 | 0.3079 |
| RoBERTa-large v2 | mams | semeval | cross-dataset | 0.3408 | 0.3119 | 0.3257 |
| TF-IDF+LR | mams | semeval | cross-dataset | 0.1414 | 0.0423 | 0.0651 |
| TF-IDF+SVM | mams | semeval | cross-dataset | 0.1414 | 0.0423 | 0.0651 |
| BERT-base v1 | semeval | mams | cross-dataset | 0.4254 | 0.0853 | 0.1421 |
| RoBERTa-base v1 | semeval | mams | cross-dataset | 0.4770 | 0.1939 | 0.2757 |
| RoBERTa-base v2 | semeval | mams | cross-dataset | 0.5749 | 0.1924 | 0.2883 |
| RoBERTa-large v2 | semeval | mams | cross-dataset | 0.6247 | 0.1931 | 0.2950 |
| TF-IDF+LR | semeval | mams | cross-dataset | 0.3034 | 0.0659 | 0.1082 |
| TF-IDF+SVM | semeval | mams | cross-dataset | 0.2828 | 0.0614 | 0.1009 |
| BERT-base v1 | semeval | semeval | in-domain | 0.6279 | 0.6426 | 0.6352 |
| RoBERTa-base v1 | semeval | semeval | in-domain | 0.6963 | 0.6865 | 0.6914 |
| RoBERTa-base v2 | semeval | semeval | in-domain | 0.7124 | 0.6912 | 0.7017 |
| RoBERTa-large v2 | semeval | semeval | in-domain | 0.7270 | 0.6928 | 0.7095 |
| TF-IDF+LR | semeval | semeval | in-domain | 0.4465 | 0.3856 | 0.4138 |
| TF-IDF+SVM | semeval | semeval | in-domain | 0.4374 | 0.3777 | 0.4054 |

## ASC accuracy by linguistic phenomenon

| model | train | test | all | negation | contrast | multi_aspect | multi_polarity | implicit | long_range |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF+LR | semeval | semeval | 0.6708 | 0.5175 | 0.5190 | 0.6623 | 0.4854 | 0.5634 | 0.6268 |
| TF-IDF+LR | semeval | mams | 0.4476 | 0.4568 | 0.4179 | 0.4476 | 0.4476 | 0.4505 | 0.3849 |
| TF-IDF+SVM | semeval | semeval | 0.6646 | 0.5385 | 0.5316 | 0.6649 | 0.4563 | 0.5352 | 0.6056 |
| TF-IDF+SVM | semeval | mams | 0.4274 | 0.4273 | 0.4034 | 0.4274 | 0.4274 | 0.4359 | 0.3401 |
| TF-IDF+LR | mams | semeval | 0.5925 | 0.4336 | 0.5190 | 0.5831 | 0.5243 | 0.5563 | 0.5141 |
| TF-IDF+LR | mams | mams | 0.7148 | 0.6750 | 0.6932 | 0.7148 | 0.7148 | 0.7839 | 0.7149 |
| TF-IDF+SVM | mams | semeval | 0.5784 | 0.4126 | 0.4810 | 0.5567 | 0.4951 | 0.5423 | 0.4859 |
| TF-IDF+SVM | mams | mams | 0.7253 | 0.6886 | 0.7005 | 0.7253 | 0.7253 | 0.7802 | 0.7210 |
| BERT-base v1 | mams | semeval | 0.6991 | 0.5944 | 0.6329 | 0.6913 | 0.6796 | 0.6197 | 0.6338 |
| BERT-base v1 | mams | mams | 0.8368 | 0.8136 | 0.8382 | 0.8368 | 0.8368 | 0.8718 | 0.8289 |
| BERT-base v1 | semeval | semeval | 0.7759 | 0.6713 | 0.6835 | 0.7652 | 0.5243 | 0.6831 | 0.6972 |
| BERT-base v1 | semeval | mams | 0.5165 | 0.4955 | 0.5000 | 0.5165 | 0.5165 | 0.4835 | 0.4705 |
| RoBERTa-base v1 | mams | semeval | 0.7539 | 0.6993 | 0.6962 | 0.7625 | 0.7767 | 0.6620 | 0.6690 |
| RoBERTa-base v1 | mams | mams | 0.8451 | 0.8227 | 0.8406 | 0.8451 | 0.8451 | 0.8571 | 0.8473 |
| RoBERTa-base v1 | semeval | semeval | 0.8041 | 0.6993 | 0.6329 | 0.7968 | 0.5728 | 0.7465 | 0.7606 |
| RoBERTa-base v1 | semeval | mams | 0.4873 | 0.4659 | 0.4300 | 0.4873 | 0.4873 | 0.4872 | 0.3890 |
| RoBERTa-base v2 | semeval | semeval | 0.8197 | 0.7413 | 0.7342 | 0.8179 | 0.7184 | 0.7535 | 0.7183 |
| RoBERTa-base v2 | semeval | mams | 0.5831 | 0.5773 | 0.6232 | 0.5831 | 0.5831 | 0.5275 | 0.4929 |
| RoBERTa-large v2 | semeval | semeval | 0.8323 | 0.7133 | 0.7848 | 0.8391 | 0.7379 | 0.7817 | 0.7887 |
| RoBERTa-large v2 | semeval | mams | 0.5561 | 0.5432 | 0.5894 | 0.5561 | 0.5561 | 0.4652 | 0.4603 |
| RoBERTa-base v2 | mams | semeval | 0.7273 | 0.6643 | 0.6582 | 0.7388 | 0.7573 | 0.6338 | 0.6831 |
| RoBERTa-base v2 | mams | mams | 0.8503 | 0.8455 | 0.8792 | 0.8503 | 0.8503 | 0.8755 | 0.8391 |
| RoBERTa-large v2 | mams | semeval | 0.7618 | 0.6853 | 0.6835 | 0.7625 | 0.7573 | 0.7113 | 0.7113 |
| RoBERTa-large v2 | mams | mams | 0.8563 | 0.8295 | 0.8357 | 0.8563 | 0.8563 | 0.8718 | 0.8493 |

Phenomenon counts in the test sets: {"semeval": {"negation": 143, "contrast": 79, "multi_aspect": 379, "multi_polarity": 103, "implicit": 142, "long_range": 142}, "mams": {"negation": 440, "contrast": 414, "multi_aspect": 1336, "multi_polarity": 1336, "implicit": 273, "long_range": 491}}

## Transformer training time (s)

{
 "ate/roberta-v2/semeval": 164.2,
 "asc/roberta-v2/semeval": 103.2,
 "ate/roberta-large-v2/semeval": 476.3,
 "asc/roberta-large-v2/semeval": 275.7,
 "ate/roberta-v2/mams": 243.3,
 "asc/roberta-v2/mams": 353.8,
 "ate/roberta-large-v2/mams": 686.6,
 "asc/roberta-large-v2/mams": 1199.3
}
