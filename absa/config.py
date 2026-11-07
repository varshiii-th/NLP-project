"""Central configuration: paths, label sets, hyper-parameters."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

DATA_FILES = {
    "semeval": {
        "train": DATA_DIR / "Laptop_Train_v2.xml",
        "test": DATA_DIR / "Laptops_Test_Gold.xml",
    },
    "mams": {
        "train": DATA_DIR / "MAMS" / "train.xml",
        "val": DATA_DIR / "MAMS" / "val.xml",
        "test": DATA_DIR / "MAMS" / "test.xml",
    },
}
MAMS_URL = "https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/{split}.xml"
DATASET_NAMES = {"semeval": "SemEval-14 Laptop", "mams": "MAMS"}

# Sentiment polarity labels. SemEval's rare "conflict" class (45 train instances) is dropped,
# as is standard practice and because MAMS has no such class (needed for cross-dataset tests).
POLARITIES = ["negative", "neutral", "positive"]
POL2ID = {p: i for i, p in enumerate(POLARITIES)}

BIO_TAGS = ["O", "B", "I"]
TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}

TRANSFORMER_MODELS = {"bert": "bert-base-uncased", "roberta": "roberta-base", "roberta-large": "roberta-large"}

SEED = 42
SEMEVAL_VAL_FRACTION = 0.1  # SemEval ships no validation split -> hold out 10% of train sentences

N_CPU = os.cpu_count() or 4


@dataclass
class TrainConfig:
    task: str                 # "asc" (aspect sentiment) or "ate" (aspect term extraction)
    model_key: str            # "bert" | "roberta"
    epochs: int = 6
    batch_size: int = 32
    eval_batch_size: int = 128
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_len: int = 128
    patience: int = 3         # early stopping: epochs without val improvement (checked every eval)
    evals_per_epoch: int = 2  # validation checks per epoch (finer-grained curves)
    seed: int = SEED
    # ---- v2 options (defaults reproduce the v1 recipe exactly) ----
    aspect_markers: bool = False   # ASC: wrap the aspect in the sentence with "# ... #" punctuation markers
    aspect_pool: bool = False      # ASC: classify from [CLS] ++ mean(aspect sub-word states) instead of [CLS] only
    label_smoothing: float = 0.0
    llrd: float = 1.0              # layer-wise LR decay factor (1.0 = off)
    head_lr: float | None = None   # LR for the freshly initialised head (None = same as lr)
    fgm_eps: float = 0.0           # FGM adversarial training on word embeddings (0 = off)
    viterbi: bool = False          # ATE: BIO-constrained decoding (no O->I transitions)
    grad_ckpt: bool = False        # gradient checkpointing (memory for large models)
    train_eval_max: int | None = None  # cap #train examples used for the train-metric curve (speed)
    extra: dict = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return TRANSFORMER_MODELS[self.model_key]


DEFAULT_HPARAMS = {
    "asc": dict(epochs=6, lr=2e-5, batch_size=32),
    "ate": dict(epochs=8, lr=5e-5, batch_size=32),
}
