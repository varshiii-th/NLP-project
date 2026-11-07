"""End-to-end inference: raw review sentence -> [(aspect, polarity, confidence)].

Uses an ATE model (BIO tagger with constrained Viterbi decoding) followed by an ASC model
(aspect-marked input + aspect-pooled head). Models are trained once with the v2 RoBERTa-base
recipe on SemEval-14 Laptops and cached under models/demo/.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .config import ROOT, TrainConfig
from .data import _tokenize_and_tag, predicted_aspect_frame
from .metrics import bio_to_spans
from .transformer_trainer import Trainer, encode_ate, run_inference, encode_asc

DEMO_DIR = ROOT / "models" / "demo"


def _demo_cfgs(train_on: str):
    import run_v2  # recipe lives in one place
    v = run_v2.VARIANTS["roberta-v2"]
    return {t: TrainConfig(task=t, model_key=v["model_key"], **v[t]) for t in ("ate", "asc")}


def train_or_load(data: dict | None = None, train_on: str = "semeval", log=print) -> "ABSAPipeline":
    out = DEMO_DIR / train_on
    trainers = {}
    for task, cfg in _demo_cfgs(train_on).items():
        ck = out / f"{task}.pt"
        tr = Trainer(cfg, out / task, log=lambda *a: None)
        if ck.exists():
            tr.model.load_state_dict(torch.load(ck, map_location="cuda"))
            log(f"[demo] loaded {task.upper()} model from {ck}")
        else:
            if data is None:
                from .data import load_datasets
                data = load_datasets()
            key = "asc" if task == "asc" else "sents"
            log(f"[demo] training {task.upper()} model on {train_on} (one-off, a few minutes)...")
            tr.log = log
            tr.fit(data[train_on]["train"][key], data[train_on]["val"][key], f"demo {task.upper()}")
            out.mkdir(parents=True, exist_ok=True)
            torch.save(tr.model.state_dict(), ck)
            (out / f"{task}_config.json").write_text(json.dumps(asdict(cfg), indent=1))
        tr.model.eval()
        trainers[task] = tr
    return ABSAPipeline(trainers["ate"], trainers["asc"])


class ABSAPipeline:
    def __init__(self, ate: Trainer, asc: Trainer):
        self.ate, self.asc = ate, asc

    @torch.no_grad()
    def __call__(self, sentences: list[str] | str) -> pd.DataFrame:
        if isinstance(sentences, str):
            sentences = [sentences]
        sents = [_tokenize_and_tag({"sid": str(i), "text": t, "aspects": []}) for i, t in enumerate(sentences)]
        d = encode_ate(self.ate.tok, sents, self.ate.cfg.max_len, self.ate.device)
        _, tags = run_inference(self.ate.model, d, "ate", 64, viterbi=True)
        frame = predicted_aspect_frame(sents, [bio_to_spans(t) for t in tags])
        if len(frame) == 0:
            return pd.DataFrame(columns=["sentence", "aspect", "polarity", "confidence"])
        ad = encode_asc(self.asc.tok, frame, self.asc.cfg.max_len, self.asc.device,
                        self.asc.cfg.aspect_markers, self.asc.cfg.aspect_pool)
        _, (pred, probs) = run_inference(self.asc.model, ad, "asc", 64)
        from .config import POLARITIES
        return pd.DataFrame({"sentence": [sentences[i] for i in frame.sent_idx], "aspect": frame.aspect,
                             "polarity": [POLARITIES[p] for p in pred], "confidence": probs.max(1).round(3)})
