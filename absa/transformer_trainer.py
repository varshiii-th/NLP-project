"""Fine-tuning BERT / RoBERTa for
   * ASC - aspect-level sentiment classification, input = (sentence, aspect) sentence pair
   * ATE - aspect term extraction, BIO token classification (label on the first sub-word of each word)

GPU throughput tricks: the whole tokenised dataset lives on the GPU (no DataLoader / host->device copies),
each batch is trimmed to its longest sequence, bf16 autocast + TF32 matmuls, fused AdamW, and loss values
are only synchronised to the host at evaluation points.

v2 options (all off by default -> v1 recipe): aspect markers + aspect-pooled head, label smoothing,
layer-wise LR decay, FGM adversarial training, BIO-constrained Viterbi decoding, gradient checkpointing.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # Rust tokenizer uses all cores
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification,  # noqa: E402
                          AutoTokenizer, get_linear_schedule_with_warmup)
from transformers.utils import logging as hf_logging  # noqa: E402

from .config import BIO_TAGS, DATASET_NAMES, POLARITIES, TrainConfig  # noqa: E402
from .metrics import bio_to_spans, classification_metrics, span_prf  # noqa: E402
from .plotting import plot_confusion_matrix, plot_lr, plot_training_curves  # noqa: E402

hf_logging.set_verbosity_error()
MODEL_LABEL = {"bert": "BERT", "roberta": "RoBERTa", "roberta-large": "RoBERTa-large"}
MARK_L, MARK_R = "# ", " #"


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available - this pipeline is configured to train on the GPU.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def amp_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# --------------------------------------------------------------------------- models
class AspectPooledClassifier(nn.Module):
    """Encoder -> concat([CLS], mean of the aspect's sub-word states) -> MLP -> 3 polarities.
    The aspect pooling tells the head *which* span the sentiment must be read for (multi-aspect sentences)."""

    def __init__(self, name: str, n_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name, add_pooling_layer=False)
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * h, h), nn.Tanh(),
                                  nn.Dropout(dropout), nn.Linear(h, n_labels))

    def forward(self, input_ids, attention_mask, aspect_mask):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = aspect_mask.unsqueeze(-1).to(hs.dtype)
        asp = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
        return SimpleNamespace(logits=self.head(torch.cat([hs[:, 0], asp], dim=-1)))


# --------------------------------------------------------------------------- encoding
class GPUData:
    """Padded tensors resident on the GPU + per-row true lengths for dynamic batch trimming."""

    def __init__(self, input_ids, attention_mask, labels, device, word_index=None, extra=None):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.extra = {k: torch.as_tensor(v, device=device) for k, v in (extra or {}).items()}
        self.lengths = self.attention_mask.sum(1)
        self.word_index = word_index  # ATE: per sentence, list of first-subword positions per word (-1 = truncated)

    def __len__(self):
        return self.input_ids.shape[0]

    def batches(self, bs, shuffle=False, generator=None):
        n = len(self)
        if shuffle:
            order = torch.randperm(n, generator=generator, device="cpu").to(self.input_ids.device)
        else:  # length-sorted for minimal padding at inference
            order = torch.argsort(self.lengths, descending=True)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            L = int(self.lengths[idx].max())
            lab = self.labels[idx]
            extra = {k: v[idx, :L].contiguous() for k, v in self.extra.items()}
            yield (idx, self.input_ids[idx, :L].contiguous(), self.attention_mask[idx, :L].contiguous(),
                   lab[:, :L].contiguous() if lab.dim() == 2 else lab, extra)


def _pad(seqs, value, L=None):
    L = L or max(len(s) for s in seqs)
    out = np.full((len(seqs), L), value, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def encode_asc(tok, df: pd.DataFrame, max_len: int, device, markers: bool = False, pool: bool = False) -> GPUData:
    labels = df["label"].values if "label" in df else np.zeros(len(df), dtype=np.int64)
    if not markers and not pool:
        enc = tok(list(df.text), list(df.aspect), truncation="only_first", max_length=max_len)
        return GPUData(_pad(enc["input_ids"], tok.pad_token_id), _pad(enc["attention_mask"], 0), labels, device)
    texts, spans = [], []
    for t, a, b in zip(df.text, df["from"], df["to"]):
        if markers:
            texts.append(t[:a] + MARK_L + t[a:b] + MARK_R + t[b:])
            spans.append((a + len(MARK_L), b + len(MARK_L)))
        else:
            texts.append(t)
            spans.append((a, b))
    enc = tok(texts, list(df.aspect), truncation="only_first", max_length=max_len, return_offsets_mapping=True)
    masks = []
    for i, (a, b) in enumerate(spans):
        seq = enc.sequence_ids(i)
        m = [1 if seq[j] == 0 and s < b and e > a and e > s else 0 for j, (s, e) in enumerate(enc["offset_mapping"][i])]
        if not any(m):  # aspect truncated away -> fall back to the aspect copy in the 2nd segment
            m = [1 if seq[j] == 1 else 0 for j in range(len(seq))]
        masks.append(m)
    return GPUData(_pad(enc["input_ids"], tok.pad_token_id), _pad(enc["attention_mask"], 0), labels, device,
                   extra={"aspect_mask": _pad(masks, 0)} if pool else None)


def encode_ate(tok, sents: list[dict], max_len: int, device) -> GPUData:
    enc = tok([s["tokens"] for s in sents], is_split_into_words=True, truncation=True, max_length=max_len)
    all_labels, word_index = [], []
    for i, s in enumerate(sents):
        wids = enc.word_ids(i)
        lab = [-100] * len(wids)
        first = [-1] * len(s["tokens"])
        prev = None
        for pos, w in enumerate(wids):
            if w is not None and w != prev:
                lab[pos] = s["tag_ids"][w]
                first[w] = pos
            prev = w
        all_labels.append(lab)
        word_index.append(first)
    return GPUData(_pad(enc["input_ids"], tok.pad_token_id), _pad(enc["attention_mask"], 0),
                   _pad(all_labels, -100), device, word_index=word_index)


# --------------------------------------------------------------------------- decoding
_ALLOWED = np.array([[0, 0, -np.inf],   # from O: O, B ok; I forbidden
                     [0, 0, 0],         # from B
                     [0, 0, 0]])        # from I
_START = np.array([0, 0, -np.inf])


def viterbi_bio(logp: np.ndarray) -> list[int]:
    """Most likely tag sequence (O,B,I) under the constraint that I never follows O / starts a sentence."""
    n = len(logp)
    if n == 0:
        return []
    score = _START + logp[0]
    back = np.zeros((n, 3), dtype=np.int64)
    for t in range(1, n):
        cand = score[:, None] + _ALLOWED          # prev x cur
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = [int(score.argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


# --------------------------------------------------------------------------- evaluation
def _forward(model, ids, mask, extra):
    return model(input_ids=ids, attention_mask=mask, **extra)


@torch.no_grad()
def run_inference(model, data: GPUData, task: str, bs: int, viterbi: bool = False):
    """Returns (mean loss, predictions). ASC preds: (N,) array + probs. ATE preds: list of word-tag lists.
    Loss is plain (unsmoothed) cross-entropy so curves stay comparable across recipes."""
    model.eval()
    tot, cnt = torch.zeros((), device=data.input_ids.device), 0
    n = len(data)
    if task == "asc":
        preds = torch.empty(n, dtype=torch.long, device=data.input_ids.device)
        probs = torch.empty(n, len(POLARITIES), device=data.input_ids.device)
    else:
        tok_out = [None] * n
    for idx, ids, mask, lab, extra in data.batches(bs):
        with torch.autocast("cuda", dtype=amp_dtype()):
            logits = _forward(model, ids, mask, extra).logits
        logits = logits.float()
        if task == "asc":
            tot += F.cross_entropy(logits, lab, reduction="sum")
            cnt += len(idx)
            preds[idx] = logits.argmax(-1)
            probs[idx] = logits.softmax(-1)
        else:
            valid = lab != -100
            tot += F.cross_entropy(logits[valid], lab[valid], reduction="sum")
            cnt += int(valid.sum())
            arr = (logits.log_softmax(-1) if viterbi else logits.argmax(-1)).cpu().numpy()
            for r, i in enumerate(idx.tolist()):
                tok_out[i] = arr[r]
    loss = (tot / max(cnt, 1)).item()
    model.train()
    if task == "asc":
        return loss, (preds.cpu().numpy(), probs.cpu().numpy())
    word_tags = []
    for i, first in enumerate(data.word_index):
        if viterbi:
            lp = np.stack([tok_out[i][p] if p >= 0 else np.array([0.0, -1e4, -1e4]) for p in first]) \
                if first else np.zeros((0, 3))
            word_tags.append(viterbi_bio(lp))
        else:
            word_tags.append([int(tok_out[i][p]) if p >= 0 else 0 for p in first])
    return loss, word_tags


def score(task, data: GPUData, preds, gold=None) -> dict:
    if task == "asc":
        return classification_metrics(data.labels.cpu().numpy(), preds[0])
    pred_spans = [bio_to_spans(t) for t in preds]
    return span_prf(gold, pred_spans)


# --------------------------------------------------------------------------- adversarial training
class FGM:
    """Fast Gradient Method (Miyato et al. 2017): perturb word embeddings along the gradient, add adversarial loss."""

    def __init__(self, model, eps):
        self.eps = eps
        self.emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") \
            else model.encoder.get_input_embeddings()
        self.backup = None

    def attack(self):
        g = self.emb.weight.grad
        if g is None:
            return False
        norm = torch.linalg.vector_norm(g.float())
        if not torch.isfinite(norm) or norm == 0:
            return False
        self.backup = self.emb.weight.data.clone()
        self.emb.weight.data.add_(self.eps * g / norm)
        return True

    def restore(self):
        if self.backup is not None:
            self.emb.weight.data.copy_(self.backup)
            self.backup = None


# --------------------------------------------------------------------------- training
class Trainer:
    def __init__(self, cfg: TrainConfig, out_dir: Path, log=print):
        self.cfg, self.out, self.log = cfg, Path(out_dir), log
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = get_device()
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        kw = {"add_prefix_space": True} if "roberta" in cfg.model_name else {}
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, **kw)
        if cfg.task == "asc" and cfg.aspect_pool:
            self.model = AspectPooledClassifier(cfg.model_name, len(POLARITIES))
        elif cfg.task == "asc":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model_name, num_labels=len(POLARITIES),
                id2label=dict(enumerate(POLARITIES)), label2id={p: i for i, p in enumerate(POLARITIES)})
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                cfg.model_name, num_labels=len(BIO_TAGS),
                id2label=dict(enumerate(BIO_TAGS)), label2id={t: i for i, t in enumerate(BIO_TAGS)})
        if cfg.grad_ckpt:
            enc = self.model.encoder if isinstance(self.model, AspectPooledClassifier) else self.model
            enc.gradient_checkpointing_enable()
        self.model.to(self.device)
        self.main_metric = "macro_f1" if cfg.task == "asc" else "f1"

    def encode(self, x):
        if self.cfg.task == "asc":
            return encode_asc(self.tok, x, self.cfg.max_len, self.device, self.cfg.aspect_markers, self.cfg.aspect_pool)
        return encode_ate(self.tok, x, self.cfg.max_len, self.device)

    def _gold(self, sents):
        return [s["gold_spans"] for s in sents] if self.cfg.task == "ate" else None

    def _loss(self, logits, labels):
        logits = logits.float()
        if logits.dim() == 3:
            logits, labels = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        return F.cross_entropy(logits, labels, ignore_index=-100, label_smoothing=self.cfg.label_smoothing)

    def _param_groups(self):
        """Weight-decay split + optional layer-wise LR decay (embeddings lowest, head highest)."""
        cfg = self.cfg
        n_layers = getattr(self.model.config if hasattr(self.model, "config") else self.model.encoder.config,
                           "num_hidden_layers")
        no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        groups = {}
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            m = re.search(r"encoder\.layer\.(\d+)\.", n)
            if m:
                depth = int(m.group(1)) + 1
                lr = cfg.lr * cfg.llrd ** (n_layers - depth + 1)
            elif "embeddings" in n:
                lr = cfg.lr * cfg.llrd ** (n_layers + 1)
            else:  # classifier / pooler / head
                lr = cfg.head_lr or cfg.lr
            wd = 0.0 if any(k in n for k in no_decay) else cfg.weight_decay
            groups.setdefault((lr, wd), []).append(p)
        return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]

    def fit(self, train, val, title: str) -> dict:
        """train/val: DataFrame (asc) or sentence list (ate)."""
        cfg = self.cfg
        t0 = time.time()
        dtr, dva = self.encode(train), self.encode(val)
        gtr, gva = self._gold(train), self._gold(val)
        # train-metric curve on a fixed subset (pure monitoring; speeds up large models)
        if cfg.train_eval_max and len(train) > cfg.train_eval_max:
            sub = np.random.RandomState(cfg.seed).choice(len(train), cfg.train_eval_max, replace=False)
            tr_sub = train.iloc[sub] if isinstance(train, pd.DataFrame) else [train[i] for i in sub]
            dtr_eval, gtr_eval = self.encode(tr_sub), self._gold(tr_sub)
        else:
            dtr_eval, gtr_eval = dtr, gtr
        self.log(f"  encoded train={len(dtr)} val={len(dva)} in {time.time() - t0:.1f}s "
                 f"(max len {int(dtr.lengths.max())}) | device={torch.cuda.get_device_name(0)} amp={amp_dtype()}")

        opt = torch.optim.AdamW(self._param_groups(), lr=cfg.lr, fused=True)
        spe = math.ceil(len(dtr) / cfg.batch_size)
        total = spe * cfg.epochs
        sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
        eval_points = {max(1, round(spe * (k + 1) / cfg.evals_per_epoch)) for k in range(cfg.evals_per_epoch)}
        gen = torch.Generator().manual_seed(cfg.seed)
        fgm = FGM(self.model, cfg.fgm_eps) if cfg.fgm_eps > 0 else None

        history, step_losses, step_lrs = [], [], []
        pending = []  # GPU loss tensors not yet synced
        best, best_state, bad = -1.0, None, 0
        patience_evals = cfg.patience * cfg.evals_per_epoch
        step, stop = 0, False
        self.model.train()
        for epoch in range(cfg.epochs):
            bar = tqdm(dtr.batches(cfg.batch_size, shuffle=True, generator=gen), total=spe,
                       desc=f"{title} | epoch {epoch + 1}/{cfg.epochs}", leave=False, dynamic_ncols=True)
            for k, (_, ids, mask, lab, extra) in enumerate(bar, start=1):
                with torch.autocast("cuda", dtype=amp_dtype()):
                    loss = self._loss(_forward(self.model, ids, mask, extra).logits, lab)
                loss.backward()
                if fgm is not None and fgm.attack():
                    with torch.autocast("cuda", dtype=amp_dtype()):
                        adv = self._loss(_forward(self.model, ids, mask, extra).logits, lab)
                    adv.backward()
                    fgm.restore()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                ep_float = epoch + k / spe
                pending.append((ep_float, loss.detach()))
                step_lrs.append((ep_float, sched.get_last_lr()[0]))
                if k % 20 == 0:
                    bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

                if k in eval_points:
                    vals = torch.stack([l for _, l in pending]).float().cpu().tolist()
                    step_losses += [(e, v) for (e, _), v in zip(pending, vals)]
                    running = float(np.mean(vals))
                    pending = []
                    tr_loss, tr_pred = run_inference(self.model, dtr_eval, cfg.task, cfg.eval_batch_size, cfg.viterbi)
                    va_loss, va_pred = run_inference(self.model, dva, cfg.task, cfg.eval_batch_size, cfg.viterbi)
                    mtr, mva = score(cfg.task, dtr_eval, tr_pred, gtr_eval), score(cfg.task, dva, va_pred, gva)
                    row = {"epoch": round(ep_float, 3), "step": step, "lr": sched.get_last_lr()[0],
                           "train_running_loss": running, "train_eval_loss": tr_loss, "val_loss": va_loss,
                           "elapsed_s": round(time.time() - t0, 1)}
                    keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall"] if cfg.task == "asc" \
                        else ["f1", "precision", "recall"]
                    for kk in keys:
                        row[f"train_{kk}"], row[f"val_{kk}"] = mtr[kk], mva[kk]
                    history.append(row)
                    improved = mva[self.main_metric] > best
                    if improved:
                        best, bad = mva[self.main_metric], 0
                        # kept in CPU RAM so large models do not need 2x GPU memory
                        best_state = {n: p.detach().to("cpu", copy=True) for n, p in self.model.state_dict().items()}
                        best_row = row
                    else:
                        bad += 1
                    self.log(f"  [ep {ep_float:5.2f}] loss tr={running:.4f}/{tr_loss:.4f} val={va_loss:.4f} | "
                             f"{self.main_metric} tr={mtr[self.main_metric]:.4f} val={mva[self.main_metric]:.4f}"
                             f"{'  *best*' if improved else ''}")
                    pd.DataFrame(history).to_csv(self.out / "history.csv", index=False)  # live progress file
                    if bad >= patience_evals:
                        self.log(f"  early stopping (no val improvement in {patience_evals} evals)")
                        stop = True
                        break
            bar.close()
            if stop:
                break

        self.model.load_state_dict(best_state)
        del best_state
        train_time = time.time() - t0
        plot_training_curves(history, step_losses, cfg.task, f"{title}: training progress",
                             self.out / "training_curves.png", best_epoch=best_row["epoch"])
        plot_lr(step_lrs, f"{title}: LR schedule (linear warm-up/decay)", self.out / "lr_schedule.png")
        with open(self.out / "history.json", "w") as f:
            json.dump({"config": asdict(cfg), "history": history, "best": best_row,
                       "train_time_s": train_time, "step_losses": step_losses}, f, indent=1)
        self.log(f"  done in {train_time / 60:.1f} min - best val {self.main_metric}={best:.4f} @ epoch {best_row['epoch']}")
        return {"history": history, "best": best_row, "train_time_s": train_time}

    # ------------------------------------------------------------------ test-time
    def test(self, data_x, ds_name: str, tag: str) -> tuple[dict, object]:
        d = self.encode(data_x)
        loss, pred = run_inference(self.model, d, self.cfg.task, self.cfg.eval_batch_size, self.cfg.viterbi)
        m = score(self.cfg.task, d, pred, self._gold(data_x))
        m["loss"] = loss
        if self.cfg.task == "asc":
            plot_confusion_matrix(m["confusion_matrix"], POLARITIES,
                                  f"{tag}\ntest: {DATASET_NAMES[ds_name]}", self.out / f"confusion_test_{ds_name}.png")
        return m, pred

    def predict_asc(self, df: pd.DataFrame):
        if len(df) == 0:
            return np.array([], dtype=int)
        d = encode_asc(self.tok, df, self.cfg.max_len, self.device, self.cfg.aspect_markers, self.cfg.aspect_pool)
        return run_inference(self.model, d, "asc", self.cfg.eval_batch_size)[1][0]

    def save(self):
        p = self.out / "model"
        p.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, AspectPooledClassifier):
            torch.save(self.model.state_dict(), p / "model.pt")
        else:
            self.model.save_pretrained(p)
        self.tok.save_pretrained(p)

    def close(self):
        del self.model
        torch.cuda.empty_cache()
