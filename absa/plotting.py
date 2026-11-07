"""All figures (matplotlib, headless). Palette: fixed categorical order, single-hue sequential ramp."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# categorical slots, fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRAIN_C, VAL_C = C[0], C[1]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MODEL_COLORS = {"TF-IDF+LR": C[2], "TF-IDF+SVM": C[3], "BERT": C[0], "RoBERTa": C[1],
                "BERT-base v1": C[0], "RoBERTa-base v1": C[1], "RoBERTa-base v2": C[4], "RoBERTa-large v2": C[5]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "font.family": "DejaVu Sans", "figure.dpi": 110,
})


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _smooth(y, k):
    if len(y) < k or k <= 1:
        return np.asarray(y)
    c = np.cumsum(np.insert(np.asarray(y, float), 0, 0))
    s = (c[k:] - c[:-k]) / k
    return np.concatenate([np.asarray(y[:k - 1], float).cumsum() / np.arange(1, k), s])


# --------------------------------------------------------------------------- training curves
def plot_training_curves(history: list[dict], step_losses: list[tuple[float, float]], task: str,
                         title: str, path: Path, best_epoch: float | None = None):
    ep = [h["epoch"] for h in history]
    if task == "asc":
        panels = [("Macro-F1", "macro_f1"), ("Accuracy", "accuracy")]
    else:
        panels = [("Span F1", "f1"), ("Precision / Recall", None)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    if step_losses:
        xs, ys = zip(*step_losses)
        ax.plot(xs, ys, color=TRAIN_C, alpha=0.18, linewidth=1)
        ax.plot(xs, _smooth(ys, max(1, len(ys) // 40)), color=TRAIN_C, linewidth=1.4, alpha=0.6,
                label="train loss (per step, smoothed)")
    ax.plot(ep, [h["train_eval_loss"] for h in history], "-o", color=TRAIN_C, ms=5, label="train loss (eval pass)")
    ax.plot(ep, [h["val_loss"] for h in history], "-o", color=VAL_C, ms=5, label="validation loss")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.legend(loc="best")

    ax = axes[1]
    lab, key = panels[0]
    ax.plot(ep, [h[f"train_{key}"] for h in history], "-o", color=TRAIN_C, ms=5, label="train")
    ax.plot(ep, [h[f"val_{key}"] for h in history], "-o", color=VAL_C, ms=5, label="validation")
    ax.set_title(lab)
    ax.set_xlabel("epoch")
    ax.legend(loc="best")

    ax = axes[2]
    lab, key = panels[1]
    if key:
        ax.plot(ep, [h[f"train_{key}"] for h in history], "-o", color=TRAIN_C, ms=5, label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in history], "-o", color=VAL_C, ms=5, label="validation")
    else:
        ax.plot(ep, [h["train_precision"] for h in history], "-o", color=TRAIN_C, ms=5, label="train precision")
        ax.plot(ep, [h["train_recall"] for h in history], "--s", color=TRAIN_C, ms=5, label="train recall")
        ax.plot(ep, [h["val_precision"] for h in history], "-o", color=VAL_C, ms=5, label="val precision")
        ax.plot(ep, [h["val_recall"] for h in history], "--s", color=VAL_C, ms=5, label="val recall")
    ax.set_title(lab)
    ax.set_xlabel("epoch")
    ax.legend(loc="best")

    if best_epoch is not None:
        for a in axes:
            a.axvline(best_epoch, color=MUTED, linestyle=":", linewidth=1.2)
        axes[1].annotate("best checkpoint", (best_epoch, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(-4, -4), textcoords="offset points", ha="right", va="top", color=INK2, fontsize=8)
    fig.suptitle(title, fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def plot_lr(step_lrs, title, path):
    fig, ax = plt.subplots(figsize=(6, 3))
    xs, ys = zip(*step_lrs)
    ax.plot(xs, ys, color=TRAIN_C)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    _save(fig, path)


# --------------------------------------------------------------------------- matrices
def plot_confusion_matrix(cm, labels, title, path):
    cm = np.asarray(cm)
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, path)


def plot_cross_heatmaps(table, value_col, models, title, path, label="Macro-F1"):
    """table: DataFrame with columns model, train, test, <value_col>. One 2x2 grid per model."""
    names = {"semeval": "SemEval", "mams": "MAMS"}
    order = ["semeval", "mams"]
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.5), squeeze=False)
    for ax, m in zip(axes[0], models):
        grid = np.full((2, 2), np.nan)
        sub = table[table.model == m]
        for _, r in sub.iterrows():
            grid[order.index(r["train"]), order.index(r["test"])] = r[value_col]
        ax.imshow(grid, cmap=SEQ, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=11,
                            color="white" if grid[i, j] > 0.55 else INK, fontweight="bold")
        ax.set_xticks([0, 1], [names[o] for o in order])
        ax.set_yticks([0, 1], [names[o] for o in order])
        ax.set_xlabel("tested on")
        ax.set_ylabel("trained on")
        ax.set_title(m, fontsize=10.5)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"{title} ({label})", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------- bars
def grouped_bars(categories, series: dict, title, ylabel, path, ylim=(0, 1), fmt="{:.3f}", colors=None,
                 figsize=None, rotate=0):
    """series: {name: [value per category]}. Values annotated above bars."""
    n = len(series)
    x = np.arange(len(categories))
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=figsize or (max(6, 1.6 * len(categories) + 2), 4))
    for k, (name, vals) in enumerate(series.items()):
        col = (colors or {}).get(name, C[k % len(C)])
        bars = ax.bar(x + (k - (n - 1) / 2) * w, vals, w * 0.92, color=col, label=name,
                      edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            if v is not None and not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center",
                        va="bottom", fontsize=7, color=INK2, rotation=90 if n > 3 else 0)
    ax.set_xticks(x, categories, rotation=rotate, ha="right" if rotate else "center")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save(fig, path)


def plot_validation_curve(xs, series: dict, title, path, xlabel="C (inverse regularisation)"):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, (name, ys) in enumerate(series.items()):
        ax.plot(xs, ys, "-o", ms=5, color=[TRAIN_C, VAL_C, C[2], C[3]][k], label=name)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("score")
    ax.set_title(title, loc="left")
    ax.legend()
    _save(fig, path)
