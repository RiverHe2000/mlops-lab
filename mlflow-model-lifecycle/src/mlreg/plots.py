"""Diagnostic plots written as PNG artefacts (headless Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def roc_curve_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tpr = np.cumsum(ys) / max(int(ys.sum()), 1)
    fpr = np.cumsum(1 - ys) / max(int((1 - ys).sum()), 1)
    return np.r_[0.0, fpr], np.r_[0.0, tpr]


def plot_roc(y: np.ndarray, p: np.ndarray, auc: float, path: Path) -> Path:
    fpr, tpr = roc_curve_points(np.asarray(y), np.asarray(p))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_calibration(table: pd.DataFrame, path: Path) -> Path:
    t = table[table["n"] > 0]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.plot(t["mean_pred"], t["obs_rate"], marker="o")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability diagram")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_score_distribution(y: np.ndarray, p: np.ndarray, threshold: float, path: Path) -> Path:
    y = np.asarray(y)
    p = np.asarray(p)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, label="good (y=0)")
    ax.hist(p[y == 1], bins=bins, alpha=0.6, label="bad (y=1)")
    ax.axvline(threshold, color="black", linestyle=":", label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Applicants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
