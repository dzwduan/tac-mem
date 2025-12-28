from __future__ import annotations
import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def export_summary_plots(summary_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)

    # Bar: OOS AUROC
    plt.figure()
    plt.bar(df["method"], df["oos_auc"])
    plt.ylabel("OOS AUROC")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "oos_auc_bar.png"))
    plt.close()

    # Bar: FPR@95TPR
    plt.figure()
    plt.bar(df["method"], df["fpr@95tpr"])
    plt.ylabel("FPR@95TPR")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fpr95_bar.png"))
    plt.close()

    # Bar: created classes
    plt.figure()
    plt.bar(df["method"], df["created_classes"])
    plt.ylabel("#Created classes")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "created_classes_bar.png"))
    plt.close()

    # Combined comparison grid for quick glance
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax1, ax2, ax3, ax4 = axes.flat

    methods = df["method"]

    ax1.bar(methods, df["oos_auc"], color="#4c72b0")
    ax1.set_title("OOS AUC")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=45)
    for x, y in zip(methods, df["oos_auc"]):
        ax1.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(methods, df["fpr@95tpr"], color="#dd8452")
    ax2.set_title("FPR @ 95 TPR (lower better)")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="x", rotation=45)
    for x, y in zip(methods, df["fpr@95tpr"]):
        ax2.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    ax3.bar(methods, df["created_classes"], color="#55a868")
    ax3.set_title("Created Classes")
    ax3.tick_params(axis="x", rotation=45)
    offset = max(df["created_classes"].max() * 0.02, 1)
    for x, y in zip(methods, df["created_classes"]):
        ax3.text(x, y + offset, f"{int(y)}", ha="center", va="bottom", fontsize=8)

    ax4.bar(methods, df["fer"], color="#c44e52")
    ax4.set_title("FER")
    ax4.set_ylim(0, max(0.01, df["fer"].max() * 1.2))
    ax4.tick_params(axis="x", rotation=45)
    fer_offset = 0.02 if df["fer"].max() > 0 else 0.01
    for x, y in zip(methods, df["fer"]):
        ax4.text(x, y + fer_offset, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_comparison.png"), dpi=200)
    plt.close(fig)
