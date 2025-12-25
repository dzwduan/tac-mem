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
