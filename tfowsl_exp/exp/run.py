from __future__ import annotations
import argparse
import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch

from .config import load_config
from .logger import JSONLLogger
from .stream import load_stream_jsonl, make_toy_stream, iter_batches
from .oracle import DelayedSparseLabelOracle
from .backbone import build_backbone
from .mapping import load_intent2domain, intent_to_domain

from .metrics.ood import oos_auc, fpr_at_95tpr
from .metrics.delay import compute_first_appearance, compute_trigger_times, compute_delay_stats
from .metrics.fer import fer_oos_trigger, fer_mixture

from .methods.er import ER
from .methods.msp import MSP
from .methods.energy import Energy
from .methods.fixed_expand import FixedThresholdExpand
from .methods.cluster_expand import ClusterExpand
from .methods.tacmem import TACMem

from .plotting import export_summary_plots

def _resolve_from_cfg_dir(cfg_dir: Path, p: str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((cfg_dir / pp).resolve())

def build_methods(cfg: Dict[str, Any], backbone, num_classes: int, device: str):
    out = []
    names = cfg["methods"]["list"]
    mcfg = cfg["methods"]

    for n in names:
        if n == "ER":
            out.append(ER(backbone, num_classes, device, lr=mcfg["ER"]["lr"], buffer_size=mcfg["ER"]["buffer_size"]))
        elif n == "MSP":
            out.append(MSP(backbone, num_classes, device))
        elif n == "Energy":
            out.append(Energy(backbone, num_classes, device))
        elif n == "FixedThresholdExpand":
            c = mcfg["FixedThresholdExpand"]
            out.append(FixedThresholdExpand(
                backbone, num_classes, device, lr=c["lr"], thr=c["thr"], frac_trigger=c["frac_trigger"],
                new_class_per_trigger=c["new_class_per_trigger"], pseudo_topk=c["pseudo_topk"], pseudo_weight=c["pseudo_weight"]
            ))
        elif n == "ClusterExpand":
            c = mcfg["ClusterExpand"]
            out.append(ClusterExpand(
                backbone, num_classes, device, lr=c["lr"], thr=c["thr"], min_unknown=c["min_unknown"],
                k=c["k"], pseudo_weight=c["pseudo_weight"]
            ))
        elif n == "TACMem":
            c = mcfg["TACMem"]
            out.append(TACMem(
                backbone, num_classes, device, lr=c["lr"], thr=c["thr"], ema_beta=c["ema_beta"],
                min_cons_steps=c["min_cons_steps"], min_unknown=c["min_unknown"], risk_oos_max=c["risk_oos_max"],
                pseudo_topk=c["pseudo_topk"], pseudo_weight=c["pseudo_weight"]
            ))
        else:
            raise ValueError(f"Unknown method: {n}")
    return out

from pathlib import Path


def _resolve_from_cfg_dir(cfg_dir: Path, p: str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((cfg_dir / pp).resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent

    cfg = load_config(str(cfg_path))  # 用绝对路径读 config

    # ---- resolve paths relative to cfg_dir ----
    cfg["run"]["out_dir"] = _resolve_from_cfg_dir(cfg_dir, cfg["run"]["out_dir"])

    stream_cfg = cfg["data"]["stream"]
    if stream_cfg.get("type") == "jsonl":
        stream_cfg["path"] = _resolve_from_cfg_dir(cfg_dir, stream_cfg.get("path"))

    cfg["data"]["intent2domain_path"] = _resolve_from_cfg_dir(cfg_dir, cfg["data"].get("intent2domain_path"))


    seed = cfg["run"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = cfg["run"]["device"]
    out_dir = cfg["run"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    steps = int(cfg["run"]["steps"])
    B = int(cfg["run"]["batch_size"])
    eval_every = int(cfg["run"]["eval_every"])

    oos_name = cfg["data"]["oos_name"]
    intent2domain = load_intent2domain(cfg["data"].get("intent2domain_path"))

    # load stream
    stream_type = cfg["data"]["stream"]["type"]
    if stream_type == "toy":
        stream = make_toy_stream(T=steps, B=B)
    elif stream_type == "jsonl":
        stream = load_stream_jsonl(cfg["data"]["stream"]["path"])
    else:
        raise ValueError("stream.type must be toy|jsonl")

    # build oracle
    sup = cfg["supervision"]
    oracle = DelayedSparseLabelOracle(
        stream, label_ratio=float(sup["label_ratio"]), delay_steps=int(sup["delay_steps"]),
        oos_name=oos_name, seed=seed, min_per_intent=int(sup["min_per_intent"])
    )

    # build shared backbone
    bb = cfg["model"]["backbone"]
    backbone = build_backbone(bb["name"], int(bb["max_length"]), device=device)

    # Build label_name -> id for supervised update (only for in-scope labels)
    # For CLINC150, you can provide a full mapping; here we infer from stream.
    label_names = sorted({s.label_name for s in stream if (not s.is_oos and s.label_name != oos_name)})
    name_to_id = {n: i for i, n in enumerate(label_names)}
    num_classes = len(name_to_id)  # starting classes (toy: 3; clinc: depends on your stream content)

    methods = build_methods(cfg, backbone, num_classes=num_classes, device=device)

    # Pre-build "stream meta" list for delay metrics
    stream_meta = [{"t": s.t, "true_intent": s.label_name, "is_oos": s.is_oos} for s in stream]

    rows = []

    for m in methods:
        logger = JSONLLogger(os.path.join(out_dir, "logs", f"{m.name}.jsonl"))

        y_is_oos = []
        unk_score = []

        for t, batch in tqdm(list(iter_batches(stream, steps, B)), desc=m.name):
            texts = [s.utterance for s in batch]
            meta = [{"t": t, "true_intent": s.label_name, "is_oos": s.is_oos, "domain": intent_to_domain(s.label_name, intent2domain)} for s in batch]

            # build supervised labels tensor for those whose labels are released
            y_list = []
            idx = []
            for j, s in enumerate(batch):
                lab = oracle.get_label(s.sid, t)
                if lab is not None and (not s.is_oos) and lab != oos_name and lab in name_to_id:
                    y_list.append(name_to_id[lab])
                    idx.append(j)
            labels = None
            if len(idx) > 0:
                # supervise only the labeled subset (simple and safe)
                labels = torch.tensor(y_list, device=device, dtype=torch.long)
                # pass only labeled subset to the method via meta? simplest: method uses full batch,
                # ER/expand use labels only to compute CE on full logits or subset — to avoid ambiguity,
                # here we compute CE outside is more complex.
                # For this minimal framework, we pass labels=None and rely on pseudo training in expand methods,
                # and ER uses labels only when not None. We'll provide labels on full batch by marking unlabeled as -1? not supported.
                # => Here we do a conservative choice: provide labels only if all samples are labeled.
                if len(idx) != len(batch):
                    labels = None

            out = m.step(t=t, texts=texts, labels=labels, meta=meta)

            unk = out.unknown.detach().cpu().numpy()
            for s, u in zip(batch, unk):
                y_is_oos.append(1 if (s.is_oos or s.label_name == oos_name) else 0)
                unk_score.append(float(u))

            if t % eval_every == 0:
                logger.log({
                    "t": t,
                    "mean_unknown": float(np.mean(unk[-min(len(unk), 32):])),
                    "created_classes": m.created_classes,
                    "n_events": len(m.events),
                })

        # --- compute metrics ---
        y_is_oos = np.array(y_is_oos, dtype=np.int32)
        unk_score = np.array(unk_score, dtype=np.float32)
        auc = oos_auc(y_is_oos, unk_score)
        fpr = fpr_at_95tpr(y_is_oos, unk_score)

        # Delay
        delay_mode = cfg["metrics"]["delay"]["mode"]
        first = compute_first_appearance(stream_meta, delay_mode, intent2domain, oos_name)
        trig = compute_trigger_times(m.events, delay_mode, intent2domain, oos_name)
        delay_stat = compute_delay_stats(first, trig)

        # FER
        fer_mode = cfg["metrics"]["fer"]["mode"]
        if fer_mode == "oos_trigger":
            fer_stat = fer_oos_trigger(m.events, oos_name)
        elif fer_mode == "mixture":
            mix = cfg["metrics"]["fer"]["mixture"]
            fer_stat = fer_mixture(
                m.events, oos_name,
                purity_threshold=float(mix["purity_threshold"]),
                oos_ratio_threshold=float(mix["oos_ratio_threshold"]),
                min_support=int(mix["min_support"]),
            )
        else:
            raise ValueError("fer.mode must be oos_trigger|mixture")

        row = {
            "method": m.name,
            "oos_auc": auc,
            "fpr@95tpr": fpr,
            "created_classes": m.created_classes,
            "delay_mode": delay_mode,
            "mean_delay": delay_stat["mean_delay"] if delay_stat["mean_delay"] is not None else -1,
            "n_entities": delay_stat["n_entities"],
            "n_triggered": delay_stat["n_triggered"],
            "fer_mode": fer_mode,
            "fer": fer_stat.get("fer", 0.0),
            "n_create": fer_stat.get("n_create", fer_stat.get("n_classes_considered", 0)),
        }
        rows.append(row)

        # dump method event trace
        pd.DataFrame(m.events).to_json(os.path.join(out_dir, f"events_{m.name}.json"), orient="records", force_ascii=False, indent=2)

        logger.close()

    summary_csv = os.path.join(out_dir, "summary.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    export_summary_plots(summary_csv, os.path.join(out_dir, "plots"))
    print("Done. Summary:", summary_csv)


if __name__ == "__main__":
    main()
