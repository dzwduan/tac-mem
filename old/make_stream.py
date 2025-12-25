#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, random
from collections import defaultdict
from datasets import load_dataset

OOS = "ood"

def load_split_config(path: str):
    cfg = json.load(open(path, "r", encoding="utf-8"))
    y0 = cfg["assignments"]["Y0"]
    phases = cfg["assignments"]["phases"]  # list of {name, domain, intents}
    return y0, phases, cfg.get("oos_name", OOS), cfg.get("dataset")

def build_phase_starts(T: int, n_phases: int, warmup: int):
    """
    Deterministic: warmup steps for Y0 only, then equally spaced phase starts.
    Example: warmup=200, T=1200, n_phases=6 -> starts at 200, 366, 533...
    """
    if n_phases <= 0:
        return []
    span = max(1, T - warmup)
    step = span // n_phases
    return [warmup + i * step for i in range(n_phases)]

def open_intents_at_t(t: int, y0, phases, phase_starts):
    open_set = set(y0)
    for i, s in enumerate(phase_starts):
        if t >= s:
            open_set.update(phases[i]["intents"])
    return open_set

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cmaldona/All-Generalization-OOD-CLINC150")
    ap.add_argument("--split", type=str, default="train", choices=["train","validation","test"])
    ap.add_argument("--split_config", type=str, required=True)
    ap.add_argument("--out", type=str, default="stream_train.jsonl")
    ap.add_argument("--T", type=int, default=1200, help="number of time steps")
    ap.add_argument("--B", type=int, default=64, help="batch size per time step")
    ap.add_argument("--oos_ratio", type=float, default=0.15)
    ap.add_argument("--warmup", type=int, default=200, help="steps before first phase opens")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)

    y0, phases, oos_name, cfg_dataset = load_split_config(args.split_config)
    dataset_name = cfg_dataset or args.dataset

    ds = load_dataset(dataset_name)[args.split]

    # bucket samples by intent label
    buckets = defaultdict(list)
    oos_bucket = []
    for ex in ds:
        lab = ex["labels"]
        txt = ex["data"]
        if lab == oos_name:
            oos_bucket.append((txt, lab))
        else:
            buckets[lab].append((txt, lab))

    # Shuffle each bucket deterministically
    for lab in buckets:
        random.shuffle(buckets[lab])
    random.shuffle(oos_bucket)

    phase_starts = build_phase_starts(args.T, len(phases), args.warmup)

    # pointers into each bucket to avoid repetition until exhaustion
    ptr = defaultdict(int)
    oos_ptr = 0

    def draw_from_label(lab):
        nonlocal ptr
        arr = buckets[lab]
        i = ptr[lab]
        if i >= len(arr):
            # if exhausted, wrap around (or you can stop / re-shuffle)
            i = 0
            ptr[lab] = 0
        txt, lab2 = arr[i]
        ptr[lab] += 1
        return txt, lab2

    def draw_oos():
        nonlocal oos_ptr
        if not oos_bucket:
            raise RuntimeError("No OOS samples in this dataset split.")
        if oos_ptr >= len(oos_bucket):
            oos_ptr = 0
        txt, lab = oos_bucket[oos_ptr]
        oos_ptr += 1
        return txt, lab

    # Prepare label pools for open intents
    all_in_scope_labels = list(buckets.keys())

    with open(args.out, "w", encoding="utf-8") as f:
        for t in range(1, args.T + 1):
            open_set = open_intents_at_t(t, y0, phases, phase_starts)
            open_labels = [lab for lab in open_set if lab in buckets]

            # fallback: if your y0/phases contain labels not present in this dataset variant
            if not open_labels:
                open_labels = all_in_scope_labels

            n_oos = int(round(args.B * args.oos_ratio))
            n_in = args.B - n_oos

            # sample in-scope uniformly over open labels
            for j in range(n_in):
                lab = random.choice(open_labels)
                txt, lab2 = draw_from_label(lab)
                rec = {"t": t, "utterance": txt, "label_name": lab2, "is_oos": False}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            for j in range(n_oos):
                txt, lab = draw_oos()
                rec = {"t": t, "utterance": txt, "label_name": lab, "is_oos": True}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Wrote stream:", args.out)
    print("Phase starts:", phase_starts)
    print("Y0 intents:", len(y0), "Phases:", len(phases), "OOS ratio:", args.oos_ratio)

if __name__ == "__main__":
    main()
