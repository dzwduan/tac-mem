#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLINC150 (All-Generalization-OOD-CLINC150) semantic split generator:
- Converts intents into a TF-OWSL-style "emergence schedule" (Y0 + phases).
- Prefers dataset-provided domain field if available; otherwise uses mapping file.

Outputs:
- split_config.json: contains Y0 intents, phases, oos name, and schedule params
- split_stats.json: counts of intents and samples per split / per phase
- split_doc.md: a human-readable doc with concrete numbers (filled after running)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datasets import load_dataset


OOS_INTENT_NAME_DEFAULT = "ood"


@dataclass(frozen=True)
class SplitSchedule:
    y0_domains: List[str]
    phase_domains: List[str]  # one domain per phase (recommended)


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_intent2domain(path: str) -> Dict[str, str]:
    """
    Mapping file format (JSON):
    {
      "intent_name_1": "domainA",
      "intent_name_2": "domainA",
      ...
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("intent2domain mapping must be a JSON object: {intent(str): domain(str)}")
    return mapping


def _detect_domain_field(split) -> str | None:
    """
    Try common field names. Return the name if exists, else None.
    """
    candidates = ["domain", "domains", "category", "group", "source_domain"]
    cols = set(split.column_names)
    for c in candidates:
        if c in cols:
            return c
    return None


def extract_intents_data(
    clinc150_split,
    oos_intent_name: str = OOS_INTENT_NAME_DEFAULT,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Extract intent names (excluding OOS) and assign 0..149 ids.
    Returns:
      intent_names: list[str] length 150
      name_to_id: dict[str,int]
    """
    intent_names = sorted(clinc150_split.unique("labels"))
    if oos_intent_name not in intent_names:
        raise ValueError(f"OOS intent name '{oos_intent_name}' not in labels. Found labels: {intent_names[:10]} ...")
    intent_names.remove(oos_intent_name)

    n_classes = len(intent_names)
    if n_classes != 150:
        raise ValueError(f"Expected 150 in-scope intents, got {n_classes}. "
                         f"Your dataset variant may differ; adjust logic accordingly.")
    name_to_id = {name: i for i, name in enumerate(intent_names)}
    return intent_names, name_to_id


def build_intent2domain(
    split,
    intent_names: List[str],
    oos_intent_name: str,
    mapping_path: str | None,
) -> Dict[str, str]:
    """
    Build intent->domain mapping.
    Priority:
      1) dataset domain field if exists
      2) mapping JSON file
    """
    domain_field = _detect_domain_field(split)
    if domain_field is not None:
        # Build mapping from dataset columns
        intent2domain: Dict[str, str] = {}
        # We iterate once; dataset is small enough
        for ex in split:
            label = ex["labels"]
            if label == oos_intent_name:
                continue
            if label not in intent2domain:
                intent2domain[label] = ex[domain_field]
        # Validate
        missing = [x for x in intent_names if x not in intent2domain]
        if missing:
            raise ValueError(f"Domain field '{domain_field}' exists but mapping missing {len(missing)} intents, "
                             f"e.g. {missing[:10]}")
        return intent2domain

    if mapping_path is None:
        raise ValueError(
            "No domain field found in dataset. Provide --intent2domain JSON mapping file.\n"
            "Tip: create a JSON file mapping each intent name to a domain name."
        )

    intent2domain = _load_intent2domain(mapping_path)
    missing = [x for x in intent_names if x not in intent2domain]
    extra = [x for x in intent2domain.keys() if x not in set(intent_names)]
    if missing:
        raise ValueError(f"intent2domain missing {len(missing)} intents, e.g. {missing[:10]}")
    if extra:
        print(f"[WARN] intent2domain has {len(extra)} extra keys not in dataset intents, e.g. {extra[:10]}")
    return {k: intent2domain[k] for k in intent_names}


def propose_schedule(domains: List[str], y0_domain_count: int = 4) -> SplitSchedule:
    """
    Deterministic schedule:
      - Y0 takes the first y0_domain_count domains in sorted order
      - Each remaining domain becomes one phase
    Why deterministic? reproducibility & easy debugging.
    """
    domains_sorted = sorted(domains)
    if len(domains_sorted) <= y0_domain_count:
        raise ValueError(f"Need > {y0_domain_count} domains to have phases. Got {len(domains_sorted)}.")
    y0 = domains_sorted[:y0_domain_count]
    phases = domains_sorted[y0_domain_count:]
    return SplitSchedule(y0_domains=y0, phase_domains=phases)


def assign_intents_to_phases(intent2domain: Dict[str, str], schedule: SplitSchedule) -> Dict[str, Any]:
    """
    Return:
      {
        "Y0": [intent...],
        "phases": [{"name":"Phase-1","domain":"X","intents":[...]}, ...]
      }
    """
    domain2intents: Dict[str, List[str]] = defaultdict(list)
    for intent, dom in intent2domain.items():
        domain2intents[dom].append(intent)
    for dom in domain2intents:
        domain2intents[dom] = sorted(domain2intents[dom])

    y0_intents: List[str] = []
    for dom in schedule.y0_domains:
        y0_intents.extend(domain2intents[dom])

    phases = []
    for i, dom in enumerate(schedule.phase_domains, start=1):
        phases.append({
            "name": f"Phase-{i}",
            "domain": dom,
            "intents": domain2intents[dom],
        })

    return {"Y0": sorted(y0_intents), "phases": phases}


def compute_sample_stats(ds, intent2domain: Dict[str, str], oos_intent_name: str) -> Dict[str, Any]:
    """
    Compute:
      - total in-scope samples / oos samples per HF split
      - in-scope per domain per HF split
    """
    stats: Dict[str, Any] = {}
    for split_name in ["train", "validation", "test"]:
        split = ds[split_name]
        total = len(split)
        oos = 0
        in_scope = 0
        dom_counts = defaultdict(int)

        for ex in split:
            lab = ex["labels"]
            if lab == oos_intent_name:
                oos += 1
            else:
                in_scope += 1
                dom = intent2domain.get(lab, "UNKNOWN_DOMAIN")
                dom_counts[dom] += 1

        stats[split_name] = {
            "total_samples": total,
            "in_scope_samples": in_scope,
            "oos_samples": oos,
            "in_scope_by_domain": dict(sorted(dom_counts.items(), key=lambda x: x[0])),
        }
    return stats


def render_doc(
    out_path: str,
    dataset_name: str,
    oos_name: str,
    schedule: SplitSchedule,
    phase_assignments: Dict[str, Any],
    stats: Dict[str, Any],
) -> None:
    """
    Write a Markdown doc that includes concrete numbers from stats.
    """
    y0_intents = phase_assignments["Y0"]
    phases = phase_assignments["phases"]

    domain_intent_sizes = {p["domain"]: len(p["intents"]) for p in phases}
    y0_domain_sizes = {d: len([i for i in y0_intents if True]) for d in schedule.y0_domains}  # not perfect
    # We'll compute per-domain sizes properly:
    per_domain_sizes = defaultdict(int)
    for p in phases:
        per_domain_sizes[p["domain"]] += len(p["intents"])
    # Y0 domains:
    for d in schedule.y0_domains:
        # count intents in Y0 that belong to domain d by checking prefix list:
        pass  # doc will use stats domains anyway; intent2domain isn't passed here.

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# CLINC150 TF-OWSL 语义划分说明（自动生成）\n\n")
        f.write(f"- Dataset: `{dataset_name}`\n")
        f.write(f"- OOS label name: `{oos_name}`（始终作为 unknown，不参与扩类）\n\n")

        f.write("## 1. 划分目标\n")
        f.write("我们将 150 个 in-scope intents 划分为 `Y0`（上线即支持）+ 若干 `Phase-k`（随时间逐步出现的新语义）。\n")
        f.write("划分单位选择 **domain**，理由：\n")
        f.write("- domain 对应真实系统中的“功能模块”；新模块上线→相关查询会持续出现，更符合时间一致性假设。\n")
        f.write("- 避免随机拆分 intent 造成语义不可解释、触发/延迟统计不稳定。\n\n")

        f.write("## 2. 划分结果（domain 级）\n")
        f.write(f"- `Y0` domains（{len(schedule.y0_domains)} 个）：{schedule.y0_domains}\n")
        f.write(f"- phases（{len(schedule.phase_domains)} 个）：每个 phase 引入 1 个 domain\n\n")

        f.write("### 2.1 Y0\n")
        f.write(f"- intents 数：{len(y0_intents)}\n\n")

        f.write("### 2.2 Phases\n")
        for p in phases:
            f.write(f"- {p['name']}: domain=`{p['domain']}`, intents={len(p['intents'])}\n")
        f.write("\n")

        f.write("## 3. 数据统计（运行脚本得到的真实计数）\n")
        for split_name, s in stats.items():
            f.write(f"### {split_name}\n")
            f.write(f"- total={s['total_samples']}\n")
            f.write(f"- in_scope={s['in_scope_samples']}\n")
            f.write(f"- oos={s['oos_samples']}\n")
            f.write(f"- in_scope_by_domain（节选前 10 项）：\n")
            items = list(s["in_scope_by_domain"].items())[:10]
            for dom, cnt in items:
                f.write(f"  - {dom}: {cnt}\n")
            f.write("\n")

        f.write("## 4. 为什么这样划分（可复现实验的关键）\n")
        f.write("1) **事件强度足够**：每次引入一个 domain（通常约 15 intents），避免“新语义太弱导致触发不稳定”。\n")
        f.write("2) **OOS 始终存在**：OOS 从头到尾混入，且永不转正为新 intent，用来评估误扩类风险。\n")
        f.write("3) **可解释可复现**：domain 列表排序后确定 Y0 与 phase 顺序，保证不同机器/不同时间跑出的划分一致。\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cmaldona/All-Generalization-OOD-CLINC150")
    ap.add_argument("--oos_name", type=str, default=OOS_INTENT_NAME_DEFAULT)
    ap.add_argument("--out_dir", type=str, default="clinc150_split_out")
    ap.add_argument("--y0_domains", type=int, default=4, help="How many domains to include in Y0 (default 4)")
    ap.add_argument("--intent2domain", type=str, default=None,
                    help="Optional JSON mapping {intent: domain}, used if dataset has no domain field.")
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)
    ds = load_dataset(args.dataset)

    intent_names, _ = extract_intents_data(ds["train"], oos_intent_name=args.oos_name)
    intent2domain = build_intent2domain(ds["train"], intent_names, args.oos_name, args.intent2domain)

    domains = sorted(set(intent2domain.values()))
    schedule = propose_schedule(domains, y0_domain_count=args.y0_domains)
    phase_assignments = assign_intents_to_phases(intent2domain, schedule)
    stats = compute_sample_stats(ds, intent2domain, args.oos_name)

    # Save artifacts
    config = {
        "dataset": args.dataset,
        "oos_name": args.oos_name,
        "schedule": {
            "y0_domains": schedule.y0_domains,
            "phase_domains": schedule.phase_domains,
        },
        "assignments": phase_assignments,
        "notes": {
            "unit": "domain",
            "rationale": [
                "Domain corresponds to functional module; emergence is interpretable.",
                "One-domain-per-phase gives sufficient signal strength per emergence event.",
                "OOS remains unknown forever; used to measure false expansion risk.",
            ],
        },
    }

    with open(os.path.join(args.out_dir, "split_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    render_doc(
        out_path=os.path.join(args.out_dir, "split_doc.md"),
        dataset_name=args.dataset,
        oos_name=args.oos_name,
        schedule=schedule,
        phase_assignments=phase_assignments,
        stats=stats,
    )

    # Print quick summary
    print("=== Split Summary ===")
    print(f"Domains: {len(domains)}")
    print(f"Y0 domains ({len(schedule.y0_domains)}): {schedule.y0_domains}")
    print(f"Phases ({len(schedule.phase_domains)}): {schedule.phase_domains[:6]}{'...' if len(schedule.phase_domains)>6 else ''}")
    print(f"Y0 intents: {len(phase_assignments['Y0'])}")
    print(f"Total intents in phases: {sum(len(p['intents']) for p in phase_assignments['phases'])}")
    print("\nArtifacts written to:", args.out_dir)


if __name__ == "__main__":
    main()
