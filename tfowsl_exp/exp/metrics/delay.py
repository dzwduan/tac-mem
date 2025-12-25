from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

from ..mapping import intent_to_domain


def compute_first_appearance(stream_meta: List[Dict[str, Any]], mode: str, intent2domain: Dict[str, str], oos_name: str) -> Dict[str, int]:
    """
    stream_meta: list of {"t","true_intent","is_oos",...}
    mode: "domain" or "intent"
    Returns: entity -> first_t
    """
    first: Dict[str, int] = {}
    for m in stream_meta:
        if m.get("true_intent") == oos_name or m.get("is_oos", False):
            continue
        t = int(m["t"])
        if mode == "intent":
            key = str(m["true_intent"])
        elif mode == "domain":
            key = intent_to_domain(str(m["true_intent"]), intent2domain)
        else:
            raise ValueError("delay mode must be domain|intent")
        if key not in first:
            first[key] = t
    return first


def compute_trigger_times(method_events: List[Dict[str, Any]], mode: str, intent2domain: Dict[str, str], oos_name: str) -> Dict[str, int]:
    """
    method_events: list of {"t","type":"create","trigger":meta,...}
    Returns: entity -> first create t triggered by that entity
    """
    trig: Dict[str, int] = {}
    for e in method_events:
        if e.get("type") != "create":
            continue
        t = int(e["t"])
        trigger = e.get("trigger", {})
        if trigger.get("true_intent") == oos_name or trigger.get("is_oos", False):
            continue
        if mode == "intent":
            key = str(trigger.get("true_intent"))
        else:
            key = intent_to_domain(str(trigger.get("true_intent")), intent2domain)
        if key not in trig:
            trig[key] = t
    return trig


def compute_delay_stats(first: Dict[str, int], trig: Dict[str, int]) -> Dict[str, Any]:
    """
    Returns summary:
      - covered_entities, missed_entities
      - mean_delay over covered
    """
    covered = []
    missed = []
    for k, t0 in first.items():
        if k in trig:
            covered.append(trig[k] - t0)
        else:
            missed.append(k)
    mean_delay = sum(covered) / len(covered) if covered else None
    return {
        "n_entities": len(first),
        "n_triggered": len(covered),
        "n_missed": len(missed),
        "mean_delay": mean_delay,
        "missed_entities": missed[:20],
    }
