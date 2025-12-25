from __future__ import annotations
from typing import Dict, List, Any
from collections import defaultdict


def fer_oos_trigger(method_events: List[Dict[str, Any]], oos_name: str) -> Dict[str, Any]:
    creates = [e for e in method_events if e.get("type") == "create"]
    if not creates:
        return {"fer": 0.0, "n_create": 0}
    oos_creates = 0
    for e in creates:
        tr = e.get("trigger", {})
        if tr.get("is_oos", False) or tr.get("true_intent") == oos_name:
            oos_creates += 1
    return {"fer": oos_creates / len(creates), "n_create": len(creates), "n_oos_create": oos_creates}


def fer_mixture(method_events: List[Dict[str, Any]], oos_name: str, purity_threshold: float, oos_ratio_threshold: float, min_support: int) -> Dict[str, Any]:
    """
    We treat each created class id as a "cluster". We compute purity from assign events:
      assign event: {"type":"assign","pred_id":int,"meta":{"true_intent", "is_oos"}}
    A created class is "bad" if:
      - support < min_support => ignored (not enough evidence)
      - purity < purity_threshold OR oos_ratio > oos_ratio_threshold
    FER = bad_classes / considered_classes
    """
    assign = [e for e in method_events if e.get("type") == "assign"]
    by_id = defaultdict(list)
    for e in assign:
        pid = int(e.get("pred_id", -1))
        m = e.get("meta", {})
        by_id[pid].append(m)

    considered = 0
    bad = 0
    details = []

    for pid, metas in by_id.items():
        n = len(metas)
        if n < min_support:
            continue
        considered += 1
        oos = sum(1 for m in metas if m.get("is_oos", False) or m.get("true_intent") == oos_name)
        oos_ratio = oos / n
        cnt = defaultdict(int)
        for m in metas:
            y = m.get("true_intent", oos_name)
            cnt[str(y)] += 1
        top = max(cnt.values()) if cnt else 0
        purity = top / n if n > 0 else 0.0
        is_bad = (purity < purity_threshold) or (oos_ratio > oos_ratio_threshold)
        if is_bad:
            bad += 1
        details.append({"pred_id": pid, "support": n, "purity": purity, "oos_ratio": oos_ratio, "bad": is_bad})

    fer = bad / considered if considered > 0 else 0.0
    return {"fer": fer, "n_classes_considered": considered, "n_bad": bad, "details_head": details[:10]}
