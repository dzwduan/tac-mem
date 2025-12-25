from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional


def load_intent2domain(path: Optional[str]) -> Dict[str, str]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    m = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(m, dict):
        raise ValueError("intent2domain must be a json object {intent: domain}")
    out: Dict[str, str] = {}
    for k, v in m.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def intent_to_domain(intent: str, intent2domain: Dict[str, str]) -> str:
    return intent2domain.get(intent, "unknown_domain")
