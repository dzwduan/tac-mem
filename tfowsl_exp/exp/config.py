from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix in [".yaml", ".yml"]:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError("Config must be .yaml/.yml or .json")
