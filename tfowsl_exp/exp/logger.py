from __future__ import annotations
import json
import os
import time
from typing import Dict, Any


class JSONLLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.t0 = time.time()

    def log(self, rec: Dict[str, Any]):
        rec = dict(rec)
        rec["_wall"] = time.time() - self.t0
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
