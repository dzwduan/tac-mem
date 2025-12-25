from __future__ import annotations
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from .stream import StreamSample


class DelayedSparseLabelOracle:
    """
    Per-intent sparse labeling + fixed delay + OOS never labeled.
    """

    def __init__(
        self,
        samples: Iterable[StreamSample],
        label_ratio: float,
        delay_steps: int,
        oos_name: str = "ood",
        seed: int = 7,
        min_per_intent: int = 1,
    ):
        if not (0.0 <= label_ratio <= 1.0):
            raise ValueError("label_ratio must be in [0,1].")
        if delay_steps < 0:
            raise ValueError("delay_steps must be >= 0.")
        self.oos_name = oos_name
        self.delay_steps = delay_steps
        self.seed = seed

        self._true: Dict[int, str] = {}
        self._t: Dict[int, int] = {}
        self._release: Dict[int, int] = {}

        by_intent: Dict[str, List[int]] = defaultdict(list)
        for s in samples:
            self._true[s.sid] = s.label_name
            self._t[s.sid] = s.t
            if s.label_name == oos_name or s.is_oos:
                continue
            by_intent[s.label_name].append(s.sid)

        rng = random.Random(seed)
        for intent, sids in by_intent.items():
            n = len(sids)
            k = int(n * label_ratio)
            if n > 0:
                k = max(k, min_per_intent) if min_per_intent > 0 else k
                k = min(k, n)
            chosen = rng.sample(sids, k) if k > 0 else []
            for sid in chosen:
                self._release[sid] = self._t[sid] + delay_steps

    def get_label(self, sid: int, current_t: int) -> Optional[str]:
        y = self._true.get(sid)
        if y is None:
            raise KeyError(sid)
        if y == self.oos_name:
            return None
        rt = self._release.get(sid)
        if rt is None:
            return None
        return y if current_t >= rt else None

    def get_release_time(self, sid: int) -> Optional[int]:
        return self._release.get(sid)
