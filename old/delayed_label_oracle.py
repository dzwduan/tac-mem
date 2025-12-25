# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class StreamSample:
    """One record in the stream."""
    sid: int
    t: int
    utterance: str
    label_name: str  # oracle true label (for eval only)
    is_oos: bool


def load_stream_jsonl(path: str) -> List[StreamSample]:
    """
    Load stream jsonl lines of:
      {"t": int, "utterance": str, "label_name": str, "is_oos": bool}
    Add deterministic sid in file order.
    """
    out: List[StreamSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            out.append(
                StreamSample(
                    sid=i,
                    t=int(r["t"]),
                    utterance=str(r["utterance"]),
                    label_name=str(r["label_name"]),
                    is_oos=bool(r["is_oos"]),
                )
            )
    return out


class DelayedSparseLabelOracle:
    """
    Implements:
      - per-intent sparse labeling ratio pi
      - fixed delay D steps from sample arrival time t to label release time
      - OOS never labeled

    Online training should call:
      oracle.get_label(sid, current_t) -> label_name | None
    """

    def __init__(
        self,
        samples: Iterable[StreamSample],
        label_ratio: float = 0.2,
        delay: int = 50,
        oos_name: str = "ood",
        seed: int = 7,
        min_per_intent: int = 1,
    ) -> None:
        if not (0.0 <= label_ratio <= 1.0):
            raise ValueError("label_ratio must be in [0,1].")
        if delay < 0:
            raise ValueError("delay must be >= 0.")
        if min_per_intent < 0:
            raise ValueError("min_per_intent must be >= 0.")

        self.label_ratio = label_ratio
        self.delay = delay
        self.oos_name = oos_name
        self.seed = seed
        self.min_per_intent = min_per_intent

        self._sid_to_true: Dict[int, str] = {}
        self._sid_to_t: Dict[int, int] = {}

        # sid -> release_time (only for chosen labelable in-scope samples)
        self._release_time: Dict[int, int] = {}

        # Build per-intent pools
        by_intent: Dict[str, List[int]] = {}
        for s in samples:
            self._sid_to_true[s.sid] = s.label_name
            self._sid_to_t[s.sid] = s.t
            if s.label_name == oos_name or s.is_oos:
                continue
            by_intent.setdefault(s.label_name, []).append(s.sid)

        rng = random.Random(seed)

        # Choose labelable subset per intent
        for intent, sids in by_intent.items():
            n = len(sids)
            k = int(n * label_ratio)
            if n > 0:
                k = max(k, min_per_intent) if min_per_intent > 0 else k
                k = min(k, n)
            else:
                k = 0

            chosen = rng.sample(sids, k) if k > 0 else []
            for sid in chosen:
                t_arrival = self._sid_to_t[sid]
                self._release_time[sid] = t_arrival + delay

    def get_label(self, sid: int, current_t: int) -> Optional[str]:
        """
        Returns true label_name if (sid is chosen labelable) and current_t >= release_time.
        Otherwise returns None. OOS always returns None.
        """
        true_label = self._sid_to_true.get(sid)
        if true_label is None:
            raise KeyError(f"Unknown sid={sid}")
        if true_label == self.oos_name:
            return None

        rt = self._release_time.get(sid)
        if rt is None:
            return None
        return true_label if current_t >= rt else None

    def get_release_time(self, sid: int) -> Optional[int]:
        """For debugging/analysis only (not used by training)."""
        return self._release_time.get(sid)

    def summary(self) -> Dict[str, int]:
        """
        Quick summary counts.
        """
        total = len(self._sid_to_true)
        oos = sum(1 for _, y in self._sid_to_true.items() if y == self.oos_name)
        labeled = len(self._release_time)
        return {
            "total_samples": total,
            "oos_samples": oos,
            "labelable_in_scope_samples": labeled,
        }


def attach_oracle_labels_for_time(
    samples_at_t: List[StreamSample],
    oracle: DelayedSparseLabelOracle,
    current_t: int,
) -> List[Tuple[StreamSample, Optional[str]]]:
    """
    Helper: returns list of (sample, available_label_or_None) at time current_t.
    This is what your online learner would receive.
    """
    return [(s, oracle.get_label(s.sid, current_t)) for s in samples_at_t]
