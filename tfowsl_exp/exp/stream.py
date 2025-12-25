from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable
from pathlib import Path


@dataclass(frozen=True)
class StreamSample:
    sid: int
    t: int
    utterance: str
    label_name: str   # oracle true label (eval-only)
    is_oos: bool


def load_stream_jsonl(path: str) -> List[StreamSample]:
    out: List[StreamSample] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stream jsonl not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            out.append(StreamSample(
                sid=i,
                t=int(r["t"]),
                utterance=str(r["utterance"]),
                label_name=str(r["label_name"]),
                is_oos=bool(r["is_oos"]),
            ))
    return out


def make_toy_stream(T: int = 30, B: int = 16) -> List[StreamSample]:
    """
    Toy stream:
      - intents: intent_a, intent_b (domain=dom1), intent_c (domain=dom2)
      - OOS: ood
      - emergence: intent_c starts appearing after t>10
    """
    samples: List[StreamSample] = []
    sid = 0
    for t in range(1, T + 1):
        for j in range(B):
            if j % 5 == 0:
                lab = "ood"; is_oos = True; utt = f"oos_{t}_{j}"
            else:
                if t <= 10:
                    lab = "intent_a" if j % 2 == 0 else "intent_b"
                else:
                    # gradually introduce intent_c
                    lab = "intent_c" if (j % 3 == 0) else ("intent_a" if j % 2 == 0 else "intent_b")
                is_oos = False
                utt = f"{lab}_{t}_{j}"
            samples.append(StreamSample(sid=sid, t=t, utterance=utt, label_name=lab, is_oos=is_oos))
            sid += 1
    return samples


def iter_batches(stream: List[StreamSample], steps: int, batch_size: int):
    """
    Assumes stream is in order and contains exactly batch_size samples per t.
    """
    i = 0
    for t in range(1, steps + 1):
        yield t, stream[i:i + batch_size]
        i += batch_size
