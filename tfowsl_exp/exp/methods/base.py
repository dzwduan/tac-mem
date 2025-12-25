from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch


@dataclass
class StepOut:
    logits: torch.Tensor
    unknown: torch.Tensor
    pred: torch.Tensor


class MethodBase:
    def __init__(self, name: str, backbone, num_classes: int, device: str):
        self.name = name
        self.backbone = backbone
        self.device = device
        self.num_classes = num_classes

        self.created_classes = 0
        self.events: List[Dict[str, Any]] = []  # record expansion/assignments etc.

    def step(self, t: int, texts: List[str], labels: Optional[torch.Tensor], meta: List[Dict[str, Any]]) -> StepOut:
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        return {"created_classes": self.created_classes}
