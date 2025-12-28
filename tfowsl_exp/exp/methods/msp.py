from __future__ import annotations
import torch
from .base import MethodBase, StepOut


class MSP(MethodBase):
    def __init__(self, backbone, num_classes: int, device: str):
        super().__init__("MSP", backbone, num_classes, device)
        H = backbone.hidden_size
        self.head = torch.nn.Linear(H, num_classes).to(device)

    def step(self, t, texts, labels, meta):
        h = self.backbone(texts)
        logits = self.head(h)
        prob = torch.softmax(logits, dim=1)
        unknown = 1.0 - prob.max(dim=1).values
        pred = logits.argmax(dim=1)
        return StepOut(logits, unknown, pred)
