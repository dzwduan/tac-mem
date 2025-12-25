from __future__ import annotations
import torch
from .base import MethodBase, StepOut


class Energy(MethodBase):
    def __init__(self, backbone, num_classes: int, device: str):
        super().__init__("Energy", backbone, num_classes, device)
        H = backbone.encoder.config.hidden_size
        self.head = torch.nn.Linear(H, num_classes).to(device)

    def step(self, t, texts, labels, meta):
        h = self.backbone(texts)
        logits = self.head(h)
        unknown = -torch.logsumexp(logits, dim=1)   # higher -> more unknown (consistent within this codebase)
        pred = logits.argmax(dim=1)
        return StepOut(logits, unknown, pred)
