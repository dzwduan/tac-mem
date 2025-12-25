from __future__ import annotations
import torch
from .base import MethodBase, StepOut

class TACMem(MethodBase):
    def __init__(
        self,
        backbone,
        num_classes: int,
        device: str,
        lr: float,
        thr: float,
        ema_beta: float,
        min_cons_steps: int,
        min_unknown: int,
        risk_oos_max: float,
        pseudo_topk: int,
        pseudo_weight: float,
    ):
        super().__init__("TACMem", backbone, num_classes, device)
        H = backbone.encoder.config.hidden_size
        self.head = torch.nn.Linear(H, num_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr)

        self.thr = thr
        self.ema_beta = ema_beta
        self.min_cons_steps = min_cons_steps
        self.min_unknown = min_unknown
        self.risk_oos_max = risk_oos_max
        self.pseudo_topk = pseudo_topk
        self.pseudo_weight = pseudo_weight
        
    def step(self, t, texts, labels, meta):
        # minimal implementation: just forward pass
        h = self.backbone(texts)
        logits = self.head(h)
        prob = torch.softmax(logits, dim=1)
        unknown = 1.0 - prob.max(dim=1).values
        pred = logits.argmax(dim=1)
        
        # Training (supervised)
        if labels is not None:
             loss = torch.nn.functional.cross_entropy(logits, labels)
             self.opt.zero_grad()
             loss.backward()
             self.opt.step()

        return StepOut(logits, unknown, pred)
