# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from backbone import TextBackbone


@dataclass
class StepOutput:
    logits: torch.Tensor        # [B, C]
    unknown_score: torch.Tensor # [B]
    method_name: str


class BaseMethod:
    """
    All methods must take a shared backbone (encoder) and ONLY differ in heads/logic.
    """

    def __init__(self, backbone: TextBackbone, num_classes: int):
        self.backbone = backbone
        self.num_classes = num_classes

    def observe(self, texts: List[str]) -> StepOutput:
        raise NotImplementedError


class ERBaseline(BaseMethod):
    """
    Placeholder for a CL baseline: uses a linear classifier head on shared embeddings.
    (Real ER would also maintain replay buffer; omitted here.)
    """

    def __init__(self, backbone: TextBackbone, num_classes: int):
        super().__init__(backbone, num_classes)
        # head only (method-specific)
        hidden = self.backbone.encoder.config.hidden_size
        self.head = torch.nn.Linear(hidden, num_classes).to(self.backbone.cfg.device)

    @torch.no_grad()
    def observe(self, texts: List[str]) -> StepOutput:
        h = self.backbone(texts)                    # [B, H]
        logits = self.head(h)                       # [B, C]
        # simple unknown score: negative max logit (dummy; replace later)
        unknown = -logits.max(dim=1).values
        return StepOutput(logits=logits, unknown_score=unknown, method_name="ER")


class OSRBaseline(BaseMethod):
    """
    Placeholder for an open-set method: still uses shared backbone,
    but unknown score is computed differently.
    """

    def __init__(self, backbone: TextBackbone, num_classes: int):
        super().__init__(backbone, num_classes)
        hidden = self.backbone.encoder.config.hidden_size
        self.head = torch.nn.Linear(hidden, num_classes).to(self.backbone.cfg.device)

    @torch.no_grad()
    def observe(self, texts: List[str]) -> StepOutput:
        h = self.backbone(texts)
        logits = self.head(h)
        prob = torch.softmax(logits, dim=1)
        # unknown score: 1 - max softmax (common baseline score)
        unknown = 1.0 - prob.max(dim=1).values
        return StepOutput(logits=logits, unknown_score=unknown, method_name="OSR")


class TACMemMethod(BaseMethod):
    """
    Placeholder for your method: shares backbone, changes decision logic.
    (Real TAC-Mem would include time-consistency, absorb-or-create, memory, etc.)
    """

    def __init__(self, backbone: TextBackbone, num_classes: int):
        super().__init__(backbone, num_classes)
        hidden = self.backbone.encoder.config.hidden_size
        self.head = torch.nn.Linear(hidden, num_classes).to(self.backbone.cfg.device)

    @torch.no_grad()
    def observe(self, texts: List[str]) -> StepOutput:
        h = self.backbone(texts)
        logits = self.head(h)

        # example "time-consistent-ish" score placeholder:
        # use logsumexp margin style score (still dummy)
        lse = torch.logsumexp(logits, dim=1)
        mx = logits.max(dim=1).values
        unknown = -(mx - lse)  # smaller margin -> higher unknown

        return StepOutput(logits=logits, unknown_score=unknown, method_name="TAC-Mem")
