from __future__ import annotations
import torch
from .base import MethodBase, StepOut


class FixedThresholdExpand(MethodBase):
    def __init__(
        self,
        backbone,
        num_classes: int,
        device: str,
        lr: float,
        thr: float,
        frac_trigger: float,
        new_class_per_trigger: int,
        pseudo_topk: int,
        pseudo_weight: float,
    ):
        super().__init__("FixedThresholdExpand", backbone, num_classes, device)
        H = backbone.encoder.config.hidden_size
        self.head = torch.nn.Linear(H, num_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr)

        self.thr = thr
        self.frac_trigger = frac_trigger
        self.new_class_per_trigger = new_class_per_trigger
        self.pseudo_topk = pseudo_topk
        self.pseudo_weight = pseudo_weight

    def _expand_head(self, k: int):
        old = self.head
        in_f = old.in_features
        out_f = old.out_features + k
        new = torch.nn.Linear(in_f, out_f).to(self.device)
        with torch.no_grad():
            new.weight[: old.out_features].copy_(old.weight)
            new.bias[: old.out_features].copy_(old.bias)
        self.head = new
        self.created_classes += k

    def step(self, t, texts, labels, meta):
        h = self.backbone(texts)
        logits = self.head(h)
        prob = torch.softmax(logits, dim=1)
        unknown = 1.0 - prob.max(dim=1).values
        pred = logits.argmax(dim=1)

        # trigger expand
        frac = (unknown > self.thr).float().mean().item()
        if frac >= self.frac_trigger:
            # record trigger sample = top unknown
            top_idx = int(torch.argmax(unknown).item())
            self.events.append({"t": t, "type": "create", "trigger": meta[top_idx]})
            self._expand_head(self.new_class_per_trigger)

        # training:
        losses = []
        if labels is not None:
            losses.append(torch.nn.functional.cross_entropy(logits, labels))

        # pseudo-label: assign topK unknown to newest class if expanded
        if self.created_classes > 0:
            new_class_id = self.head.out_features - 1
            k = min(self.pseudo_topk, len(texts))
            idx = torch.topk(unknown, k=k).indices
            pseudo_y = torch.full((k,), new_class_id, device=self.device, dtype=torch.long)
            losses.append(self.pseudo_weight * torch.nn.functional.cross_entropy(self.head(h[idx]), pseudo_y))
            # track assignment for mixture FER
            for j in idx.tolist():
                self.events.append({"t": t, "type": "assign", "pred_id": int(new_class_id), "meta": meta[j]})

        if losses:
            loss = sum(losses)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return StepOut(logits, unknown, pred)
