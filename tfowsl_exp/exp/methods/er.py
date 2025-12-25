from __future__ import annotations
import random
import torch
from .base import MethodBase, StepOut


class ER(MethodBase):
    def __init__(self, backbone, num_classes: int, device: str, lr: float, buffer_size: int):
        super().__init__("ER", backbone, num_classes, device)
        H = backbone.encoder.config.hidden_size
        self.head = torch.nn.Linear(H, num_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr)
        self.buffer_size = buffer_size
        self.buffer = []  # list[(text, y_id)]

    def _add(self, texts, y):
        for x, yi in zip(texts, y.tolist()):
            self.buffer.append((x, int(yi)))
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

    def step(self, t, texts, labels, meta):
        h = self.backbone(texts)
        logits = self.head(h)
        prob = torch.softmax(logits, dim=1)
        unknown = 1.0 - prob.max(dim=1).values
        pred = logits.argmax(dim=1)

        if labels is not None:
            replay = []
            if len(self.buffer) > 0:
                k = min(len(self.buffer), len(texts))
                replay = random.sample(self.buffer, k)
            all_texts = list(texts) + [x for x, _ in replay]
            all_y = labels.tolist() + [yy for _, yy in replay]
            all_y = torch.tensor(all_y, device=self.device, dtype=torch.long)

            h2 = self.backbone(all_texts)
            logits2 = self.head(h2)
            loss = torch.nn.functional.cross_entropy(logits2, all_y)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self._add(texts, labels)

        return StepOut(logits, unknown, pred)
