from __future__ import annotations
import torch
from .base import MethodBase, StepOut


class MSP(MethodBase):
    """Maximum Softmax Probability baseline.

    This method maintains a linear classifier head on top of a fixed backbone.
    Whenever labels are available for the current batch, the head is trained
    using a cross‑entropy loss. Unknown scores are computed as 1 minus the
    maximum softmax probability across classes.
    """

    def __init__(self, backbone, num_classes: int, device: str, lr: float = 1e-4):
        super().__init__("MSP", backbone, num_classes, device)
        hidden_dim = backbone.hidden_size
        self.head = torch.nn.Linear(hidden_dim, num_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr)

    def step(
        self, t: int, texts: List[str], labels: Optional[torch.Tensor], meta
    ) -> StepOut:
        # forward pass
        h = self.backbone(texts)
        logits = self.head(h)
        prob = torch.softmax(logits, dim=1)
        unknown = 1.0 - prob.max(dim=1).values
        pred = logits.argmax(dim=1)

        if labels is not None:
            # supervised cross‑entropy training
            loss = torch.nn.functional.cross_entropy(logits, labels)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return StepOut(logits, unknown, pred)