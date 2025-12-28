from __future__ import annotations
import torch
from .base import MethodBase, StepOut


class Energy(MethodBase):
    """Energy baseline for open set recognition.

    This baseline trains the classifier head just like MSP, but uses
    negative energy (−logsumexp of logits) as the unknown score. Higher energy
    corresponds to higher confidence for known classes, so we take its
    negative as the unknown score for consistency (larger values → more unknown).
    """

    def __init__(self, backbone, num_classes: int, device: str, lr: float = 1e-4):
        super().__init__("Energy", backbone, num_classes, device)
        hidden_dim = backbone.hidden_size
        self.head = torch.nn.Linear(hidden_dim, num_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr)

    def step(
        self, t: int, texts: List[str], labels: Optional[torch.Tensor], meta
    ) -> StepOut:
        h = self.backbone(texts)
        logits = self.head(h)
        # energy score (more negative → less confident in known classes)
        energy = -torch.logsumexp(logits, dim=1)
        # convert energy to unknown score (higher → more unknown)
        unknown = -energy
        pred = logits.argmax(dim=1)

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return StepOut(logits, unknown, pred)


class ER(MethodBase):
    """Experience Replay baseline.

    This baseline maintains a replay buffer of past labelled examples to
    mitigate forgetting. On each step with supervision, it trains the
    classifier head on a mixture of current labelled data and samples from
    the buffer. Unknown scores are computed as 1 − max softmax probability,
    matching MSP.
    """

    def __init__(self, backbone, num_classes: int, device: str, lr: float = 1e-4, buffer_size: int = 2000):
        super().__init__("ER", backbone, num_classes, device)
        hidden_dim = backbone.hidden_size
        self.head = torch.nn.Linear(hidden_dim, num_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr)
        self.buffer_size = buffer_size
        # store (text, class_id) tuples
        self.buffer: List[Tuple[str, int]] = []

    def _add_to_buffer(self, texts: List[str], labels: torch.Tensor) -> None:
        for x, y in zip(texts, labels.tolist()):
            self.buffer.append((x, int(y)))
        # keep only the most recent samples up to buffer_size
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size :]

    def step(
        self, t: int, texts: List[str], labels: Optional[torch.Tensor], meta
    ) -> StepOut:
        # forward pass on current batch
        h = self.backbone(texts)
        logits = self.head(h)
        prob = torch.softmax(logits, dim=1)
        unknown = 1.0 - prob.max(dim=1).values
        pred = logits.argmax(dim=1)

        if labels is not None:
            # assemble training batch: current labelled samples + replay
            batch_texts = list(texts)
            batch_labels = labels.clone().detach().to(self.device)
            # sample from buffer (up to size of current batch)
            if len(self.buffer) > 0:
                k = min(len(self.buffer), len(texts))
                replay_samples = random.sample(self.buffer, k)
                batch_texts.extend([x for x, _ in replay_samples])
                replay_y = torch.tensor([y for _, y in replay_samples], device=self.device, dtype=torch.long)
                batch_labels = torch.cat([batch_labels, replay_y], dim=0)

            # compute embeddings and logits for the combined batch
            h2 = self.backbone(batch_texts)
            logits2 = self.head(h2)
            # cross‑entropy loss on combined batch
            loss = torch.nn.functional.cross_entropy(logits2, batch_labels)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # add current labelled samples to buffer
            self._add_to_buffer(texts, labels)

        return StepOut(logits, unknown, pred)