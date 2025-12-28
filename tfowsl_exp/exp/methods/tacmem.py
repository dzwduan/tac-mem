from __future__ import annotations
import torch
from .base import MethodBase, StepOut


class TACMem(MethodBase):
    """
    Implementation of a simple variant of the TAC‑Mem algorithm.

    This version augments the minimal baseline with a prototype memory,
    exponential moving average tracking of unknown scores and a more
    conservative class creation policy. It creates a new class only when
    the model has seen a high proportion of unknown predictions for a
    sustained period and the average unknown score indicates the model
    is sufficiently confident that the samples are truly out‑of‑scope.

    Parameters
    ----------
    backbone : object
        A text encoder returning hidden vectors of dimension `H`.  It
        must implement the call interface `backbone(texts: List[str]) -> Tensor[B,H]`.
    num_classes : int
        Number of in‑scope classes at initialization.
    device : str
        Device identifier (`"cpu"` or `"cuda"`).
    lr : float
        Learning rate for the classifier head.
    thr : float
        Threshold on unknown scores above which a sample is considered
        potentially out‑of‑scope.
    ema_beta : float
        Momentum term (between 0 and 1) for the exponential moving average
        of the batch unknown scores.
    min_cons_steps : int
        Minimum number of consecutive steps where the unknown EMA exceeds
        the threshold before creating a new class.
    min_unknown : int
        Minimum number of samples in the current batch whose unknown
        scores exceed the threshold before creating a new class.
    risk_oos_max : float
        Upper bound on the unknown EMA for triggering class creation;
        prevents expansion when the unknown scores are extremely high
        (indicating the model is very uncertain about everything).
    pseudo_topk : int
        Number of most unknown samples to pseudo‑label when creating a
        new class.
    pseudo_weight : float
        Weight applied to the pseudo‑label cross‑entropy loss.
    """

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
    ) -> None:
        super().__init__("TACMem", backbone, num_classes, device)
        H = backbone.encoder.config.hidden_size
        self.head = torch.nn.Linear(H, num_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr)

        # Hyperparameters
        self.thr = thr
        self.ema_beta = ema_beta
        self.min_cons_steps = min_cons_steps
        self.min_unknown = min_unknown
        self.risk_oos_max = risk_oos_max
        self.pseudo_topk = pseudo_topk
        self.pseudo_weight = pseudo_weight

        # Prototype memory for each class (lazy initialisation on first label)
        self.memory: dict[int, torch.Tensor] = {c: None for c in range(num_classes)}
        # Exponential moving average of batch unknown scores
        self.unknown_ema = 0.0
        # Number of consecutive steps above threshold
        self.cons_unknown_steps = 0

    def _expand_head(self, k: int = 1) -> int:
        """Expand the classification head by k outputs and return the id of the
        first new class."""
        old = self.head
        in_f = old.in_features
        out_f = old.out_features + k
        new = torch.nn.Linear(in_f, out_f).to(self.device)
        with torch.no_grad():
            # copy old weights
            new.weight[: old.out_features].copy_(old.weight)
            new.bias[: old.out_features].copy_(old.bias)
        self.head = new
        new_class_id = self.head.out_features - k
        self.created_classes += k
        return new_class_id

    def step(self, t: int, texts, labels, meta) -> StepOut:
        # Compute embeddings and logits
        h = self.backbone(texts)
        logits = self.head(h)
        prob = torch.softmax(logits, dim=1)
        # Unknown score: 1 - max softmax probability
        unknown = 1.0 - prob.max(dim=1).values
        pred = logits.argmax(dim=1)

        losses = []
        # Update prototype memory with supervised labels (if provided for the whole batch)
        if labels is not None:
            with torch.no_grad():
                for cid in torch.unique(labels).tolist():
                    idx = (labels == cid).nonzero(as_tuple=False).flatten()
                    proto = h[idx].detach().mean(dim=0)
                    if self.memory[cid] is None:
                        self.memory[cid] = proto
                    else:
                        self.memory[cid] = self.ema_beta * self.memory[cid] + (1 - self.ema_beta) * proto
            losses.append(torch.nn.functional.cross_entropy(logits, labels))

        # Update EMA of unknown scores and consecutive step counter
        batch_unknown_mean = unknown.mean().item()
        self.unknown_ema = self.ema_beta * self.unknown_ema + (1 - self.ema_beta) * batch_unknown_mean
        if self.unknown_ema > self.thr:
            self.cons_unknown_steps += 1
        else:
            self.cons_unknown_steps = 0
        # Number of high unknown samples in current batch
        num_unknown = int((unknown > self.thr).sum().item())

        # Decide whether to create a new class
        if (
            self.cons_unknown_steps >= self.min_cons_steps
            and num_unknown >= self.min_unknown
            and self.unknown_ema < self.risk_oos_max
        ):
            # Expand head by one new class
            new_class_id = self._expand_head(1)
            # Identify top‑k most uncertain samples
            k = min(self.pseudo_topk, len(texts))
            top_idx = torch.topk(unknown, k=k).indices
            # Record creation event and the trigger sample metadata
            trigger_meta = meta[int(top_idx[0].item())]
            self.events.append({"t": t, "type": "create", "trigger": trigger_meta})
            # Compute prototype as mean embedding of the selected samples
            new_proto = h[top_idx].detach().mean(dim=0)
            self.memory[new_class_id] = new_proto
            # Assign pseudo‑labels to top‑k unknown samples
            pseudo_y = torch.full((k,), new_class_id, device=self.device, dtype=torch.long)
            # Compute pseudo‑label loss
            losses.append(self.pseudo_weight * torch.nn.functional.cross_entropy(self.head(h[top_idx]), pseudo_y))
            # Record assignment events for FER evaluation
            for j in top_idx.tolist():
                self.events.append({"t": t, "type": "assign", "pred_id": int(new_class_id), "meta": meta[j]})

        # Optimise model
        if losses:
            loss = sum(losses)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return StepOut(logits, unknown, pred)