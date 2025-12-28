from __future__ import annotations
import torch
from sklearn.cluster import KMeans
from .base import MethodBase, StepOut


class ClusterExpand(MethodBase):
    def __init__(self, backbone, num_classes: int, device: str, lr: float, thr: float, min_unknown: int, k: int, pseudo_weight: float):
        super().__init__("ClusterExpand", backbone, num_classes, device)
        H = backbone.hidden_size
        self.head = torch.nn.Linear(H, num_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr)

        self.thr = thr
        self.min_unknown = min_unknown
        self.k = k
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

        idx = (unknown > self.thr).nonzero(as_tuple=False).flatten()
        if idx.numel() >= self.min_unknown:
            # cluster unknown embeddings into k groups, create k new classes
            x = h[idx].detach().cpu().numpy()
            km = KMeans(n_clusters=self.k, n_init="auto", random_state=0).fit(x)
            self.events.append({"t": t, "type": "create", "trigger": meta[int(idx[0].item())], "k": self.k})
            self._expand_head(self.k)

            # pseudo label each cluster to a new class id
            base = self.head.out_features - self.k
            for c in range(self.k):
                members = idx[(torch.tensor(km.labels_) == c).nonzero(as_tuple=False).flatten()]
                if members.numel() == 0:
                    continue
                cid = base + c
                pseudo_y = torch.full((members.numel(),), cid, device=self.device, dtype=torch.long)
                loss_p = self.pseudo_weight * torch.nn.functional.cross_entropy(self.head(h[members]), pseudo_y)
                self.opt.zero_grad()
                loss_p.backward()
                self.opt.step()
                for j in members.tolist():
                    self.events.append({"t": t, "type": "assign", "pred_id": int(cid), "meta": meta[j]})

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return StepOut(logits, unknown, pred)
