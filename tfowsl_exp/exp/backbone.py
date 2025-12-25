from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class BackboneCfg:
    name: str
    max_length: int
    device: str


class TextBackbone(torch.nn.Module):
    def __init__(self, cfg: BackboneCfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(cfg.name).to(cfg.device)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        self.encoder.eval()
        tok = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt"
        )
        tok = {k: v.to(self.cfg.device) for k, v in tok.items()}
        out = self.encoder(**tok)
        hidden = out.last_hidden_state  # [B,L,H]
        mask = tok["attention_mask"].unsqueeze(-1)  # [B,L,1]
        hidden = hidden * mask
        denom = mask.sum(dim=1).clamp(min=1)
        emb = hidden.sum(dim=1) / denom
        return emb

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


def build_backbone(name: str, max_length: int, device: str) -> TextBackbone:
    return TextBackbone(BackboneCfg(name=name, max_length=max_length, device=device))
