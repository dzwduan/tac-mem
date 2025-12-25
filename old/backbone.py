# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class BackboneConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 64
    device: str = "cpu"


class TextBackbone(torch.nn.Module):
    """
    A unified text encoder (backbone).
    Output: sentence-level embedding h(x) of shape [B, H]
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        self.encoder.to(cfg.device)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts to embeddings.
        Uses mean pooling over token embeddings with attention mask.
        """
        self.encoder.eval()
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(self.cfg.device) for k, v in tok.items()}

        out = self.encoder(**tok)  # last_hidden_state: [B, L, H]
        hidden = out.last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        hidden = hidden * mask

        # mean pool over valid tokens
        denom = mask.sum(dim=1).clamp(min=1)
        emb = hidden.sum(dim=1) / denom  # [B, H]
        return emb

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


def build_shared_backbone(
    model_name: str = "distilbert-base-uncased",
    max_length: int = 64,
    device: str = "cpu",
) -> TextBackbone:
    """
    Single factory for a unified backbone instance.
    All methods should receive this SAME instance.
    """
    cfg = BackboneConfig(model_name=model_name, max_length=max_length, device=device)
    return TextBackbone(cfg)
