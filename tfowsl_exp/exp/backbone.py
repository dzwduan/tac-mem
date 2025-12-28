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
    num_gpus: int | None


class TextBackbone(torch.nn.Module):
    def __init__(self, cfg: BackboneCfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name, use_fast=True)
        base_model = AutoModel.from_pretrained(cfg.name).to(cfg.device)
        self.hidden_size = base_model.config.hidden_size

        # Optionally wrap the encoder with DataParallel to use multiple GPUs.
        self.primary_device = torch.device(cfg.device)
        self.encoder = base_model
        self.data_parallel = False
        if torch.cuda.is_available() and self.primary_device.type == "cuda":
            total_gpus = torch.cuda.device_count()
            requested = cfg.num_gpus or total_gpus
            if total_gpus > 1 and requested > 1:
                start = self.primary_device.index or 0
                device_ids = list(range(start, min(start + requested, total_gpus)))
                self.encoder = torch.nn.DataParallel(base_model, device_ids=device_ids)
                self.primary_device = torch.device(f"cuda:{device_ids[0]}")
                self.data_parallel = True

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        self.encoder.eval()
        tok = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt"
        )
        tok = {k: v.to(self.primary_device) for k, v in tok.items()}
        out = self.encoder(**tok)
        hidden = out.last_hidden_state  # [B,L,H]
        mask = tok["attention_mask"].unsqueeze(-1)  # [B,L,1]
        hidden = hidden * mask
        denom = mask.sum(dim=1).clamp(min=1)
        emb = hidden.sum(dim=1) / denom
        return emb

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


def build_backbone(name: str, max_length: int, device: str, num_gpus: int | None) -> TextBackbone:
    return TextBackbone(BackboneCfg(name=name, max_length=max_length, device=device, num_gpus=num_gpus))
