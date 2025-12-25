# -*- coding: utf-8 -*-
import torch

from backbone import build_shared_backbone
from methods import ERBaseline, OSRBaseline, TACMemMethod


def test_all_methods_share_same_backbone_object():
    backbone = build_shared_backbone(model_name="distilbert-base-uncased", device="cpu")
    m1 = ERBaseline(backbone, num_classes=10)
    m2 = OSRBaseline(backbone, num_classes=10)
    m3 = TACMemMethod(backbone, num_classes=10)

    assert m1.backbone is backbone
    assert m2.backbone is backbone
    assert m3.backbone is backbone

    # also ensure underlying encoder object is identical (shared weights)
    assert m1.backbone.encoder is m2.backbone.encoder
    assert m2.backbone.encoder is m3.backbone.encoder


def test_shared_backbone_produces_identical_embeddings():
    backbone = build_shared_backbone(model_name="distilbert-base-uncased", device="cpu")
    texts = ["hello world", "book a meeting tomorrow"]

    e1 = backbone.encode(texts)
    e2 = backbone.encode(texts)

    assert e1.shape == e2.shape
    # In eval + no_grad, embeddings should be identical across repeated calls
    assert torch.allclose(e1, e2, atol=0, rtol=0)


def test_methods_only_change_heads_not_backbone():
    backbone = build_shared_backbone(model_name="distilbert-base-uncased", device="cpu")
    # snapshot a backbone parameter tensor (should not change in observe)
    p_before = next(backbone.encoder.parameters()).detach().clone()

    m = ERBaseline(backbone, num_classes=10)
    _ = m.observe(["check my balance", "what is the weather"])

    p_after = next(backbone.encoder.parameters()).detach().clone()
    assert torch.allclose(p_before, p_after), "Backbone params changed during observe (shouldn't in this step)."
