# -*- coding: utf-8 -*-
import pytest

from delayed_label_oracle import DelayedSparseLabelOracle, StreamSample


def make_toy_stream():
    """
    Build a small deterministic stream:
      - intent A: 10 samples at t=1..10
      - intent B: 10 samples at t=1..10
      - ood:      10 samples at t=1..10
    """
    samples = []
    sid = 0
    for t in range(1, 11):
        samples.append(StreamSample(sid=sid, t=t, utterance=f"a-{t}", label_name="intent_a", is_oos=False)); sid += 1
        samples.append(StreamSample(sid=sid, t=t, utterance=f"b-{t}", label_name="intent_b", is_oos=False)); sid += 1
        samples.append(StreamSample(sid=sid, t=t, utterance=f"o-{t}", label_name="ood", is_oos=True)); sid += 1
    return samples


def test_oos_never_labeled():
    samples = make_toy_stream()
    oracle = DelayedSparseLabelOracle(samples, label_ratio=0.5, delay=3, oos_name="ood", seed=0)

    oos_sids = [s.sid for s in samples if s.label_name == "ood"]
    for t in range(0, 50):
        for sid in oos_sids:
            assert oracle.get_label(sid, t) is None


def test_no_future_leakage_before_release():
    samples = make_toy_stream()
    delay = 5
    oracle = DelayedSparseLabelOracle(samples, label_ratio=0.3, delay=delay, oos_name="ood", seed=1)

    # For every labelable sample, before its release time, label must be None
    for s in samples:
        rt = oracle.get_release_time(s.sid)
        if rt is None:
            continue
        assert oracle.get_label(s.sid, rt - 1) is None
        assert oracle.get_label(s.sid, rt) == s.label_name


def test_label_ratio_per_intent_reasonable():
    samples = make_toy_stream()
    # each intent has 10 samples; label_ratio=0.2 -> k=int(10*0.2)=2
    oracle = DelayedSparseLabelOracle(samples, label_ratio=0.2, delay=0, oos_name="ood", seed=2, min_per_intent=0)

    a_sids = [s.sid for s in samples if s.label_name == "intent_a"]
    b_sids = [s.sid for s in samples if s.label_name == "intent_b"]

    a_labelable = sum(1 for sid in a_sids if oracle.get_release_time(sid) is not None)
    b_labelable = sum(1 for sid in b_sids if oracle.get_release_time(sid) is not None)

    assert a_labelable == 2
    assert b_labelable == 2


def test_determinism_with_seed():
    samples = make_toy_stream()
    o1 = DelayedSparseLabelOracle(samples, label_ratio=0.3, delay=7, oos_name="ood", seed=123)
    o2 = DelayedSparseLabelOracle(samples, label_ratio=0.3, delay=7, oos_name="ood", seed=123)
    o3 = DelayedSparseLabelOracle(samples, label_ratio=0.3, delay=7, oos_name="ood", seed=124)

    rel1 = {s.sid: o1.get_release_time(s.sid) for s in samples}
    rel2 = {s.sid: o2.get_release_time(s.sid) for s in samples}
    rel3 = {s.sid: o3.get_release_time(s.sid) for s in samples}

    assert rel1 == rel2
    assert rel1 != rel3


def test_min_per_intent_guard():
    samples = make_toy_stream()
    # label_ratio very small but enforce at least 1 per intent
    oracle = DelayedSparseLabelOracle(samples, label_ratio=0.01, delay=1, oos_name="ood", seed=0, min_per_intent=1)

    a_sids = [s.sid for s in samples if s.label_name == "intent_a"]
    b_sids = [s.sid for s in samples if s.label_name == "intent_b"]

    assert sum(1 for sid in a_sids if oracle.get_release_time(sid) is not None) >= 1
    assert sum(1 for sid in b_sids if oracle.get_release_time(sid) is not None) >= 1
