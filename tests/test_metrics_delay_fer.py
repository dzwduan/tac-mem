from exp.metrics.delay import compute_first_appearance, compute_trigger_times, compute_delay_stats
from exp.metrics.fer import fer_oos_trigger, fer_mixture

def test_delay_modes():
    stream_meta = [
        {"t": 1, "true_intent": "intent_a", "is_oos": False},
        {"t": 2, "true_intent": "intent_b", "is_oos": False},
        {"t": 3, "true_intent": "ood", "is_oos": True},
    ]
    intent2domain = {"intent_a": "dom1", "intent_b": "dom1"}
    first_dom = compute_first_appearance(stream_meta, "domain", intent2domain, "ood")
    first_int = compute_first_appearance(stream_meta, "intent", intent2domain, "ood")
    assert "dom1" in first_dom
    assert "intent_a" in first_int

    events = [{"t": 5, "type": "create", "trigger": {"true_intent": "intent_a", "is_oos": False}}]
    trig_dom = compute_trigger_times(events, "domain", intent2domain, "ood")
    trig_int = compute_trigger_times(events, "intent", intent2domain, "ood")
    assert trig_dom["dom1"] == 5
    assert trig_int["intent_a"] == 5

    stat = compute_delay_stats(first_dom, trig_dom)
    assert stat["mean_delay"] is not None

def test_fer_modes():
    events = [
        {"t": 5, "type": "create", "trigger": {"true_intent": "ood", "is_oos": True}},
        {"t": 10, "type": "create", "trigger": {"true_intent": "intent_a", "is_oos": False}},
        {"t": 10, "type": "assign", "pred_id": 151, "meta": {"true_intent": "intent_a", "is_oos": False}},
        {"t": 11, "type": "assign", "pred_id": 151, "meta": {"true_intent": "intent_b", "is_oos": False}},
        {"t": 12, "type": "assign", "pred_id": 151, "meta": {"true_intent": "ood", "is_oos": True}},
        {"t": 13, "type": "assign", "pred_id": 151, "meta": {"true_intent": "intent_a", "is_oos": False}},
        {"t": 14, "type": "assign", "pred_id": 151, "meta": {"true_intent": "intent_a", "is_oos": False}},
    ]
    s1 = fer_oos_trigger(events, "ood")
    assert s1["n_create"] == 2
    assert 0.0 <= s1["fer"] <= 1.0

    s2 = fer_mixture(events, "ood", purity_threshold=0.7, oos_ratio_threshold=0.3, min_support=3)
    assert 0.0 <= s2["fer"] <= 1.0
