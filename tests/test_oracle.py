from exp.stream import make_toy_stream
from exp.oracle import DelayedSparseLabelOracle

def test_oos_never_labeled():
    stream = make_toy_stream(T=10, B=10)
    oracle = DelayedSparseLabelOracle(stream, label_ratio=0.5, delay_steps=2, oos_name="ood", seed=0)
    for s in stream:
        if s.is_oos:
            for t in range(1, 30):
                assert oracle.get_label(s.sid, t) is None
