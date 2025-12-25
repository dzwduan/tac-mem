import json
import os
import subprocess
import tempfile


def test_split_config_integrity():
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, "out")
        # NOTE: this test assumes you can download HF dataset in your environment.
        # If running in offline CI, mark as xfail or pre-cache dataset.
        cmd = ["python3", "clinc150_semantic_split.py", "--out_dir", out_dir]
        subprocess.check_call(cmd)

        cfg_path = os.path.join(out_dir, "split_config.json")
        assert os.path.exists(cfg_path)

        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        assignments = cfg["assignments"]

        y0 = assignments["Y0"]
        phases = assignments["phases"]
        phase_intents = [it for p in phases for it in p["intents"]]

        # 1) coverage
        all_intents = set(y0) | set(phase_intents)
        assert len(all_intents) == 150, f"Expected 150 intents, got {len(all_intents)}"

        # 2) disjointness
        assert set(y0).isdisjoint(set(phase_intents)), "Y0 and phases must be disjoint"

        # 3) no duplicates within phases
        assert len(phase_intents) == len(set(phase_intents)), "Duplicate intents in phases"

        # 4) phase naming
        for i, p in enumerate(phases, start=1):
            assert p["name"] == f"Phase-{i}"
            assert isinstance(p["domain"], str) and p["domain"]

        # 5) determinism: running again should give identical config
        out_dir2 = os.path.join(td, "out2")
        subprocess.check_call(["python3", "clinc150_semantic_split.py", "--out_dir", out_dir2])
        cfg2 = json.load(open(os.path.join(out_dir2, "split_config.json"), "r", encoding="utf-8"))

        assert cfg["schedule"] == cfg2["schedule"]
        assert cfg["assignments"]["Y0"] == cfg2["assignments"]["Y0"]
        assert cfg["assignments"]["phases"] == cfg2["assignments"]["phases"]
