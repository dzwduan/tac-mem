import os
from exp.config import load_config

def test_load_yaml():
    config_path = os.path.join(os.path.dirname(__file__), "../tfowsl_exp/configs/toy.yaml")
    cfg = load_config(config_path)
    assert cfg["metrics"]["delay"]["mode"] in ["domain", "intent"]
    assert cfg["metrics"]["fer"]["mode"] in ["oos_trigger", "mixture"]