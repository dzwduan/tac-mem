import os
import subprocess
import pandas as pd
import sys

def test_runner_toy_smoke():
    # Setup paths
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(tests_dir, "../tfowsl_exp"))
    
    # run toy config (fast)
    # We run from project_root so 'exp' module is found and config path 'configs/toy.yaml' works
    subprocess.check_call(
        [sys.executable, "-m", "exp.run", "--config", "configs/toy.yaml"],
        cwd=project_root
    )
    
    out_dir = os.path.join(project_root, "outputs/toy_run")
    assert os.path.exists(os.path.join(out_dir, "summary.csv"))
    df = pd.read_csv(os.path.join(out_dir, "summary.csv"))
    assert len(df) >= 1
    assert os.path.exists(os.path.join(out_dir, "plots", "oos_auc_bar.png"))