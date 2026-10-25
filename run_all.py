"""One command for the whole pipeline.

    python run_all.py --quick   # smoke test (<1 min on CPU): tiny dataset, 1 epoch per model,
                                # written to a temp folder so real checkpoints/results are untouched
    python run_all.py           # full run: generate data, train 3 models x 3 seeds, evaluate
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from utils.config import ROOT, load_config


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def quick():
    from dataset.generator import save_all
    from train import train_one

    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_all(cfg, tmp / "data", cfg["quick"]["bursts_per_class_snr"])
        for m in ("cnn_base", "cnn_plain_deep", "resnet1d"):
            train_one(cfg, m, 0, tmp / "data", epochs=1, out_root=tmp)
    print("quick run OK")


def full():
    py = sys.executable
    run([py, "-m", "dataset.generator"])
    run([py, "-m", "pytest", "-q"])
    run([py, "train.py", "--all"])
    run([py, "evaluate.py"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    t0 = time.time()
    quick() if ap.parse_args().quick else full()
    print(f"done in {time.time() - t0:.0f} s")
