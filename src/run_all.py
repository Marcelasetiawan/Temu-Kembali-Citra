"""Menjalankan 3 model x 3 skenario secara berurutan. Run yang sudah selesai dilewati."""
import argparse

from common import MODELS, SCENARIOS, run_dir
from summarize import summarize
from train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    for scenario in SCENARIOS:
        for arch in MODELS:
            if (run_dir(scenario, arch) / "metrics.json").exists():
                print(f"Lewati {scenario} | {arch} (sudah selesai)", flush=True)
                continue
            train(scenario, arch, epochs=args.epochs, workers=args.workers)
            summarize()
    summarize()
