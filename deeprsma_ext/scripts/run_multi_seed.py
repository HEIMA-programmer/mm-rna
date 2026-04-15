"""
Multi-seed driver: runs train_cv.py for seeds=[1,2,3] (or custom) on a given config.

Usage:
  python deeprsma_ext/scripts/run_multi_seed.py --use-adapter --use-bias --epochs 100 [--seeds 1,2,3]

Equivalent to calling train_cv.py three times with --seed 1, --seed 2, --seed 3.
Passes through all other args.
"""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='1,2,3')
    # Pass-through all other args
    known, rest = p.parse_known_args()
    seeds = [int(s.strip()) for s in known.seeds.split(',') if s.strip()]

    for s in seeds:
        cmd = [
            sys.executable,
            str(ROOT / "deeprsma_ext" / "scripts" / "train_cv.py"),
            "--seed", str(s),
        ] + rest
        print(f"\n\n===== Launching seed={s}: {' '.join(cmd)} =====\n", flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"[warn] seed={s} exited with code {rc}; continuing to next seed")

    print("\nAll seeds done. Use compare_configs.py to aggregate.")


if __name__ == '__main__':
    main()
