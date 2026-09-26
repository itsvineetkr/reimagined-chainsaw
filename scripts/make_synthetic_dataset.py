"""Generate a synthetic dataset with the challenge layout (for smoke tests / demos only).

python scripts/make_synthetic_dataset.py --out dataset_synth --n-train 3000 --n-test 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from er_common.synth import write_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dataset_synth")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = write_dataset(args.out, args.n_train, args.n_test, args.seed)
    print(f"wrote synthetic dataset to {out}/(train|test)")


if __name__ == "__main__":
    main()
