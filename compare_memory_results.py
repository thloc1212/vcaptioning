"""Print side-by-side HiCM2 evaluation metrics and medoid minus LLM deltas."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llm_metrics", type=Path)
    parser.add_argument("medoid_metrics", type=Path)
    args = parser.parse_args()
    llm = json.loads(args.llm_metrics.read_text())
    medoid = json.loads(args.medoid_metrics.read_text())
    print(f"{'metric':32} {'LLM':>12} {'medoid':>12} {'delta':>12}")
    for key in sorted(llm.keys() & medoid.keys()):
        a, b = llm[key], medoid[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            print(f"{key:32} {a:12.4f} {b:12.4f} {b-a:+12.4f}")


if __name__ == "__main__":
    main()
