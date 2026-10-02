"""Build a zero-embedding memory bank as a same-architecture negative control."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="hierarchical_clustering_results_yc2_70B.pkl")
    parser.add_argument("--output", default="hierarchical_clustering_results_yc2_zero.pkl")
    args = parser.parse_args()
    with Path(args.source).open("rb") as file:
        bank = pickle.load(file)
    count = 0
    for clusters in bank.values():
        for cluster in clusters.values():
            cluster["clip_embedding"] = np.zeros_like(cluster["clip_embedding"])
            count += 1
    with Path(args.output).open("wb") as file:
        pickle.dump(bank, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.output}: zeroed {count} cluster embeddings")


if __name__ == "__main__":
    main()
