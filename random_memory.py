"""Build a random-exemplar bank with the same FINCH hierarchy as HiCM2.

Every cluster receives one uniformly sampled original sentence. Its CLIP
embedding replaces the LLaMA summary embedding. This is a seeded control for
medoid selection, not a random-vector bank. Load trusted pickle files only.
"""

import argparse
import copy
import pickle
from pathlib import Path

import numpy as np

from medoid_memory import encode_clip, original_sentences


def choose_representatives(source, seed):
    rng = np.random.default_rng(seed)
    selected = {}
    for level in sorted(source, key=lambda name: int(name.split("_")[1])):
        for key in sorted(source[level], key=lambda name: int(name.split("_")[1])):
            indices = np.asarray(source[level][key]["indices"], dtype=np.int64)
            if not len(indices):
                raise ValueError(f"empty cluster: {level}/{key}")
            selected[(level, key)] = int(rng.choice(indices))
    return selected


def build_random_bank(source, selected, encoded_by_index):
    sentences = original_sentences(source)
    result = copy.deepcopy(source)
    for (level, key), index in selected.items():
        cluster = result[level][key]
        if index not in cluster["indices"]:
            raise ValueError(f"index {index} is not in {level}/{key}")
        vector = np.asarray(encoded_by_index[index])
        original_vector = np.asarray(source[level][key]["clip_embedding"])
        if vector.shape != original_vector.shape[1:]:
            raise ValueError(f"CLIP shape {vector.shape} differs from {original_vector.shape}")
        cluster["summary"] = sentences[index]
        cluster["clip_embedding"] = vector[None, :].astype(original_vector.dtype)
        cluster["random_index"] = index
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="hierarchical_clustering_results_yc2_70B.pkl")
    parser.add_argument("--output", default="hierarchical_clustering_results_yc2_random_seed42.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    with Path(args.source).open("rb") as file:
        source = pickle.load(file)
    sentences = original_sentences(source)
    selected = choose_representatives(source, args.seed)
    unique_indices = sorted(set(selected.values()))
    vectors = encode_clip([sentences[index] for index in unique_indices],
                          batch_size=args.batch_size, device=args.device)
    encoded_by_index = dict(zip(unique_indices, vectors))
    result = build_random_bank(source, selected, encoded_by_index)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {output}: {len(selected)} clusters, "
          f"{len(unique_indices)} unique sentences, seed={args.seed}")


if __name__ == "__main__":
    main()
