"""Replace LLaMA summaries with CLIP sentence medoids in a HiCM2 memory bank.

The released bank supplies the FINCH partitions and hierarchy. No LLM is used.
Only load pickle files from trusted sources.
"""

import argparse
import copy
import pickle
from pathlib import Path

import numpy as np


def original_sentences(bank):
    """Reconstruct FINCH input order from level_1's original sentence indices."""
    indexed = {}
    for cluster in bank["level_1"].values():
        indices, sentences = cluster["indices"], cluster["sentences"]
        if len(indices) != len(sentences):
            raise ValueError("level_1 indices and sentences have different lengths")
        for index, sentence in zip(indices, sentences):
            index = int(index)
            if index in indexed and indexed[index] != sentence:
                raise ValueError(f"conflicting sentence at index {index}")
            indexed[index] = sentence
    if sorted(indexed) != list(range(len(indexed))):
        raise ValueError("level_1 does not cover a contiguous sentence index range")
    return [indexed[i] for i in range(len(indexed))]


def encode_clip(sentences, model_name="ViT-L/14", batch_size=128, device=None):
    import clip
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load(model_name, device=device)
    model.eval()
    chunks = []
    for start in range(0, len(sentences), batch_size):
        tokens = clip.tokenize(sentences[start:start + batch_size], truncate=True).to(device)
        with torch.no_grad():
            chunks.append(model.encode_text(tokens).float().cpu().numpy())
        print(f"CLIP encoded {min(start + batch_size, len(sentences))}/{len(sentences)}", flush=True)
    return np.concatenate(chunks).astype(np.float32)


def medoid_index(indices, embeddings):
    """Exact cosine medoid: maximize mean cosine to all cluster members."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("empty FINCH cluster")
    vectors = np.asarray(embeddings[indices], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    scores = normalized @ normalized.sum(axis=0)
    return int(indices[int(np.argmax(scores))])


def build_medoid_bank(source, embeddings):
    sentences = original_sentences(source)
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sentences):
        raise ValueError(f"expected embeddings [{len(sentences)}, D], got {embeddings.shape}")
    first_cluster = next(iter(source["level_1"].values()))
    expected_shape = np.asarray(first_cluster["clip_embedding"]).shape
    if expected_shape != (1, embeddings.shape[1]):
        raise ValueError(f"CLIP output dimension differs from original memory: {expected_shape}")
    result = copy.deepcopy(source)
    for level, clusters in result.items():
        for key, cluster in clusters.items():
            index = medoid_index(cluster["indices"], embeddings)
            cluster["summary"] = sentences[index]
            original_dtype = np.asarray(source[level][key]["clip_embedding"]).dtype
            cluster["clip_embedding"] = embeddings[index:index + 1].astype(original_dtype)
            cluster["medoid_index"] = index
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="hierarchical_clustering_results_yc2_70B.pkl")
    parser.add_argument("--output", default="hierarchical_clustering_results_yc2_medoid.pkl")
    parser.add_argument("--embeddings", help="Existing CLIP text embeddings in FINCH input order, [N,768] .npy")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    with open(args.source, "rb") as file:
        source = pickle.load(file)
    sentences = original_sentences(source)
    if args.embeddings:
        embeddings = np.load(args.embeddings)
    else:
        embeddings = encode_clip(sentences, batch_size=args.batch_size, device=args.device)
    result = build_medoid_bank(source, embeddings)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {output}: {sum(map(len, result.values()))} clusters, {len(sentences)} original sentences")


if __name__ == "__main__":
    main()
