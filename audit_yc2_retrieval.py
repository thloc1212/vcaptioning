"""Audit whether YC2 medoid memory changes the actual HiCM2 retrieval path.

Uses the released validation features, the model's retrieval method, and the
checkpoint's trained retrieval projection. It does not run caption generation.
Only load the released pickle/checkpoint files if you trust their sources.
"""

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model.HiCM2 import LinearLayer, Vid2Seq


def load_bank(path, device):
    with Path(path).open("rb") as file:
        bank = pickle.load(file)
    for clusters in bank.values():
        for cluster in clusters.values():
            cluster["clip_embedding"] = torch.as_tensor(cluster["clip_embedding"], device=device)
    return bank


def bank_difference(original, medoid):
    changed = total = 0
    similarities = []
    for level in original:
        if set(original[level]) != set(medoid[level]):
            raise ValueError(f"cluster keys differ at {level}")
        for key, cluster in original[level].items():
            other = medoid[level][key]
            a = cluster["clip_embedding"].float()
            b = other["clip_embedding"].float()
            if a.shape != b.shape:
                raise ValueError(f"embedding shape differs at {level}/{key}")
            changed += not torch.equal(a, b)
            total += 1
            similarities.append(float(F.cosine_similarity(a, b).item()))
    return {"changed_clusters": changed, "total_clusters": total,
            "changed_fraction": changed / total,
            "mean_cluster_cosine": float(np.mean(similarities))}


def load_projection(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    prefix = "ret2t5_proj."
    projection_state = {key[len(prefix):]: value for key, value in state.items()
                        if key.startswith(prefix)}
    if not projection_state:
        raise ValueError("checkpoint has no trained ret2t5_proj weights")
    projection = nn.Sequential(
        LinearLayer(768, 768, layer_norm=True, dropout=0.5, relu=True),
        LinearLayer(768, 768, layer_norm=True, dropout=0.5, relu=False),
    )
    projection.load_state_dict(projection_state, strict=True)
    return projection.to(device).eval()


def sample_video(features, video_id, max_feats=100):
    video = features[video_id[-11:]].float()
    if len(video) > max_feats:
        video = torch.stack([video[(j * len(video)) // max_feats] for j in range(max_feats)])
    elif len(video) < max_feats:
        video = torch.cat([video, torch.zeros(max_feats - len(video), video.shape[1])])
    return video


def compare_predictions(first_path, second_path):
    if not first_path.is_file() or not second_path.is_file():
        return None
    first = json.loads(first_path.read_text())["results"]
    second = json.loads(second_path.read_text())["results"]
    keys = first.keys() & second.keys()
    changed = sum(first[key] != second[key] for key in keys)
    return {"compared_videos": len(keys), "changed_videos": changed,
            "changed_fraction": changed / len(keys) if keys else 0.0}


def audit(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    original = load_bank(args.original, device)
    medoid = load_bank(args.medoid, device)
    bank_stats = bank_difference(original, medoid)
    projection = load_projection(args.checkpoint, device)
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    video_ids = list(json.loads(Path(args.annotations).read_text()).keys())[:args.videos]
    retrieval = SimpleNamespace(args=SimpleNamespace(
        hier_use=["level_4", "level_3", "level_2", "level_1"],
        hier_ret_num="top-k",
    ))
    level_changes = {level: 0 for level in original}
    raw_cosines, projected_cosines = [], []
    raw_l2, projected_l2 = [], []
    windows = 0
    first_changed_example = None
    with torch.no_grad():
        for video_id in video_ids:
            video = sample_video(features, video_id).to(device)
            segment_length = len(video) // args.window_size
            for window in range(args.window_size):
                target = video[window * segment_length:(window + 1) * segment_length].mean(0)
                a, a_trace = Vid2Seq.hierarchical_memory_search(
                    retrieval, target, args.soft_k, original, return_trace=True)
                b, b_trace = Vid2Seq.hierarchical_memory_search(
                    retrieval, target, args.soft_k, medoid, return_trace=True)
                for a_level, b_level in zip(a_trace, b_trace):
                    if a_level["clusters"] != b_level["clusters"]:
                        level_changes[a_level["level"]] += 1
                a_proj = projection(a.float())
                b_proj = projection(b.float())
                raw_cosines.append(float(F.cosine_similarity(a.float(), b.float()).item()))
                projected_cosines.append(float(F.cosine_similarity(a_proj, b_proj).item()))
                raw_distance = float(torch.linalg.vector_norm(a.float() - b.float()).item())
                projected_distance = float(torch.linalg.vector_norm(a_proj - b_proj).item())
                raw_l2.append(raw_distance)
                projected_l2.append(projected_distance)
                if first_changed_example is None and projected_distance > 1e-6:
                    first_changed_example = {
                        "video_id": video_id, "window": window,
                        "llm_selection": a_trace, "medoid_selection": b_trace,
                        "raw_l2": raw_distance, "projected_l2": projected_distance,
                    }
                windows += 1
    if not windows:
        raise ValueError("no validation windows sampled")
    return {
        "bank": bank_stats,
        "retrieval": {
            "videos": len(video_ids), "windows": windows,
            "changed_selection_by_level": level_changes,
            "mean_raw_cosine": float(np.mean(raw_cosines)),
            "mean_raw_l2": float(np.mean(raw_l2)),
            "mean_projected_cosine": float(np.mean(projected_cosines)),
            "mean_projected_l2": float(np.mean(projected_l2)),
            "projected_changed_windows": sum(value > 1e-6 for value in projected_l2),
            "first_changed_example": first_changed_example,
        },
        "predictions": compare_predictions(Path(args.original_predictions), Path(args.medoid_predictions)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="hierarchical_clustering_results_yc2_70B.pkl")
    parser.add_argument("--medoid", default="hierarchical_clustering_results_yc2_medoid.pkl")
    parser.add_argument("--checkpoint", default="presave/yc2/best_model.pth")
    parser.add_argument("--features", default="data/yc2/clipvitl14.pth")
    parser.add_argument("--annotations", default="data/yc2/val.json")
    parser.add_argument("--original-predictions", default="presave/yc2_llm_eval/youcook_test_preds.json")
    parser.add_argument("--medoid-predictions", default="presave/yc2_medoid_eval/youcook_test_preds.json")
    parser.add_argument("--videos", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--soft-k", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.videos <= 0 or args.window_size <= 0 or args.soft_k <= 0:
        parser.error("--videos, --window-size and --soft-k must be positive")
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
