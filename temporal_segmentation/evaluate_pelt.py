"""Evaluate PELT proposals against YouCook2 annotations without training DIBS."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from temporal_segmentation.metrics import summarize
from temporal_segmentation.pelt import segment


def evaluate(annotation, feature_dir, penalty=10.0, limit=None):
    annotations = json.loads(Path(annotation).read_text(encoding="utf-8"))
    predictions, references = [], []
    missing = []
    started = time.perf_counter()
    for key, item in annotations.items():
        # Match data/video_dataset.py's YouCook2 UniVL filename convention.
        path = Path(feature_dir) / (key[2:13] + ".npy")
        if not path.is_file():
            missing.append(key)
            continue
        x = np.load(path)
        duration = float(item["duration"])
        if x.ndim != 2 or len(x) == 0 or duration <= 0:
            raise ValueError(f"invalid feature or duration for {key}")
        proposals = [(start * duration / len(x), end * duration / len(x))
                     for start, end in segment(x, penalty=penalty)]
        predictions.append(proposals)
        references.append(item["timestamps"])
        if limit and len(predictions) >= limit:
            break
    if not predictions:
        raise FileNotFoundError(f"no matching .npy features in {feature_dir}; first missing ID: {missing[:1]}")
    return {"videos": len(predictions), "missing_videos": len(missing),
            "preprocessing_seconds": time.perf_counter() - started,
            "metrics": summarize(predictions, references)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/yc2/captiondata/yc2_val.json")
    parser.add_argument("--features", default="data/features/yc2/UniVL_features/UniVL_visual")
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="results/yc2_pelt.json")
    args = parser.parse_args()
    result = evaluate(args.annotations, args.features, args.penalty, args.limit)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
