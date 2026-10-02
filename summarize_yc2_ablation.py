"""Print the four-arm YC2 inference comparison from official evaluator output."""

import json
from pathlib import Path


PATHS = {
    "LLM": Path("presave/yc2_llm_eval/youcooksummary.json"),
    "Medoid": Path("presave/yc2_medoid_eval/youcooksummary.json"),
    "Random": Path("presave/yc2_random_eval/youcooksummary.json"),
    "No-memory": Path("presave/yc2_no_memory_eval/youcooksummary.json"),
}
PRIMARY = ("CIDEr", "METEOR", "F1", "F1@0.3", "F1@0.5", "F1@0.7", "F1@0.9", "soda_c", "soda_d")


def main():
    results = {name: json.loads(path.read_text()) for name, path in PATHS.items()}
    print(f"{'metric':14} " + " ".join(f"{name:>12}" for name in PATHS))
    for metric in PRIMARY:
        values = [results[name].get(metric) for name in PATHS]
        if any(not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"missing numeric metric {metric} in one of the result files")
        print(f"{metric:14} " + " ".join(f"{value:12.4f}" for value in values))


if __name__ == "__main__":
    main()
