#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import transformers
from transformers import pytorch_utils
if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
    raise SystemExit(
        f"Transformers {transformers.__version__} is too new for upstream HiCM2. "
        "On molab run: uv pip install --python /tmp/uv-venv/bin/python 'transformers==4.57.6'"
    )
PY

python smoke_test_generation.py

checkpoint=./presave/yc2/best_model.pth
original=./hierarchical_clustering_results_yc2_70B.pkl
medoid=./hierarchical_clustering_results_yc2_medoid.pkl

for path in "$checkpoint" "$original" "$medoid" ./data/yc2/clipvitl14.pth ./data/yc2/val.json; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

common=(
  --bank_type yc2 --window_size 10 --sim_match anchor_cos --sampling origin
  --load "$checkpoint" --epochs 20 --lr 3e-4
  --combine_datasets youcook --combine_datasets_val youcook
  --batch_size 2 --batch_size_val 2 --schedule cosine_with_warmup
  --ret_option hier_concat --hier_ret_num top-k --soft_k 10
  --LLM_ver 70 --hier_use level_4 level_3 level_2 level_1 --eval
)

for method in llm medoid; do
  if [[ "$method" == llm ]]; then bank="$original"; else bank="$medoid"; fi
  python -m torch.distributed.run --nproc_per_node=1 dvc_ret.py \
    "${common[@]}" --ret_path "$bank" --save_dir "yc2_${method}_eval"
done

python compare_memory_results.py \
  presave/yc2_llm_eval/youcooksummary.json \
  presave/yc2_medoid_eval/youcooksummary.json
