#!/usr/bin/env bash
set -euo pipefail

checkpoint=./presave/yc2/best_model.pth
random_bank=./hierarchical_clustering_results_yc2_random_seed42.pkl

for path in "$checkpoint" ./hierarchical_clustering_results_yc2_70B.pkl \
  ./data/yc2/clipvitl14.pth ./data/yc2/val.json \
  ./presave/yc2_llm_eval/youcooksummary.json \
  ./presave/yc2_medoid_eval/youcooksummary.json; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if [[ ! -f "$random_bank" ]]; then
  python random_memory.py --source hierarchical_clustering_results_yc2_70B.pkl \
    --output "$random_bank" --seed 42 --batch-size 128
fi

common=(
  --bank_type yc2 --window_size 10 --sim_match anchor_cos --sampling origin
  --load "$checkpoint" --epochs 20 --lr 3e-4
  --combine_datasets youcook --combine_datasets_val youcook
  --batch_size 2 --batch_size_val 2 --schedule cosine_with_warmup
  --hier_ret_num top-k --soft_k 10 --LLM_ver 70
  --hier_use level_4 level_3 level_2 level_1 --eval
)

python -m torch.distributed.run --nproc_per_node=1 dvc_ret.py \
  "${common[@]}" --ret_option hier_concat --ret_path "$random_bank" \
  --save_dir yc2_random_eval

python -m torch.distributed.run --nproc_per_node=1 dvc_ret.py \
  "${common[@]}" --ret_option no_ret --save_dir yc2_no_memory_eval

python summarize_yc2_ablation.py
