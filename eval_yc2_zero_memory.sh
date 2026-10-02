#!/usr/bin/env bash
set -euo pipefail

python make_zero_memory.py
python -m torch.distributed.run --nproc_per_node=1 dvc_ret.py \
  --bank_type yc2 --window_size 10 --sim_match anchor_cos --sampling origin \
  --load ./presave/yc2/best_model.pth --epochs 20 --lr 3e-4 \
  --combine_datasets youcook --combine_datasets_val youcook \
  --batch_size 2 --batch_size_val 2 --schedule cosine_with_warmup \
  --ret_option hier_concat --hier_ret_num top-k --soft_k 10 \
  --LLM_ver 70 --hier_use level_4 level_3 level_2 level_1 --eval \
  --ret_path ./hierarchical_clustering_results_yc2_zero.pkl \
  --save_dir yc2_zero_eval

python compare_memory_results.py \
  presave/yc2_llm_eval/youcooksummary.json \
  presave/yc2_zero_eval/youcooksummary.json
