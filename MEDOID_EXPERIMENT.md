# HiCM²: LLaMA memory versus medoid memory

The [upstream HiCM² repository](https://github.com/ailab-kyunghee/HiCM2-DVC) includes the YouCook2 FINCH hierarchy with LLaMA summaries in `hierarchical_clustering_results_yc2_70B.pkl`. This experiment reuses those exact FINCH clusters and parent links. For every cluster, `medoid_memory.py` chooses the original training sentence with the largest mean cosine similarity to other sentences in that cluster, measured in CLIP ViT-L/14 text space. Its CLIP embedding replaces the LLaMA-summary embedding. The output has the same `level_* / cluster_* / clip_embedding` interface and `[1,768]` embedding shape. It does not invoke LLaMA or train the caption model.

## Build the medoid bank

Install the upstream environment or a compatible environment with `numpy`, `torch`, and [OpenAI CLIP](https://github.com/openai/CLIP). From the repository root:

```bash
python medoid_memory.py \
  --source hierarchical_clustering_results_yc2_70B.pkl \
  --output hierarchical_clustering_results_yc2_medoid.pkl \
  --batch-size 128
```

The script encodes 10,337 original YC2 training sentences using CLIP ViT-L/14. A GPU is recommended; reduce `--batch-size` if necessary. If you already have CLIP embeddings **in the original FINCH input order**, pass `--embeddings path/to/embeddings.npy` with shape `[10337,768]` to skip encoding. The generated bank is ignored by Git.

## Paired inference with one checkpoint

The public [YC2 dataset files](https://huggingface.co/datasets/Geppa/HiCM2/tree/main/data/yc2) and [YC2 checkpoint](https://huggingface.co/Geppa/HiCM2/tree/main/presave/yc2) total about 4.4 GB. Install `huggingface_hub`, then download the exact files needed for inference:

```bash
python download_yc2_assets.py
bash eval_yc2_memory_ablation.sh
```

Both runs load `presave/yc2/best_model.pth`, use the same validation dataset, seed and evaluation arguments, and differ only in `--ret_path`. Metrics and predictions are written separately under `presave/yc2_llm_eval/` and `presave/yc2_medoid_eval/`; the shell script prints side-by-side metric deltas. `--eval` performs inference only. The repository's `requirements.txt` pins Python 3.7-era packages and PyTorch 1.13; molab's current Python/GPU stack may require compatibility work before full inference runs. The memory construction step can be run independently.

The upstream retrieval code computed child clusters but failed to assign them as the current clusters. This branch fixes that step in `model/HiCM2.py`. **Both** LLaMA and medoid runs use the fix and the same released checkpoint. As a result, the paired comparison is fair, but the LLaMA arm may differ from metrics obtained with the unmodified upstream code. If medoid inference is promising, fine-tune with the new bank using the same training schedule as the LLaMA bank for the final comparison.
