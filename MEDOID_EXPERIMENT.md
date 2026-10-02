# HiCM²: LLaMA memory versus medoid memory

The [upstream HiCM² repository](https://github.com/ailab-kyunghee/HiCM2-DVC) includes the YouCook2 FINCH hierarchy with LLaMA summaries in `hierarchical_clustering_results_yc2_70B.pkl`. This experiment reuses those exact FINCH clusters and parent links. For every cluster, `medoid_memory.py` chooses the original training sentence with the largest mean cosine similarity to other sentences in that cluster, measured in CLIP ViT-L/14 text space. Its CLIP embedding replaces the LLaMA-summary embedding. The output has the same `level_* / cluster_* / clip_embedding` interface, `[1,768]` embedding shape, and `float16` storage dtype. It does not invoke LLaMA or train the caption model.

## Build the medoid bank

Install the upstream environment or a compatible environment with `numpy`, `torch`, and [OpenAI CLIP](https://github.com/openai/CLIP). From the repository root:

If `clip.load` is missing on molab, the environment is importing another package named `clip`. The upstream OpenAI CLIP `setup.py` uses `pkg_resources`, which was removed from setuptools 82. On molab, install the pinned revision **into the same uv environment used by the notebook** and disable build isolation:

```bash
uv pip install --python /tmp/uv-venv/bin/python 'setuptools<82' wheel
uv pip install --python /tmp/uv-venv/bin/python --no-build-isolation --reinstall --no-deps 'git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1'
/tmp/uv-venv/bin/python -c 'import clip; print(clip.__file__, clip.load)'
```

Use `/tmp/uv-venv/bin/python` for the bank command too, so installation and execution use the same environment. The pip cache ownership warning is unrelated to the build failure. Then build the bank:

```bash
/tmp/uv-venv/bin/python medoid_memory.py \
  --source hierarchical_clustering_results_yc2_70B.pkl \
  --output hierarchical_clustering_results_yc2_medoid.pkl \
  --batch-size 128
```

The script encodes 10,337 original YC2 training sentences using CLIP ViT-L/14. A GPU is recommended; reduce `--batch-size` if necessary. If you already have CLIP embeddings **in the original FINCH input order**, pass `--embeddings path/to/embeddings.npy` with shape `[10337,768]` to skip encoding. The generated bank is ignored by Git.

## Paired inference with one checkpoint

The public [YC2 dataset files](https://huggingface.co/datasets/Geppa/HiCM2/tree/main/data/yc2) and [YC2 checkpoint](https://huggingface.co/Geppa/HiCM2/tree/main/presave/yc2) total about 4.4 GB. Install `huggingface_hub`, then download the exact files needed for inference:

```bash
uv pip install --python /tmp/uv-venv/bin/python 'transformers==4.57.6'
python download_yc2_assets.py
bash eval_yc2_memory_ablation.sh
```

Molab's newer Transformers 5 removes `find_pruneable_heads_and_indices`, which the upstream custom T5 imports. Transformers 4.57.6 still exports that helper and has Python 3.13-compatible dependencies. Use molab's `/tmp/uv-venv` rather than the system pip environment.

The branch also makes the custom T5 explicitly inherit `GenerationMixin`, as required by Transformers 4.57, and uses uncached decoding because its attention code expects legacy tuple caches. This is slower but applies equally to both memory arms. The evaluation script runs a tiny generation smoke test before loading the full checkpoint. The public checkpoint contains Python objects beyond tensors, so the code uses `weights_only=False`; load it only from the trusted release.

Both runs load `presave/yc2/best_model.pth`, use the same validation dataset, seed and evaluation arguments, and differ only in `--ret_path`. Metrics and predictions are written separately under `presave/yc2_llm_eval/` and `presave/yc2_medoid_eval/`; the shell script prints side-by-side metric deltas. `--eval` performs inference only. The repository's `requirements.txt` pins Python 3.7-era packages and PyTorch 1.13; molab's current Python/GPU stack may require compatibility work before full inference runs. The memory construction step can be run independently.

The upstream retrieval code computed child clusters but failed to assign them as the current clusters. This branch fixes that step in `model/HiCM2.py`. **Both** LLaMA and medoid runs use the fix and the same released checkpoint. As a result, the paired comparison is fair, but the LLaMA arm may differ from metrics obtained with the unmodified upstream code. If medoid inference is promising, fine-tune with the new bank using the same training schedule as the LLaMA bank for the final comparison.

## Verify that medoid changes retrieval

Close evaluation metrics do not show whether the model consumed the replacement memory. After both inference runs, run:

```bash
python audit_yc2_retrieval.py --videos 8
```

This audit loads the same validation frame features and trained retrieval projection. It reports how many bank embeddings differ, how many sampled windows change the **set** of selected FINCH clusters versus only their ranking, cosine/L2 differences of the retrieved vectors before and after projection, and how many saved video predictions differ. It calls the model's own `hierarchical_memory_search` method; it does not regenerate captions. `projected_changed_windows > 0` demonstrates that different memory tokens enter the T5 encoder. A high cosine after projection or few changed predictions can explain near-identical aggregate metrics without implying the medoid bank was ignored. At YC2's top level there are only eight clusters and `soft_k=10`, so the selected set there cannot change; only its order can.

To determine whether this checkpoint uses memory **content** at all, run `bash eval_yc2_zero_memory.sh`. It keeps the same hierarchy, retrieval code, projection weights and checkpoint, but sets every bank embedding to zero. If quality drops substantially, memory matters and medoid preserves useful information. If quality remains close, the current model relies mainly on the video/subtitle path or the constant projection output; then a fair fine-tuning comparison is needed before making a claim about medoid quality. This control does not measure memory construction cost.

## Random and No-memory controls

The Random and No-memory evaluations can run before or after the LLM and medoid evaluations:

```bash
git pull origin hicm2-medoid
bash eval_yc2_controls.sh
```

The script builds `hierarchical_clustering_results_yc2_random_seed42.pkl` if needed. For each FINCH cluster, it samples one original sentence uniformly with NumPy seed 42, encodes the chosen sentences with the same CLIP ViT-L/14 text model, and preserves the original bank hierarchy and embedding dtype. This tests whether medoid selection improves on an arbitrary member of the same cluster. For a stronger estimate, repeat the Random run with several seeds and report mean and variation; seed 42 is one reproducible draw.

The No-memory run passes `--ret_option no_ret`, which leaves the memory bank unloaded and omits retrieved tokens from the T5 encoder input. It still loads the same released checkpoint; the unused retrieval projection weights appear as unexpected checkpoint keys because that module is absent in this arm. This differs from the zero-memory control, which retains the retrieval path and its projection but feeds zero embeddings. No-memory is an inference ablation of a model trained with memory, so a large drop measures reliance on the trained memory path, while a small drop limits claims about its necessity. It is not a separately trained no-memory baseline.

The script evaluates Random and No-memory on the same YC2 validation data with the same checkpoint and decoding arguments as the paired script. The new outputs are under `presave/yc2_random_eval/` and `presave/yc2_no_memory_eval/`. If the LLM and medoid results already exist, it prints CIDEr, METEOR, F1, threshold F1, and SODA for all four arms. Otherwise, after those runs finish, use `python summarize_yc2_ablation.py`. Full four-arm evaluation needs the Molab GPU and has not been run locally.
