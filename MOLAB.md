# DIBS on molab: first YouCook2 experiment

This repository is based on the [official DIBS implementation](https://github.com/haowuxc/DIBS). The first experiment evaluates PELT event proposals against the included YouCook2 validation annotations. It does not need raw videos, a DIBS checkpoint, or GPU training.

## Open in molab

Open [molab notebooks](https://molab.marimo.io/notebooks/) and use **Mirror from GitHub** with `https://github.com/thloc1212/vcaptioning`. Open `notebooks/yc2_pelt.py`. In its terminal, change to the repository directory and run:

```bash
python -m pip install -r requirements-segmentation.txt
python scripts/download_yc2_univl.py
python -m temporal_segmentation.evaluate_pelt --penalty 10
```

The notebook's **Evaluate** button runs a smaller interactive sample. The last command above runs the complete validation set.

The official [feature archive](https://huggingface.co/datasets/Exclibur/dibs-feature) is 4 GB compressed. The download script streams the archive and retains only YouCook2 UniVL visual and text `.npy` files under `data/features/`; allow time and storage for this. Downloads made in a molab terminal may need to be repeated in a later session; molab documents [persistent caching and storage limits](https://molab.marimo.io/blog/seamless-storage-in-molab).

The JSON output is written to `results/yc2_pelt.json`. Run with `--limit 20` for a quick smoke test. `--penalty` controls how many segments PELT proposes; choose it on training data before reporting validation results. The metrics use one-to-one interval matching at IoU 0.3, 0.5, 0.7 and 0.9, with micro precision, recall and F1. They are exploratory metrics, not the official DIBS/ActivityNet evaluator. Comparing them to DIBS requires extracting DIBS proposals and running the same matching rule.

## Full DIBS training later

`cfgs/yc2_univl_refine_molab.yml` replaces the original machine-specific feature paths. The upstream training code also requires the packages in `requirement.txt`, a compatible PyTorch/CUDA toolchain, and compilation of `pdvc/ops` using `bash make.sh`. The upstream pinned PyTorch 1.12/CUDA 11.6 stack predates current molab Blackwell GPUs, so validate compatibility before attempting training. The [released checkpoints](https://huggingface.co/Exclibur/DIBS) and ActivityNet/HowTo100M features are deferred until the first experiment is verified.
