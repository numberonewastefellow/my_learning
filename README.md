# EfficientNetV2 — A Hands-On Computer Vision Learning Path

A self-contained, notebook-driven course that takes you from
**"what is an image to a computer?"** all the way to **training, interpreting,
extending and deploying EfficientNetV2 models**.

Every notebook runs **out-of-the-box** in two environments:

1. **Google Colab** (free T4 GPU) — click the badge next to any notebook.
2. **Locally** on Windows/Linux/macOS with a conda env.

No copy-paste between notebooks — shared code lives in [`utils/`](utils/) and
is imported by every notebook.

---

## Prerequisites

- Basic Python (variables, functions, classes, list/dict comprehensions)
- Curiosity and a willingness to read short markdown explanations between code cells

Not required: prior PyTorch experience, deep-learning theory, or CV background.
We start at absolute fundamentals.

---

## Quickstart — Google Colab (no local install)

This repo lives at [`github.com/numberonewastefellow/my_learning`](https://github.com/numberonewastefellow/my_learning).

1. Open any notebook directly in Colab via:
   `https://colab.research.google.com/github/numberonewastefellow/my_learning/blob/main/notebooks/01_images_and_tensors.ipynb`
   (swap the notebook filename for any other).
2. Runtime → **Change runtime type** → **T4 GPU** → Save.
3. **Run the first cell** — it clones the repo, installs deps, verifies CUDA. Done.

See [`colab_setup.md`](colab_setup.md) for troubleshooting.

## Quickstart — Local (Windows, recommended with conda)

```bash
# 1. Create a fresh env
conda create -n effnetv2 python=3.11 -y
conda activate effnetv2

# 2. Install torch + torchvision with CUDA 12.1 wheels (works with any 12.x driver)
pip install -r requirements-torch-cuda.txt

# 3. Install everything else
pip install -r requirements.txt

# 4. Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 5. Launch Jupyter
jupyter lab
```

See [`setup_env.md`](setup_env.md) for a step-by-step walkthrough, including
how to handle CPU-only machines.

---

## The 13-Notebook Curriculum

| # | Notebook | Phase | You'll learn |
|---|---|---|---|
| 01 | [`images_and_tensors`](notebooks/01_images_and_tensors.ipynb) | A. Fundamentals | Pixels, channels, tensors, normalization |
| 02 | [`pytorch_cv_fundamentals`](notebooks/02_pytorch_cv_fundamentals.ipynb) | A. Fundamentals | `Dataset`, `DataLoader`, a tiny CNN from scratch |
| 03 | [`training_loop_and_metrics`](notebooks/03_training_loop_and_metrics.ipynb) | A. Fundamentals | Training loop, metrics, loss/accuracy curves |
| 04 | [`transfer_learning_efficientnetv2`](notebooks/04_transfer_learning_efficientnetv2.ipynb) | B. EfficientNetV2 | Pretrained models, `timm` vs `torchvision`, frozen-head fine-tune |
| 05 | [`finetuning_strategies`](notebooks/05_finetuning_strategies.ipynb) | B. EfficientNetV2 | Discriminative LR, LR schedulers, LR finder, progressive unfreeze |
| 06 | [`custom_multi_and_attention_heads`](notebooks/06_custom_multi_and_attention_heads.ipynb) | B. EfficientNetV2 | **Custom heads, multi-task heads, attention heads, fwd/bwd pass inspection** |
| 07 | [`efficientnetv2_architecture`](notebooks/07_efficientnetv2_architecture.ipynb) | B. EfficientNetV2 | MBConv, Fused-MBConv, SE, compound scaling, stochastic depth |
| 08 | [`augmentation_deep_dive`](notebooks/08_augmentation_deep_dive.ipynb) | C. Advanced | Every augmentation visualised; Mixup/CutMix; label smoothing |
| 09 | [`class_imbalance_and_metrics`](notebooks/09_class_imbalance_and_metrics.ipynb) | C. Advanced | Weighted sampling, focal loss, ROC/PR, threshold tuning |
| 10 | [`interpretability_gradcam`](notebooks/10_interpretability_gradcam.ipynb) | C. Advanced | Grad-CAM / Grad-CAM++ / ScoreCAM, failure-mode debugging |
| 11 | [`regularization_and_inference`](notebooks/11_regularization_and_inference.ipynb) | C. Advanced | EMA, SWA, TTA, AMP, `torch.compile`, ONNX, INT8 quantization |
| 12 | [`capstone_oxford_flowers`](notebooks/12_capstone_oxford_flowers.ipynb) | D. Capstone | Apply everything on Oxford Flowers-102 |
| 13 | [`capstone_script_classifier`](notebooks/13_capstone_script_classifier.ipynb) | D. Capstone | Binary English-vs-Arabic/Mixed image classifier (real-world) |

### Notebook structure (same template everywhere)

1. **Universal setup cell** (detects Colab, installs deps, imports `utils`)
2. **Learning goals** (3–5 bullets)
3. **Concept explanation** in markdown
4. **Code cells** with inline comments and printed shapes
5. **Visualizations** — curves, confusion matrices, Grad-CAM, grad-norm bars, …
6. **Metrics table**
7. **Key takeaways**
8. **Exercises** (2–3 self-practice prompts)

---

## Shared Utilities ([`utils/`](utils/))

| File | Created in | Purpose |
|---|---|---|
| `env.py` | Batch 0 | `bootstrap()` — env detect, torch check, seed RNG |
| `plotting.py` | Notebook 03 | Curves, confusion matrix, ROC/PR, grad-norm bars |
| `metrics.py` | Notebook 03 | `torchmetrics` wrappers for classification |
| `training.py` | Notebook 03 | `train_one_epoch`, `evaluate`, `fit` with early-stop |
| `heads.py` | Notebook 06 | Custom / multi-task / attention head classes |
| `gradcam.py` | Notebook 10 | Grad-CAM wrapper for EfficientNetV2 |

---

## Datasets

| Notebook(s) | Dataset | Source | Size |
|---|---|---|---|
| 01 | Sample images + CIFAR-10 preview | `torchvision.datasets.CIFAR10` | 170 MB |
| 02 | Fashion-MNIST | `torchvision.datasets.FashionMNIST` | 30 MB |
| 03, 08, 09 | CIFAR-10 (± synthetic imbalance) | `torchvision.datasets.CIFAR10` | 170 MB |
| 04, 05, 06 | Oxford-IIIT Pet | `torchvision.datasets.OxfordIIITPet` | 800 MB |
| 12 | Oxford Flowers-102 | `torchvision.datasets.Flowers102` | 330 MB |
| 13 | User-provided + `sample_data/` | committed | <10 MB |

All `torchvision` datasets auto-download to `data/` (gitignored). Everything
works identically in Colab and locally.

---

## Recommended Study Pace

| Week | Notebooks | Focus |
|---|---|---|
| 1 | 01, 02 | Comfortable with tensors + `DataLoader` |
| 2 | 03 | Own the training loop end-to-end |
| 3 | 04, 05 | Transfer learning & fine-tuning intuition |
| 4 | 06 | Custom / multi / attention heads — the big "how models are built" notebook |
| 5 | 07 | Architecture deep dive |
| 6 | 08, 09 | Augmentation + imbalance |
| 7 | 10, 11 | Interpretability + deployment |
| 8 | 12 | Flowers-102 capstone |
| 9 | 13 | Ship the script classifier |

Total: ~2 months part-time. Much faster if you're already comfortable with PyTorch.

---

## Extensions / Next Steps (not in scope, but worth knowing)

- **ViT comparison** — train ViT-B/16 on Oxford Pet, compare with notebook 05
- **Object detection** — EfficientDet (uses EfficientNet as backbone) or YOLOv8
- **Segmentation** — EfficientNet encoder in U-Net via `segmentation_models_pytorch`
- **Self-supervised** — DINOv2 / MAE pretraining, then fine-tune
- **Multimodal** — CLIP encoder + this classifier head

---

## Troubleshooting

**`torch.cuda.is_available() == False` locally**
→ You likely have the CPU wheel. Run `pip uninstall torch torchvision -y` then
`pip install -r requirements-torch-cuda.txt` again. See [`setup_env.md`](setup_env.md).

**Colab setup cell errors on first run**
→ Check that `utils/env.py`'s `GITHUB_REPO` matches your fork (if you forked
instead of using `numberonewastefellow/my_learning` directly).
For private repos, use `https://<TOKEN>@github.com/<USER>/<REPO>.git`
in the clone URL.

**Out of memory on RTX 3060 (6 GB)**
→ Reduce batch size. EfficientNetV2-S at 300×300 fits batch 16 with AMP.
Use `torch.cuda.amp.autocast` (covered in notebook 11).

**Oxford Pet download fails**
→ Torchvision mirror has occasional outages. Retry, or follow the notebook's
manual download instructions.

---

## Credits & References

- **EfficientNetV2** — Tan & Le, *"EfficientNetV2: Smaller Models and Faster Training"*, ICML 2021. [[arXiv]](https://arxiv.org/abs/2104.00298)
- **`timm`** — Ross Wightman, *PyTorch Image Models*. [[GitHub]](https://github.com/huggingface/pytorch-image-models)
- **`pytorch-grad-cam`** — Jacob Gildenblat. [[GitHub]](https://github.com/jacobgil/pytorch-grad-cam)
- **PyTorch** docs — [pytorch.org/docs](https://pytorch.org/docs/)
- **TorchVision** docs — [pytorch.org/vision](https://pytorch.org/vision/stable/)

---

*This README is the single source of truth for the learning path. Bookmark it,
clone the repo, and return to it any time you want to resume.*
