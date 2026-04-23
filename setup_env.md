# Local Environment Setup

Complete walkthrough for running the notebooks on your own machine (Windows,
Linux, or macOS). Colab users — see [`colab_setup.md`](colab_setup.md) instead.

---

## 1. Check your GPU (optional but recommended)

```bash
nvidia-smi
```

You want to see a CUDA-capable GPU and a driver version **≥ 11.8**. Any CUDA
12.x driver works with our wheels (we use cu121 builds which are
forward/backward compatible).

No GPU? Skip to [CPU-only install](#cpu-only-install) below — notebooks 01–03
run fine on CPU; later notebooks work but training is much slower.

---

## 2. Create a fresh conda environment

Why a dedicated env: avoids conflicts with any existing PyTorch / CUDA versions
on your system.

```bash
conda create -n effnetv2 python=3.11 -y
conda activate effnetv2
```

Any time you return to these notebooks: `conda activate effnetv2`.

---

## 3. Install PyTorch with CUDA

```bash
pip install -r requirements-torch-cuda.txt
```

This pulls `torch==2.3.1+cu121` and `torchvision==0.18.1+cu121` — a stable
combination with wide `timm` compatibility.

### Verify CUDA is usable

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output:
```
CUDA: True
Device: NVIDIA GeForce RTX 3060 ...
```

If you see `CUDA: False`: your NVIDIA driver may be too old, or the cu121
wheel is incompatible. Try the matching wheel for your driver:

| Driver | Wheel URL |
|---|---|
| 11.8.x | `https://download.pytorch.org/whl/cu118` |
| 12.1+ | `https://download.pytorch.org/whl/cu121` (default) |

---

## 4. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

Installs: `timm`, `torchmetrics`, `torchinfo`, `albumentations`,
`scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `jupyterlab`, `grad-cam`,
`onnx`, `onnxruntime`, etc.

---

## 5. Launch Jupyter

```bash
jupyter lab
```

A browser tab opens. Navigate to `notebooks/` and open
`01_images_and_tensors.ipynb` to begin.

---

## CPU-only install

If you have no GPU (or want to avoid CUDA):

```bash
conda create -n effnetv2 python=3.11 -y
conda activate effnetv2
pip install torch==2.3.1 torchvision==0.18.1     # CPU wheels
pip install -r requirements.txt
```

Caveats:
- Notebooks 01–03 are fully CPU-friendly.
- Notebooks 04+ will train, but expect **20–50× slower** than a GPU.
- Recommendation: run fundamentals locally on CPU, open notebooks 04+ in Colab
  for the GPU. Works seamlessly either way.

---

## Common pitfalls

**"torch installed but CUDA: False"**
You previously installed the CPU wheel. Fix:
```bash
pip uninstall -y torch torchvision
pip install -r requirements-torch-cuda.txt
```

**"DLL load failed" on Windows when importing torch**
Microsoft Visual C++ Redistributable missing. Install from
[aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
and reboot.

**`jupyter lab` command not found**
The `conda activate effnetv2` step was missed; `jupyterlab` is per-env.

**Out of memory (6 GB GPU) during notebook 12 (Flowers-102)**
- Drop batch size from 32 → 16
- Enable AMP: the notebook shows how
- Or temporarily move to Colab for that one notebook

---

## Updating

When notebooks add new deps (happens when you pull updates), re-run step 4:
```bash
pip install -r requirements.txt --upgrade
```
