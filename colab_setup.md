# Running the Notebooks in Google Colab

Colab gives you a free T4 GPU (or pay a little for faster hardware). No local
install needed — everything runs in your browser.

---

## One-time prep

1. **Push this repo to your GitHub.**

   ```bash
   cd d:/llm/efficientnetv2_examples
   git init
   git add .
   git commit -m "Initial scaffold"
   git branch -M main
   git remote add origin https://github.com/numberonewastefellow/my_learning.git
   git push -u origin main
   ```

2. The `GITHUB_REPO` constant in [`utils/env.py`](utils/env.py) already points
   at `github.com/numberonewastefellow/my_learning.git` — no edit needed.
   If you fork, update that line to your fork's URL.

3. (Private repo only) Create a GitHub Personal Access Token with `repo` scope
   and use this URL form in `GITHUB_REPO`:
   ```
   https://<TOKEN>@github.com/numberonewastefellow/my_learning.git
   ```
   ⚠️ Never commit a token. If you hard-code one, rotate it immediately.

---

## Running a notebook

### Option A — Direct (GitHub → Colab)

Open:
```
https://colab.research.google.com/github/numberonewastefellow/my_learning/blob/main/notebooks/01_images_and_tensors.ipynb
```
(swap the notebook filename for any other).

### Option B — Open from Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub tab**
3. Paste your repo URL, pick the notebook.

### Option C — Upload

For a quick one-off: download the `.ipynb` file and upload it to Colab directly
(File → Upload notebook).

---

## Enable the GPU

**Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**

Check in the first cell (the universal setup cell handles this automatically,
but you can also run):
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
→ `True  Tesla T4`

---

## Run the universal setup cell

Every notebook starts with:

```python
import sys, os
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    if not os.path.exists("my_learning"):
        !git clone --quiet https://github.com/numberonewastefellow/my_learning.git
    %cd my_learning
    !pip install -q -r requirements.txt

repo_root = os.path.abspath(".") if IN_COLAB else os.path.abspath("..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.env import bootstrap
bootstrap()
```

What this does:
1. Detects Colab.
2. Clones your repo on first run (subsequent runs reuse the existing clone).
3. Installs pip deps quietly.
4. Makes `utils/` importable.
5. `bootstrap()` prints CUDA status, seeds RNGs, sets `device`.

---

## Working with data

### `torchvision` datasets (CIFAR-10, Fashion-MNIST, Oxford Pet, Flowers-102)

Nothing special — they auto-download into `./data/` on first call. The first
run downloads ~1 GB total; subsequent runs are instant (until the runtime is
recycled, usually after 12 h of idle time).

### Your own images (notebook 13 — script classifier)

Pick one:

**(a) Upload a ZIP to the Colab filesystem**
```python
from google.colab import files
uploaded = files.upload()  # opens a picker
!unzip -q my_script_images.zip -d data/script_classifier/
```

**(b) Mount Google Drive**
```python
from google.colab import drive
drive.mount("/content/drive")
# Then read from "/content/drive/MyDrive/my_script_images/"
```

**(c) Commit to your fork of the repo**
Push your labeled images to `data/script_classifier/{ltr,rtl}/` in your fork,
then the universal setup cell's `git clone` pulls them along with the code.
Best for small datasets (<100 MB). Use Git LFS for larger sets (already
configured in `.gitattributes`).

---

## Runtime tips

- **T4 has 16 GB VRAM** — much more than an RTX 3060. You can raise batch sizes.
- **Runtime is ephemeral** — your `data/` and `checkpoints/` vanish when the
  runtime disconnects. For large trainings, save checkpoints to Drive:
  ```python
  torch.save(model.state_dict(), "/content/drive/MyDrive/effnetv2/best.pt")
  ```
- **Colab kicks you after 90 min of inactivity** — keep the tab active during
  long training runs.
- **A100 / V100** are available on Colab Pro for faster training if you upgrade.

---

## Common Colab gotchas

**"fatal: repository not found"**
Your repo is private. Use the PAT URL form in `utils/env.py`, or make the repo
public for ease.

**"ModuleNotFoundError: No module named 'utils'"**
The setup cell wasn't run, or you're outside the repo dir. Re-run the first
cell from top.

**"RuntimeError: CUDA out of memory"**
Drop batch size, or switch to a smaller model variant. Restart the runtime
(Runtime → Restart) to fully clear CUDA state.

**Colab "Runtime disconnected"**
Just reconnect — the clone and installs skip since files are still present.

---

## Back to the learning path

Once setup works: open [`README.md`](README.md) and start with notebook 01.
