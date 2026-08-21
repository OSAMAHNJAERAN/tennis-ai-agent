# Baseline Environment — T88J709 Tennis Vision System

> **Date:** 2026-08-22
> **Machine:** MSI Katana 15 (as described in FYP report)

---

## 1. Operating System

| Field | Value |
|-------|-------|
| OS | Microsoft Windows 11 Home Single Language |
| Version | 10.0.26200 |
| Architecture | 64-bit |

---

## 2. CPU

| Field | Value |
|-------|-------|
| Model | Intel Core i7-14700HX |
| Cores | 20 |
| Logical Processors | 28 |

---

## 3. RAM

| Field | Value |
|-------|-------|
| Total Physical Memory | 15.71 GB |

---

## 4. GPU

| Field | Value |
|-------|-------|
| Model | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6141 MiB (~6 GB) |
| Driver Version | 596.36 |
| CUDA Driver Version | 13.2 |
| Temperature (idle) | 40°C |
| TDP | 77W max |

### GPU Verification (nvidia-smi output captured 2026-08-22 07:22:50)
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 596.36                 Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   40C    P0             11W /   77W |       0MiB /   6141MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

---

## 5. Python Environment

| Field | Value |
|-------|-------|
| Python Version | 3.13.5 |
| Python Location | C:\Users\ac-98\AppData\Local\Programs\Python\Python313 |
| pip Version | 26.1.2 |
| Virtual Environment | None (system-wide install) |

---

## 6. Key Python Packages

| Package | Version | CUDA Support |
|---------|---------|:------------:|
| torch | 2.11.0+cpu | **NO** |
| torchvision | 0.26.0 | **NO** |
| ultralytics | 8.4.36 | Via PyTorch |
| ultralytics-thop | 2.0.18 | — |
| opencv-python | 4.13.0.92 | — |
| opencv-contrib-python | 4.13.0.92 | — |
| opencv-python-headless | 4.10.0.84 | — |
| numpy | 2.3.1 | — |
| pandas | 2.3.1 | — |
| scipy | 1.17.1 | — |
| roboflow | 1.3.1 | — |

---

## 7. CUDA Smoke Test

```python
>>> import torch
>>> torch.__version__
'2.11.0+cpu'
>>> torch.cuda.is_available()
False
```

**RESULT: FAIL** — PyTorch is CPU-only build. CUDA is NOT available to PyTorch despite the GPU hardware supporting it.

### Required Fix
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

After fix, expected result:
```python
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 4050 Laptop GPU'
```

---

## 8. Disk Space

| Drive | Used | Free | Percentage Free |
|-------|------|------|:-:|
| C: | 943.3 GB | 9.2 GB | 0.97% |

**STATUS: CRITICAL** — Less than 10 GB free. Insufficient for datasets, models, and training outputs.

**Required:** Minimum 20-25 GB free for baseline work:
- Datasets: ~1.5 GB
- Model weights (pretrained + trained): ~1 GB
- Training outputs/checkpoints: ~2 GB
- Output videos: ~1 GB
- Working space: ~5 GB

---

## 9. Available Tools

| Tool | Available | Version |
|------|:---------:|---------|
| git | TBD | Need to verify |
| ffmpeg/ffprobe | TBD | Need to verify |
| pytest | TBD | Need to install |

---

## 10. Environment Compatibility Assessment

| Requirement | Status | Notes |
|-------------|:------:|-------|
| Python 3.10+ | PASS | Python 3.13.5 installed |
| PyTorch 2.x | PARTIAL | 2.11.0 installed but CPU-only |
| CUDA GPU support | FAIL | Need CUDA PyTorch |
| Ultralytics YOLO11 | PASS | 8.4.36 supports YOLO11 |
| OpenCV | PASS | 4.13.0.92 |
| Sufficient VRAM (>4GB) | PASS | 6 GB available |
| Sufficient RAM (>8GB) | PASS | 15.71 GB |
| Sufficient disk space | FAIL | Only 9.2 GB free |
| Virtual environment | FAIL | Not set up |
| Dependency lock | FAIL | No requirements.txt |

---

## 11. Environment Recreation Commands

Once blockers are resolved:

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 3. Install project dependencies
pip install ultralytics>=8.4.0 opencv-python pandas numpy scipy roboflow pyyaml pytest

# 4. Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available()); from ultralytics import YOLO; print('YOLO OK')"

# 5. Freeze
pip freeze > requirements.txt
```
