# Repository Audit — T88J709 Tennis Vision System

> **Date:** 2026-08-22
> **Auditor:** AI/ML Engineering Agent
> **Repository:** `c:\Semester 9\FYP2\tennis_ai_agent_bundle`

---

## 1. Repository Overview

### Git Status
- **Git initialized:** NO — `fatal: not a git repository`
- **Starting commit SHA:** N/A (no git history)
- **Branch:** N/A
- **Remote:** N/A

### Project State: GREENFIELD
This repository contains **specification documents only**. There is zero implementation code, zero model weights, zero datasets, zero tests, zero configuration files, and zero infrastructure.

---

## 2. Complete File Tree

```
tennis_ai_agent_bundle/
├── AGENT.md                              (35,410 bytes)  Engineering contract
├── MASTER_PROMPT.md                      (21,151 bytes)  Agent instructions
├── PLAN.md                               (17,141 bytes)  15-phase execution plan
├── PROJECT_REPORT.md                    (132,950 bytes)  FYP1 report (Markdown)
├── README_BUNDLE.md                       (1,770 bytes)  Bundle usage guide
├── VIDEO_REFERENCE.md                    (13,021 bytes)  Tutorial extraction
├── report_assets/
│   └── media/
│       ├── image10.png                  (622,744 bytes)
│       ├── image17.emf                  (694,460 bytes)
│       ├── image18.jpeg                  (15,679 bytes)
│       ├── image19.png                  (149,115 bytes)
│       ├── image2.png                     (3,267 bytes)
│       ├── image20.jpg                   (17,204 bytes)
│       ├── image3.png                    (55,056 bytes)
│       ├── image4.png                     (9,341 bytes)
│       ├── image5.jpeg                   (33,254 bytes)
│       ├── image6.png                   (350,462 bytes)
│       └── image8.png                   (621,203 bytes)
└── docs/                                 (created during audit)
    ├── audit/
    ├── data/
    ├── environment/
    ├── experiments/
    └── plans/
```

**Total files:** 17 (6 markdown + 11 images)
**Total size:** ~2.1 MB

---

## 3. Inventory

### 3.1 Languages and Runtimes
| Item | Status |
|------|--------|
| Python source files | None |
| JavaScript/TypeScript | None |
| Notebooks (.ipynb) | None |
| Shell scripts | None |

### 3.2 Dependencies
| Item | Status |
|------|--------|
| requirements.txt | Missing |
| pyproject.toml | Missing |
| setup.py | Missing |
| Pipfile | Missing |
| conda environment | Missing |
| package-lock.json | Missing |

### 3.3 Training Code
| Item | Status |
|------|--------|
| Training scripts | None |
| Training notebooks | None |
| Data loading code | None |
| Augmentation code | None |

### 3.4 Inference Code
| Item | Status |
|------|--------|
| Inference pipeline | None |
| main.py or equivalent | None |
| CLI entry points | None |

### 3.5 Model Assets
| Item | Status |
|------|--------|
| YOLO weights (.pt) | None |
| Court model weights (.pth) | None |
| ONNX models | None |
| Model configs | None |

### 3.6 Data Assets
| Item | Status |
|------|--------|
| Training datasets | None |
| Validation datasets | None |
| Test datasets | None |
| Sample videos | None |
| Data manifests | None |

### 3.7 Tests
| Item | Status |
|------|--------|
| Unit tests | None |
| Integration tests | None |
| Test fixtures | None |
| pytest.ini / conftest.py | None |

### 3.8 UI / Dashboard
| Item | Status |
|------|--------|
| Frontend code | None |
| HTML/CSS/JS | None |
| Streamlit/Gradio | None |
| Figma exports | Referenced in report only |

### 3.9 Database / Storage
| Item | Status |
|------|--------|
| SQLite database | None |
| Migration scripts | None |
| Schema definitions | None |

### 3.10 Configuration
| Item | Status |
|------|--------|
| YAML configs | None |
| JSON configs | None |
| .env file | None |
| .env.example | None |

### 3.11 CI/CD
| Item | Status |
|------|--------|
| GitHub Actions | None |
| Makefile | None |
| Dockerfile | None |

### 3.12 Secrets / Credentials
| Item | Status |
|------|--------|
| API keys in code | None found |
| .env with secrets | None |
| Roboflow API key | Not present (will be needed) |

### 3.13 Cache / Generated Files
| Item | Status |
|------|--------|
| __pycache__ | None |
| .pytest_cache | None |
| runs/ | None |
| wandb/ | None |

---

## 4. Current-State Execution

### 4.1 Installation
- **Result:** No requirements.txt exists. System-wide Python 3.13 has relevant packages installed globally.
- **PyTorch:** CPU-only build (2.11.0+cpu) — CUDA unavailable despite GPU hardware.
- **Ultralytics:** 8.4.36 — supports YOLO11.

### 4.2 Tests
- **Result:** No test suite exists. Nothing to run.

### 4.3 Inference
- **Result:** No inference code exists. Nothing to run.

### 4.4 Missing Assets
- All model weights
- All datasets
- All training code
- All inference code
- Sample video input
- All configuration files

---

## 5. Gap Matrix

See `docs/plans/BASELINE_IMPLEMENTATION_PLAN.md` Section 2.1 for the full 29-requirement gap matrix.

**Summary:** 0 of 29 requirements are currently implemented. All must be built from scratch.

---

## 6. Technical Debt

| Category | Issues |
|----------|--------|
| Missing Git | No version control initialized |
| No dependency management | Everything installed globally; no venv, no lock file |
| CPU-only PyTorch | GPU training impossible without reinstall |
| Low disk space | Only 9.2 GB free |
| No .gitignore | Risk of committing large files |
| No project structure | Flat directory with only spec files |

---

## 7. Reference Implementation Analysis

The tutorial repository (abdullahtarek/tennis_analysis) provides the architectural blueprint:

### Structure
```
analysis/                  -> Exploratory ball analysis notebook
constants/                 -> ITF court dimensions
court_line_detector/       -> ResNet-50 court keypoint model
input_videos/              -> Sample tennis video
mini_court/                -> Mini-court renderer
models/                    -> Model weights (placeholder)
output_videos/             -> Output samples
runs/detect/               -> YOLO run artifacts
tracker_stubs/             -> Cached detection pickles
trackers/                  -> Player and ball tracker classes
training/                  -> Training notebooks + Roboflow dataset
utils/                     -> Video I/O, bbox, conversions, drawing
main.py                    -> Pipeline entry point
yolo_inference.py          -> Standalone YOLO test
```

### Key Weights (Google Drive)
- Ball detector: YOLOv5l6u (yolo5_last.pt)
- Court model: ResNet-50 (keypoints_model.pth)

### Known Tutorial Weaknesses
1. Hardcoded 24 FPS throughout
2. Uses `last.pt` instead of `best.pt`
3. Blind pandas `bfill()` for ball interpolation
4. No homography — uses keypoint-relative pixel scaling
5. No automated tests
6. No structured output
7. Loads entire video into memory
8. No license file
9. Hardcoded file paths
10. Player height assumptions (1.88m, 1.91m)

---

## 8. Conclusion

This is a complete greenfield implementation starting from detailed specifications. The tutorial repository provides a proven architectural reference. The primary blockers are:

1. **PyTorch CUDA installation** (critical for training)
2. **Disk space** (need 15-20 GB more)
3. **Sample video** (need authorized test footage)

All implementation must be built from scratch following the specifications in AGENT.md, PLAN.md, PROJECT_REPORT.md, and VIDEO_REFERENCE.md.
