# Repository Audit — T88J709 Tennis Vision System

## Master-prompt execution audit — 9 October 2026

The active scope is [the YOLO26 master prompt](../plans/TENNIS_YOLO26_MASTER_PROMPT_EN.md). It supersedes historical model-family comparison objectives and includes scoring, statistical prediction and genuine live video. The single production candidate is `configs/production/yolo26.yaml`, served by `Phase6Pipeline` and the separate FastAPI worker. It uses YOLO26m ball detection, YOLO26s human pose supplying boxes/keypoints, and a separate YOLO26s court pose head with fourteen landmarks. It remains unqualified; configuration is not proof of a model upgrade.

The API environment default now selects that candidate and requires accounts. Historical explicit configurations and saved job config paths remain recoverable. Missing selected assets or qualification evidence fail before model construction; there is no automatic fallback to the historical bundle. Private offline pilot evaluation has a separate explicit constructor/CLI and cannot activate through the web worker.

The legacy bundle remains recoverable. No dataset, saved analysis or old checkpoint was deleted. An audit-time source ZIP, 891-file hash inventory, starting git status and head `f854804f4f0d3766e70218d59f06cd6ebb8aaff1` reside in `artifacts/validation/yolo26_execution_2026-10-08/baseline/`. Parallel work had begun before capture, so this is an audit-time snapshot, not a pristine pre-edit checkout. Branch `codex/yolo26-platform` preserves existing dirty work without committing it.

The ML runtime is Python 3.13, PyTorch 2.13.0+cu126 and Ultralytics 8.4.36. GPU: RTX 4050 Laptop, 6,141 MiB total VRAM; a root query observed 1,919 MiB free, so runs must check current resources. React/TypeScript/Vite, locked pnpm dependencies, Node, FFmpeg and ffprobe are present. The working environment is preserved. Installed APIs and official documentation both require inspection because their current versions differ.

| Required feature | What works | Operating evidence | Gap | Implementation | Verification method |
|---|---|---|---|---|---|
| Upload → saved analysis → replay/report/exports | FastAPI/SQLite jobs, separate worker, source timing and generation-based publication | Prior real 30-frame worker smoke and three public examples | New revision integration and vision accuracy pending | Retain workflow and add strict YOLO26 path | Actual upload, video decode, report/data/revision agreement |
| Resumable multiple files | Byte-offset sessions, batches and per-file UI | Two actual HTTP and in-app browser uploads; exact server hashes, resume, cancellation/retry and owner isolation | Long-file capacity and match continuity unproven | Bounded 8 MiB chunks, ordering, retry/cancel and independent errors | Restart, offset conflict, one-file failure, duplicate clips |
| Accounts and isolation | New opaque cookie sessions, salted hashes and owner columns | New endpoint tests; no public deployment | Quotas/retention/deployment hardening incomplete | Owner checks on every private endpoint and separate curated reads | Two accounts attacking all IDs, media/downloads and mutations |
| Exactly three YOLO26 models | Adapters, shared human pose and pinned BoT-SORT without ReID; specialist learning pilots have run | Constructor/schema/coordinate tests, official-weight smoke and retained training receipts | Useful qualified specialist weights and independent tennis accuracy missing | Authorized learning tests and controlled training, hashes/provenance | Role metrics and no fourth model or silent fallback |
| Data and splits | Existing registry/development data; internet acquisition now authorized | Earlier audit excludes reused holdouts | New independent qualification population pending | Publisher research, separate data/weight/example rights, grouped splits | Checksums, duplicates/leakage, independent review |
| Player identity/movement | Separate reviewed A/B identity layer, segment-local near/far proposals and YOLO26 ankle estimates; unqualified movement withheld | 98 focused candidate camera/identity integration tests | Actual stable identity and physical accuracy unqualified | Source-timed corrections, track reset/redetection and ground-contact quality | End changes/occlusions/officials, HOTA/IDF1, reference ground error |
| Court/timing | RANSAC, nullable projections, interval/camera gates, current-observation reanchors and original PTS audits | Candidate integration plus calibration/FFmpeg tests | Actual cut/replay/view classification, independent meter accuracy and distortion pending | Fourteen landmarks, independent withheld checks, recalibration | VFR/rotation/cuts/zoom, error by court region and valid-time coverage |
| Ball/events/zone fill | Candidate association, diagnostic contacts and full-polygon video/mini-court fill with shared controls | Revision/calibration/event binding and geometry/API/frontend tests; automatic authority withheld | Independent event/contact accuracy pending | Native tiles, bounded gaps, confirmed ground contacts only | Native pixel errors, dead-ball negatives, ±80 ms matching, resize/seek polygons |
| Score and review | Manual desk, corrected rules and new point ledger | Current rule test receipts | Video outcomes/unknown initial state not qualified | Freeze after gaps, stable IDs and correction replay | All formats, service order, complete match drift and duplication |
| Win prediction | Causal beta prior, game/set/match engine and labeled probability/history UI; development-only logistic calibration | Rule tests and actual 114-match publisher score-snapshot evaluation with 19 held-out test matches | Independent human outcomes, broader sample and automatic video-score qualification pending | Explicit prior/unqualified output, source-state-only features and revision replay | Chronological grouped Brier/log loss/reliability versus score-only baseline |
| Physical speed/tactical reports | Proxy player estimates; software calibrated-camera flight reconstruction/evaluator | Software math/report tests; successful flight solutions remain estimated/unqualified | Synchronized references and independent physical accuracy missing | Defined measurement types and physical reference protocol | Same-definition reference errors, bias/P95 and stroke-type coverage |
| Long and live video | Some intermediates disk-backed; recorded worker | Storage tests and short smoke | Remaining O(N) arrays; no full match/live qualification | Bounded chunks/checkpoints; later timed ingest and updates | 10/30 minutes, real 2–3 hours, crash resume, sustained live latency/backlog |
| Deployment and operations | Loopback guide/restoration manifest | Historical runtime receipts | Public license route, capacity and final gates unproven | Recovery path, private storage, queues/quotas and qualified exports | Fresh install and full master acceptance audit |

Software tests cannot qualify vision, physical speed or live operation. Root's first pytest attempt failed sandbox temporary fixtures during concurrent edits; it is not baseline evidence. The first Vitest attempt ran no tests because a transform-cache rename failed. Subsequent workspace-temp or narrowly approved local-socket runs must be recorded separately. The full master goal remains active and incomplete.

See [current execution evidence and training outcomes](../plans/YOLO26_EXECUTION_PROGRESS.md). Ball/court learning pilots have executed; the larger native-crop ball pilot has weak development recall, and its full-frame diagnostic matched zero targets under the frozen criterion. Training completion is not promotion.

> **6 September 2026 update:** This document records the original greenfield audit. The repository now contains the Python inference pipeline and a React application. Read the [Astra upgrade audit and architecture](../plans/ASTRA_3D_UPGRADE.md) and [implemented modules and verification](../ASTRA_TENNIS.md) for the current visualization work. The historical inventory below is retained for provenance.

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
