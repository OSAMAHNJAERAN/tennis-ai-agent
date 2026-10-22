# Reviewed static-court development training

The `--development` option in `scripts/train/train_yolo26_tennis.py` accepts only the exact reviewed court export with manifest SHA256 `0fadd340acc2fdfa5db9083474f620b396eea1d1eaf8fe025fdf050e43c3c90f`. It is separate from the existing four-epoch `--pilot --smoke` route and the frozen independent-test qualification route. It never supplies a test split or promotes a checkpoint.

The export contains 100 training and 20 development images, each with one class 0 court and fourteen semantic slots: 1,657 visible and 23 unknown/null. Its frozen decisions, original publisher image bytes, native review boards, manual observations, root approval, transitive video/event context assignment, label bytes and post-export verification are rechecked before use. Original FPS, PTS, match identity and sessions remain unknown. Context strings stored in `session_id` are conservative grouping keys and are explicitly not verified session or match identities. The author release permits private local academic development; raw broadcast redistribution remains uncleared.

Run a data-only preflight in a fresh directory:

```powershell
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' -m scripts.train.train_yolo26_tennis --role court --development --output outputs/yolo26_court_reviewed_data_preflight_v1
```

This hashes data, writes train/DEV lists and a dataset YAML, and records installed package versions through package metadata. It does not import Torch, Ultralytics or psutil, construct a model, initialize CUDA or prove current resource availability. The metadata marks `runtime_probe_performed: false` and `resource_guard_pending: true`. Reusing its output for execution is refused to preserve that preflight receipt.

After the controlling run has released GPU execution, the initial command is:

```powershell
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' -m scripts.train.train_yolo26_tennis --role court --development --weights models/yolo26/yolo26s-pose.pt --epochs 20 --patience 20 --output outputs/yolo26_court_reviewed_development_run1 --execute
```

The official source checkpoint is hash-pinned (`a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646`). The existing model builder creates a separate YOLO26s pose architecture with one class and a `[14, 3]` head, and records transferred versus initialized tensor keys. It does not reuse the old six-image court specialist.

The protocol fixes 1280 pixels, batch 1, workers 0, no cache, AMP disabled, AdamW, `nbs=1`, zero warmup and learning rate 0.0005. It disables flipping, rotation, perspective, translation, shear, mosaic, mixup, cutmix, copy-paste, erasing, BGR channel swaps, multiscale/rectangular training and automatic augmentation. It explicitly passes `augmentations=[]`: the installed detection/pose factory otherwise enables its default Albumentations list even when the classification-only `auto_augment` field is null. Scale is 0.05 with conservative HSV changes (0.01/0.15/0.15). Early-stop patience may be 15–20. Initial runs may request 1–20 epochs and default to 20; the ceiling after reviewed extensions is 100 total epochs. The optimizer-state receipts record measured update counts; no update count is assumed from the epoch budget.

Execution probes current RAM after CUDA initialization and refuses to construct a model below 1.5 GiB available RAM, 2 GiB free GPU memory or 1 GiB free disk. Existing resource floors for the other modes are unchanged. These are launch floors, not a guarantee that the workload fits. This trainer guard is checked at launch only; runtime resource sampling and any operator stop belong to the controlling launcher, which must retain their actual receipts separately.

Every development run, including resume, requires a fresh empty output directory. The trainer records script, dataset, argument, weight and checkpoint hashes. It retains periodic optimizer checkpoints using the pinned runtime's `save_period=5` convention (zero-based epochs), plus `development_epoch20.pt` at the initial final epoch before standard `last.pt`/`best.pt` stripping. An interrupted optimizer-bearing checkpoint can resume with the same total epoch budget. Stripped checkpoints cannot resume optimizer state.

The saved state follows the actual Ultralytics 8.4.36 implementation inspected locally in `engine/trainer.py`: `save_model()` stores the zero-based epoch, best fitness, half-precision EMA model, EMA update count, optimizer state converted to FP16, scaler state, training arguments, metrics and CSV history. `_load_checkpoint_state()` restores the optimizer/scaler/EMA/best-fitness fields; `resume_training()` restores the next epoch. The scheduler is reconstructed from arguments and its `last_epoch` is set from that next epoch; its state dictionary is not saved. Python/NumPy/Torch/CUDA RNG states, data-loader cursor and early-stopping history are not checkpointed; trainer initialization seeds RNGs again and constructs a new early stopper. Thus this is an optimizer-bearing epoch continuation with the pinned runtime's precision and restart behavior, not a bitwise continuation of every stochastic or scheduler state. Extension review must account for that limitation.

Extending beyond epoch 20 requires `--resume`, an explicitly requested total of at most 100 and `--development-extension-evidence <review.json>`. The evidence is produced only after root review of actual retained development metrics and learning behavior. The trainer validates the following structure; this is a schema description, not an approval receipt:

- `schema`: `tennis.yolo26.court-development-extension.v1`;
- `status`: `ROOT_REVIEWED_USEFUL_LEARNING_EXTENSION_APPROVED`, with `reviewer_type` equal to `ROOT_AGENT_DEVELOPMENT_METRICS_REVIEW`;
- concrete `review_note` and `observed_useful_learning` text, and false `qualification_evidence`/`final_test_used`;
- `requested_total_epochs` matching the command;
- `prior_run_metadata`, `checkpoint` and `metrics_csv` objects containing exact absolute `path` and `sha256` values;
- at least two ordered `observed_epoch_ids` present in the actual run CSV, ending at epoch 20 or later and before the requested total;
- `checkpoint_epoch` equal to the final reviewed epoch; the actual loaded optimizer checkpoint must agree;
- `improved_loss_columns` naming retained train/DEV loss columns with finite nonnegative values and an observed decrease across the reviewed epochs.

The CSV must be the original checkpoint run's `results.csv`. A boolean claiming useful learning cannot satisfy these checks. A decrease only permits a reviewed development extension; it does not establish court accuracy, an independent validation result, calibrated geometry or production qualification. The deployment qualification gate and temporal/video provenance requirements remain unchanged.
