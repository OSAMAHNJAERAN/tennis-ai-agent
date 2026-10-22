# Private integrated pipeline evaluation

The production worker and `build_yolo26_bundle` continue to require independent
tennis qualification. Private engineering evaluation uses an explicit
`Phase6Pipeline(..., evaluation_only=True)` constructor argument and the separate
`build_evaluation_bundle` entrypoint. A YAML flag alone cannot activate this mode.
Asset hashes, model roles, specialist provenance, the pinned runtime and the
three-model limit remain mandatory. This entrypoint never returns a verified
qualification receipt.

The isolated configuration for the previously inspected original match172 is
`configs/evaluation/yolo26.private-pilot-match172.yaml`. Its artifact manifest
pins the completed native-crop ball pilot, the court14 pilot and the official
human pose checkpoint directly at their existing paths. Its source manifest
pins `data/external/racketvision_training20/tennis/videos/match172_000.mp4` with
SHA256 `eb1dd91dbdf4c270f02f3f26d86cd9e7ed81606123f354805263771f6f9a5a1a`.
The record retains unresolved original-match/session independence and private
academic use restrictions. Existing pilot samples classify this source as
development training footage; it cannot become independent final-test evidence.

Run from the repository root, with a new output directory for every attempt:

```powershell
python -m scripts.evaluate.run_yolo26_candidate --config configs/evaluation/yolo26.private-pilot-match172.yaml --source-id match172 --output outputs/private_yolo26_candidate_new_run --device 0 --frames 30 --execute
```

Omit `--execute` for asset/source/resource preflight without model construction.
The entrypoint accepts 1–30 native frames. It checks source and manifest hashes,
then requires at least 1.5 GiB available RAM, 1 GiB free disk and, for device 0,
2 GiB free VRAM. RAM is rechecked after CUDA runtime initialization. A resource
failure is terminal for that output directory and retains actual measurements.
It does not change floors or stop other processes.

FFmpeg trims the native prefix before normalization and retains decoder source
frame index, PTS and time-base tokens in `source_frame_mapping.jsonl`. The run
requires distinct mapped source frames, all three actual model adapters, exact
complete decoding of both bounded source and annotated output, and finite JSON
artifacts. `evaluation_receipt.json` retains checkpoint hashes, inference call
counts/times, combined media/pipeline stage times, RAM/VRAM snapshots, process
peak working set, CUDA allocation peaks and the encoded artifact's hash/frame
count. Annotated encoding time is included in the pipeline stage; an isolated
encoding duration is explicitly unavailable. Failed partial artifacts and logs
remain available for diagnosis.

A completed run produces a local `report.html`, `annotated.mp4`, detections,
camera interval and identity audits. It does not create a web job or publish raw
footage. The empty review/anchor inputs leave normal-speed match activity and
stable A/B identity unconfirmed. Unqualified geometry, physical speeds,
distances, ground positions, events, line calls, score, complete-match capacity
and live latency remain unavailable. Detection confidence, observed landmark
slots and software execution are not accuracy measurements.

On 9 October 2026 the source-bound GPU attempts `private_yolo26_candidate_run1`
and `private_yolo26_candidate_run2` stopped before model loading at the RAM guard.
Run2 retained available RAM 866,926,592 bytes, free VRAM 5,326,766,080 bytes and
free disk 50,873,257,984 bytes. The resource receipt is
`outputs/private_yolo26_candidate_run2_2026-10-09/evaluation_receipt.json`.
These attempts supply no actual integrated inference, detection accuracy or
rendering evidence. The source-alignment test uses a synthetic bounded clip and
is separately labeled software verification.

Run3 passed the unchanged resource guard but stopped at a device-placement bug:
PyTorch rejected the Ultralytics device ID string `0`. Its failed receipt is
retained. The shared three-role adapter now translates integer GPU IDs to
`cuda:index` for PyTorch placement while retaining the original Ultralytics
inference argument; CPU and automatic selection keep their established behavior.
Normalization and model-boundary tests passed 42 checks without GPU work.

Run4 completed at
`outputs/private_yolo26_candidate_run4_2026-10-09/evaluation_receipt.json`.
It inferred the human model 30 times, ball model 30 times and court model 31 times:
one initial calibration observation plus 30 current-frame observations. Both
source and annotated videos fully decoded to 30 frames. Source indices 0–29,
PTS 0–14848 at `1/12800`, and all 29 internal native transitions were retained.
All 31 generated JSON files were finite. Every `src/**/*.py` file and the exact
script/config/tracker/manifests/source/checkpoint/provenance inputs were hashed
before and after the report; no changed, added or removed inputs were found.

Initial court geometry retained 14 ordered null landmark slots and a null
homography. All 30 camera frames withheld analysis eligibility; observed identity
rows remained unknown with no A/B binding. Physical player metrics, event and
scoring authority stayed unavailable. The extracted annotated review frame is
`outputs/private_yolo26_candidate_run4_2026-10-09/review/annotated_frame_000015.png`.
Visual review found the observed ball marker within the publisher scoreboard.
This is an apparent development false positive without independent human error
scoring; 30 detector-state rows are not 30 correct detections.

Run4's initial guard retained 2,016,223,232 bytes available RAM and 5,326,766,080
bytes free VRAM. At 0.25-second sampling, maximum owned RSS was 1,910,906,880
bytes and minimum available RAM was 959,340,544 bytes. The process peak working
set was 1,930,022,912 bytes; CUDA peak allocated/reserved memory was
535,009,792/689,963,008 bytes. Sampled extrema are lower bounds on observed peaks;
CUDA allocation counters exclude other processes and driver allocations. The
1.5 GiB guard is an admission check, not a guarantee of sustained available RAM.
The combined pipeline stage took 5.062 seconds for this 1.2-second prefix; model
construction took 0.429 seconds. These measurements do not establish complete-match
capacity, live latency, useful specialist quality or production qualification.
