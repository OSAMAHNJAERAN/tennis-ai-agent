# Private native640 ball repair proposal, 16 October 2026

Current status: the ROOT-approved export completed and passed full integrity
verification at 11:50:17 UTC on 16 October. The new dataset exists at
`data/yolo26/ball_author_repair_development_v2`. It remains private academic
development data; no model training or qualification was performed. Earlier
preflight and resource-refusal evidence below is retained as history.

The read-only proposal combines the immutable 79-image development dataset with
22 additional reviewed positive crops. It proposes **80 TRAIN and 21 DEV images,
97 visible-ball boxes and the original four reviewed absence crops**, with no
TEST data. The appended crops are private noncommercial academic development
data. Dataset export and model training have not run as part of this preflight.

The 22 appended crops come from two author-published tennis compilations:
13 from `America_Best_10_1` and nine from `France_Review_Red_Court_1`. All
32 selected source frames and 34 publisher proposals remain in the proposal,
including ten rejected frames containing 12 rejected proposals. Each admitted
extent must match its actual native temporal review and its complete unmarked
640-pixel crop review. Those reviews are attributed to agents; they are not
independent human annotations or qualification evidence.

The original 79 PNG images, label bytes, order, split and every other source
metadata field are retained. Their 58 TRAIN and 21 DEV assignments do not move.
The full old review lineage and conservative context components are reconstructed
through the original verified exporter. In particular, Miami/Washington,
Dubai/Rome and Stockholm stay DEV, while Paris, St Petersburg and all other old
quarantine components stay excluded. Rejected observations remain graph edges.

The new US Open and Roland Garros event-family observations are TRAIN proposals.
Their whole compilations remain grouped. Still-unseen `America_Best_Move_1` and
the three `Federer_Top_40` aliases remain conservatively connected to the US
compilation. Every one of the 16 publisher original/logical aliases remains in
the graph. An evidenced bridge from any new source, rejected frame or connected
unseen alias to old DEV or quarantine exposure refuses the proposed TRAIN
admission. Lack of an evidenced bridge does not prove original-match independence.
Original match IDs, camera session IDs and verified normal-speed timing remain
unavailable. The old `paris` label is not automatically interpreted as Roland
Garros.

The publisher's explicit research/academic grant is retained with its pinned
source texts. It supports this private academic development proposal; it does
not establish underlying broadcaster ownership, public redistribution rights,
public weight-release rights or commercial clearance. No independent test match,
event annotation or synchronized physical speed reference is created by this
proposal.

## Actual preflight evidence

The exporter is `scripts/data/export_yolo26_author_ball_repair_v2.py`, SHA256
`b4067e9cd0068628c5c0cf464c86ac78ab17f61cb6e0569690f9bcf198120ac0`.

The frozen concrete plan is
`artifacts/validation/yolo26_execution_2026-10-16/ball101_export_preflight_v2/plan.json`,
SHA256 `7c6fede7dc51f1a3d555d434276d440fd12786647bedc2e8dd1df8b46b62c135`.
Its canonical proposal hash is
`d24c2ed8a7249aa95a12cf91cfd0b83b33d05ee6b33f98f03577d36b92f332b0`.
It binds 891 input files, including actual review assets, native timing, source
media, old dataset provenance and the exporter source. The proposed fresh output
is `data/yolo26/ball_author_repair_development_v2`.

The actual read-only run completed all 22 source-PNG to crop-PNG comparisons in
35.86 seconds. Every RGB pixel exactly matched its native integral 640-pixel
ROI, and every stored RGB crop hash matched. A bounded stdlib PNG decoder checks
chunk CRCs, exact scanline byte counts, the zlib stream boundary and all five PNG
filter modes. It accepts only non-interlaced RGB8 and limits each input to
32 MiB and 2,073,600 pixels. No source video was decoded during this check.
No PIL, NumPy, Torch, Ultralytics or OpenCV module was loaded. Across 135 resource
samples, peak owned RSS was 66,162,688 bytes and minimum available RAM was
1,186,873,344 bytes. The existing 1 GiB RAM and 2 GiB disk guard was unchanged.
These measurements cover metadata and PNG verification, not model training.

ROOT review requested preservation of the already available native probe data.
The metadata refresh adds it to only the new 22 rows: nominal `r_frame_rate`,
separate `avg_frame_rate`, stream time base, frame count, start and duration,
container duration and size, and hash-bound probe attribution. USA nominal rate
is `2997/50`, average rate `9597060/160111`; France nominal rate is `25/1`, average
rate `1881000/75211`. Neither nominal rate is evidence of constant cadence or
normal-speed match footage, and both verification flags stay false. The source
repository, commit/blob, acquisition-receipt creation UTC (distinct from video
creation time), and exact author README URL/hash remain explicit.

The metadata refresh completed in 3.56 seconds. It proved that all original 79
rows are identical to the earlier proposal and all 22 pixel inputs, source
images, native geometry, PTS and actual review records are unchanged. The only
changed input binding is the exporter source. It links the completed native RGB
check in `ball101_export_preflight_v1/plan.json`, SHA256
`fa2321364d2bf5bce5d07f807c429d6ed4d7bab3028c8109480452ab10468d5b`,
and its receipt SHA256
`4856ba7e36009dd144457b69617b736d41146cbe0ef6ffa74543027c5e6b6360`.
The metadata refresh does not claim a second pixel decode; execution repeats the
complete native ROI check before copying any supervised data.

The latest scoped check receipt is
`artifacts/validation/yolo26_execution_2026-10-16/ball101_export_checks_v5/receipt.json`:
126 tests passed in 2.78 seconds. It covers native coordinate serialization,
unmodified inherited absence labels, source and closure review admission,
rejected/transitive leakage links, exact approval binding, input-hash changes,
all PNG filters, corrupt and oversized streams, one-pixel ROI alterations and
unapproved export refusal. It also checks source/probe mismatches, wrong native
shape/time base/frame PTS/count, nonmonotonic PTS, file size, publisher URL/commit,
rate and timestamp failures. The actual pinned metadata reconstruction is included.
The test file SHA256 is
`41653e396fd8cc2ab1de000c263ef8c60c91ea499cddfb03844506aa862f3575`.

Failed harness/check receipts remain intact. Check v1 stopped before test
collection because its pytest reporting hook used the wrong argument name.
Check v2 passed 111 tests and failed the actual metadata check because the new
comparison used Windows path separators against a canonical forward-slash
source-review path. The exporter comparison was corrected without altering any
source review. Check v3 passed all 112. Check v4 changed only the actual metadata
test so it remains repeatable after a later authorized export.

## Concrete approval and export

The frozen plan contains an `approval_template`. A separate ROOT review must
retain its exact proposal hash, input bindings and ordered 101 sample IDs, add
a nonempty review rationale, and save the approval as a new file. An approval
that changes private use, asserts human qualification, permits TEST data, or
grants public/commercial rights is refused. This is a project-internal artifact
review, not a new request for user permission.

Execution reconstructs the same complete proposal before creating the fresh
dataset, rechecks all 22 native pixel ROIs, copies the original 79 image/label
pairs byte-identically, writes precisely one reviewed positive label for each
new crop, and checks all input hashes again. It cannot overwrite an existing
dataset. The terminal manifest and receipt are then verified against the exact
file set, image and label hashes, train/val YAML, approval and lineage. No TEST
directories, empty new labels, resizing, jitter, model inference or training are
part of this export.

The current checkpoint remains weak on previously reviewed DEV material. A
successful export would prepare a controlled data repair; it would not qualify
that checkpoint or establish ball precision, recall, speed accuracy or long-match
readiness.

## ROOT approval and first actual attempt

ROOT approved the frozen proposal and source after reviewing the exporter and
the 126-test receipt. The separate approval is saved at
`artifacts/validation/yolo26_execution_2026-10-16/ball101_export_execution_v1/root_approval.json`,
SHA256 `5eeac680bafb6c6ebf24da2896646d08dafcf4de7d6f8c9a1a1e0cfdb0460409`.
It preserves the exact proposal, all 891 bindings and the ordered 101 sample IDs.

The actual attempt at 06:57:15 UTC on 16 October stopped at the unchanged launch
guard: available RAM was 881,541,120 bytes, below the required 1 GiB. Free disk
was 50,273,439,744 bytes. This was a terminal guard refusal, before native pixel
decoding or supervised data creation. The intended dataset destination remained
absent, with zero files copied or labels written. All five old frozen scripts
and the new exporter remained unchanged. No model, image library, GPU or training
runtime was loaded. The full failed attempt and resource evidence are retained
in `ball101_export_execution_v1/receipt.json`; a later attempt must use a new
execution receipt and the same resource guard and authorization.

## Completed fresh export after verified host-state change

After ROOT confirmed the separate semantic capacity and frontend check processes
were terminal, one stdlib-only headroom snapshot at 11:47:12 UTC measured
1,493,155,840 bytes available RAM and 64,738,787,328 bytes free disk. This passed
the extra practical launch requirement of 1.3 GiB, without changing the original
runtime 1 GiB RAM and 2 GiB disk guards. Its receipt is
`ball101_export_headroom_v3/receipt.json`, SHA256
`e73feec1f5a7df778ca1f12dba9588f313d3df53620f9f1d6fa94d50664d5cd1`.

Fresh `ball101_export_execution_v2` reused the original approval bytes without
rewriting them and verified all 891 frozen input bindings before export. The
actual process completed with exit code 0 from 11:48:16 to 11:50:17 UTC, in 120.79 seconds.
It repeated all 22 native RGB source/crop ROI comparisons, copied all 79 inherited
image/label pairs byte-identically, wrote exactly 22 new positive labels, and ran
the complete `verify_export()` reconstruction and file-set verification. The
result is 101 images: 80 TRAIN, 21 DEV, 97 positive boxes, four original absence crops,
zero TEST. There are 205 files totaling 26,133,296 logical bytes. No additional
empty labels or TEST directories were created.

The terminal hashes are:

- `data/yolo26/ball_author_repair_development_v2/source_manifest.json`:
  `fdd63635665ebd03426ab452496cf0b93ad52d462f9c32886e6ff7f3ab564c22`.
- `data/yolo26/ball_author_repair_development_v2/export_receipt.json`:
  `554fb861957b130f65b3936f63ac159dc65e2fc9fe0ed9a1eb5bc667f50568d6`.
- `data/yolo26/ball_author_repair_development_v2/ball/data.yaml`:
  `0941286e0cd90efab514b033378be4738f47549ab15e94ae7d123f8f5ebca0c8`.

The full observed receipt is
`artifacts/validation/yolo26_execution_2026-10-16/ball101_export_execution_v2/receipt.json`.
Launch available RAM was 1,820,561,408 bytes; the minimum across 428 resource samples
was 1,502,793,728 bytes. Peak sampled owned RSS was 69,226,496 bytes and Windows
peak working set was 82,391,040 bytes, with zero sampling errors. All 891 input
bindings, the exporter and the five old frozen scripts remained unchanged.
No PIL, NumPy, Torch, Ultralytics or OpenCV module was imported. No source video
decode, GPU operation, inference or training ran. These resource measurements
cover this dataset export only, not a prospective training run.

## Subsequent controlled training preparation and actual CPU inspection

The separate new trainer/diagnostic passed 100 pure tests in
`ball101_trainer_checks_v6/receipt.json`. The current concrete plan is
`ball101_trainer_preflight_v2/plan.json`, SHA256
`3fa459b60ae14b616c67f4664e3ee8488c445f7b9c6c2be4a90c892edb3b9cd2`;
its canonical hash is
`25993d6193aac6e12c8ab9c59501e086d42ce5f1debee5b387cbcebf4a8147ca`.
The prior v1 plan is historical: its preview selection predates the requirement
that both positive previews come from the new22 TRAIN examples.

The fixed native diagnostic protocol is
`ball101_trainer_preflight_v2/native20_protocol.json`, SHA256
`e0b53c112444270db6d2e020c108accde67aaaa4da5209b22bf9a17663be84fd`.
It preserves all twenty old DEV targets, eight native640 tiles per target and
one reviewed absence crop, 161 image forwards, confidence0.01/IoU0.5 matching,
the same deduplication and all misses in the denominator. Unmatched full-frame
proposals remain unadjudicated; precision and video false alarms per minute
remain null. The old trained best is the only initialization; the proposal is
two completed epochs of the same twenty-epoch schedule before another learning
review, with no new head, automatic parameter search or automatic retries.

The separately approved CPU inspection actually ran from 14:14:12 to 14:14:45 UTC
in 33.21 seconds. All **768/768 keys, shapes and exact FP32 values** transferred
from the registered old ball best, with zero random initialization. The actual
training factory retained 640×640 geometry and label extent for the smallest
and largest new22 TRAIN ball and an inherited reviewed absence crop. Root locally
viewed all three unmarked transformed tensors and three separate overlays.
CUDA was not initialized, and no optimizer/training or diagnostic inference ran.

Actual CPU receipt:
`outputs/yolo26_ball101_cpu_preview_v1/receipt.json`, SHA256
`159714bc4187c33784c57d62ec4fcfd105313ca509c049bc96558846dfde1efe`.
Root approval/execution/visual evidence:
`artifacts/validation/yolo26_execution_2026-10-16/ball101_cpu_release_v1/`;
use `root_visual_review_v2.json`, which explicitly corrects the first review's
end-memory wording. Available RAM was 2,433,081,344 bytes before Torch,
1,870,258,176 afterward, and 1,344,106,496 at the helper's final boundary.
The outer runner later saw 1,627,951,104 after model objects were released.
Peak owned working set was 1,054,941,184 bytes. Runtime/helper/source bindings
were unchanged throughout; transfer success does not establish GPU headroom.
The cheap 4.5 GiB pre-runtime GPU floor remains unchanged. Fixed DEV baseline,
actual two-epoch learning and independent qualification remain incomplete.

The next read-only GPU admission assessment at 14:19:09 UTC observed
2,642,948,096 available RAM bytes against the unchanged 4,831,838,208-byte
pre-runtime floor. `ball101_cpu_release_v1/next_gpu_admission_snapshot.json`
records that no Torch import, CUDA initialization, GPU attempt or training was
started. This is a resource assessment, not another failed inference run.
