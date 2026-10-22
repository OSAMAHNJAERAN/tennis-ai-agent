# Continue the frozen ball review wave in a fresh output

The 600-center sampler remains the existing file
`outputs/ball_data_repair_plan600_v2/frozen_sampler_source_groups.json`, with
SHA256 `920be9491390ef62327a19bf8c034a623ee14bd233c7801b93565af279eaf428`.
It contains 480 proposed training centers and 120 development centers in
16 conservative event groups, 120 complete-tile review proposals, 96 selected
first-wave center IDs, 16 selected tile IDs, and one unannotated frame15 triage.
Its actual native PTS, source groups, train/development assignments and selected
IDs are preserved. This is development data, with no independent test claim.

The previous attempt stopped at the unchanged runtime memory guard after
writing 14 native PNG files and partial crops. It published no completed review
boards or review manifest. The retained root triage confirms three visible
center proposals; it accepts no exact supervised boxes and no negative tiles.
The failed attempt, its receipts and its partial files must remain in place.

`prepare_yolo26_ball_review_wave.py` now accepts the old sampler as an explicit,
hash-pinned input and renders into an entirely new sibling directory. It does
not prepare another sample, consume the old partial media, copy labels or
modify the old JSON/sidecar. The launch manifest distinguishes the original
sampler implementation SHA from the current renderer implementation SHA.
Updating the renderer is not represented as an unchanged sampler program.

Run only after the separate court GPU work has finished and the root has
released media work. The root should choose an unused output path; this is an
example complete PowerShell command using a new sibling directory:

```powershell
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' -m scripts.data.prepare_yolo26_ball_review_wave --render --frozen-plan 'outputs/ball_data_repair_plan600_v2/frozen_sampler_source_groups.json' --expected-plan-sha256 920be9491390ef62327a19bf8c034a623ee14bd233c7801b93565af279eaf428 --output 'outputs/ball_data_repair_review_wave96_v3'
```

An existing output directory is refused. A fresh child under the original
sampler directory is also refused. `--wave-size` is prepare-only: rendering
cannot select a different wave. Both the caller's explicit digest and the
original digest sidecar must match. Preflight checks the full sampler shape,
the unaccepted proposal state, groups, exact source PTS, source geometry,
wave IDs, protected cached CSV/card/manifests, the unchanged metadata/helper
programs, and the selected native videos and original PTS receipts.

The media guard still requires at least 1 GiB available RAM and 2 GiB free
disk. It runs before image-stack imports and again after those imports before
any output directory is created. Every measured RAM/disk/RSS observation must
be a nonnegative exact integer byte count; booleans, NaN, infinities and floats
are refused. Existing per-source and per-board guards
remain active. An admission failure creates no output directory. A failure
after admission retains only the new attempt's partial files, an immutable
launch manifest and a terminal failure receipt; it does not publish a complete
review manifest. The input hashes and renderer hash are checked again before
publishing successful review aids.

Launch-manifest and final-manifest write errors also produce a terminal failure
receipt when that file can be written. A partially written review-manifest file
is recorded separately and does not establish completion. A completed attempt
requires the `COMPLETED_PENDING_VISUAL_REVIEW_ONLY` terminal receipt and its
matching review-manifest hash. If saving a failure receipt itself fails, the
original error is retained with a note identifying the receipt write failure.

A completed render still accepts zero boxes and zero negatives, writes zero
YOLO labels, uses no model and performs no training. Every visible smear needs
native edge review and an exact supervised box. Every proposed negative needs
complete visual inspection of its 640-pixel native tile, including active and
spare balls. Missing or `Visibility=0` publisher rows do not establish absence.

The new `tests/test_yolo26_ball_frozen_render_contract.py` uses only stdlib test
utilities, JSON metadata and mocked media operations. It covers the retained
sampler, explicit hash/sidecar binding, changed PTS/groups/proposal acceptance,
fresh-directory refusal, guard ordering, separation of sampler/renderer
identity, old-artifact preservation, CLI restrictions, and retained failure
state. No image/model stack is imported by its test operations. These tests
were prepared without execution during the court training run; run them only
after the root releases testing.
