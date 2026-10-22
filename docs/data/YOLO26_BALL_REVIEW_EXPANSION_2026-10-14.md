# Ball data repair: grouped proposal review

This is private local academic development for the unchanged three specialist roles: YOLO26m ball Detect, YOLO26s player Pose, and YOLO26s court Pose with fourteen fixed landmarks. It does not run a model, start training, export YOLO labels, or establish independent accuracy.

## Retained starting evidence

The native640 crop pilot had71 train and10 development boxes and weak development recall. The fixed native-tile diagnostic matched1/10 development targets. The30-frame private engineering run also placed a ball candidate on a scoreboard digit at source frame15 of `match172_000`. These failures motivate broader reviewed data; another repeated five-image pilot cannot establish useful coverage.

The three existing RacketVision caches are publisher `TRAINING_ONLY` releases pinned to `85157ca21faa2abca96d837dd2b963738029bcc8`. Their acquired dataset card declares MIT. Use here remains private local academic development because raw broadcast redistribution is not separately cleared. Original sparse CSV states are retained as publisher evidence: Visibility1 supplies a center; Visibility0 means not visible; a missing row is unannotated. Neither Visibility0 nor a missing row establishes ball absence.

The previous saved `scripts/data/prepare_ball_review_expansion.py` is retained unchanged. No proposal output from it was found. Its source-group table covers59 of60 clips and would refuse the existing scope inventory because it omits `match210`. The fresh workflow accounts for that clip in `unresolved_event_quarantine`, without assigning a guessed tournament or decoding it.

## Frozen sampling contract

The new `scripts/data/prepare_yolo26_ball_review_wave.py` freezes600 distinct publisher-visible center candidates before this workflow opens source frames:480 proposed training and120 proposed development. It selects across16 conservative visual event groups. Visual branding does not prove original match/session identity; every included source remains development for qualification purposes.

| Proposed split | Conservative event groups |
|---|---|
| Train | Beijing, Shanghai, Halle, Queen's, Miami, Rome, Torino, Sydney, Monte Carlo, Perth, Melbourne, Rotterdam, Madrid |
| Development | Washington, Dubai, Stockholm |

Entire Barcelona, Paris, Montreal, Indian Wells, Montpellier, St Petersburg and unresolved event groups are excluded. Historical internal selection clips are excluded individually as an additional check. Doubles clips174/187/213 and closeup209 are excluded. Beijing172/201 is the user's explicit exception despite historic selection neighbor19;19 is excluded and no independence or comparison to the old selection result is claimed.

Within each eligible event group, publisher center height and image displacement supply upper-image, lower-image and motion strata, with a temporally spread residual. These are review proxies, not labels for far/near, blur, contact or active-ball status. Duplicate source frames are rejected. Source/event groups never span proposed train and development.

The first wave is96 center proposals, six evenly spaced selected candidates per eligible group. It also renders one complete native640 tile proposal per group and the unannotated source-frame15 triage. The full freeze retains120 potential tile review items for later batches. The first wave does not decode/render the remaining504 center proposals or104 tile proposals. Decoder work proceeds one source at a time; no model or GPU is used.

## Native pixels, timing and review

Each item retains source/video and CSV hashes, publisher center and state, exact integer source PTS/time base, native frame index, a96px patch origin, proposed box edges in native/patch pixels, and lossless PNG and pixel hashes. Context uses nearest actual source PTS to minus/plus0.10 seconds, with exact rational arithmetic and the earlier source frame chosen on an exact tie. There is no manufactured nominal-FPS timeline.

Review boards show a native96px patch enlarged six times with nearest-neighbor sampling, both temporal patches, and a full-frame thumbnail. The bright/yellow connected-component rectangle is explicitly a PROPOSAL. The16px fallback is explicitly a hypothesis. Unmarked native frames and patches remain separate from display overlays. Six candidate panels make one manageable review board; native640 tiles are displayed at full scale with adjacent context.

Every candidate needs per-item visual adjudication of visible ball/smear, native box edges, active/spare status, blur, view and uncertainty. Component output is never automatically accepted ground truth. Every proposed negative tile needs complete native-pixel inspection for every visible active or spare ball; uncertain tiles cannot receive an empty label. The script deliberately contains no label export operation.

For Beijing source frame15, the publisher CSV has no row. The approximate reported ball region near847,275 is only a triage crop origin; it creates no proposed or accepted box. The reported distractor472.99,976.44 lies in the complete scoreboard tile at origin0,440. Both reviews use unmarked source PNGs, never an annotated pipeline HUD as ground truth.

NKUST outdoor reference and CalTennis January28–29/October16 provisional test groups remain unopened by this workflow. Previously inspected sources are development. No fresh independently frozen TEST is created by this sampler.

## Execution and checks

Use the explicit authorized interpreter and sequential prepare/render commands:

```powershell
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' -m scripts.data.prepare_yolo26_ball_review_wave --prepare --wave-size 96 --output outputs/ball_data_repair_review_wave96_v1
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' -m scripts.data.prepare_yolo26_ball_review_wave --render --output outputs/ball_data_repair_review_wave96_v1
```

Each new preparation directory is immutable. Render refuses altered sampler/code/protected source hashes. Light work retains the 1 GiB available-RAM floor and requires 2 GiB free disk. A refused attempt is retained and does not authorize reducing either guard.

Fourteen focused tests passed initially, covering exact VFR/nonzero-origin timing and ties, invalid timelines, native-to-patch inverse geometry, all60-source accounting, mixed event-group split rejection, duplicate source-frame rejection and lossless PNG pixels. No full suite was run for this data preparation.

The first actual preparation launch refused before creating its output directory or opening source frames: available RAM610,476,032 bytes against1,073,741,824. Its retained evidence is `outputs/ball_data_repair_review_wave96_v1_preflight001.json`. The script was then strengthened to hash-verify protected inputs before/after render; the refusal receipt explicitly distinguishes its current source hash from the prior attempt.

Completed proposal paths/counts will be recorded here only after the guarded preparation and render actually finish. Accepted visible boxes, accepted negatives, YOLO label exports and training runs remain zero until a separate per-item review and root decision.
