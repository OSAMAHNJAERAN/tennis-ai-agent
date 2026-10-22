# Court initial20: native DEV overlay review

All twenty saved 1280×720 overlays were actually opened at original detail by `/root/court_training_protocol`. This review created no predictions, tests, training runs, label changes or checkpoint changes. The diagnostic SHA256 remains `768fb2f4d016e0d704c2bb555e5927395f3a3c76b5453f8cc6fbcb7d98b09aec`; all twenty viewed overlay hashes were rechecked against that receipt.

The initial20 run produced useful learning signal, but the retained best checkpoint does not yet localize the court precisely. The next repair should be a **predeclared paired DEV input-transform investigation before further optimization or data expansion**. No new inference or longer retrain is authorized by this review. The proposal is in `proposed_next_diagnostic.json` and remains unexecuted pending root release.

The exact checkpoint reviewed is `outputs/yolo26_court_reviewed_development_run1/training/weights/best.pt`, SHA256 `e47db3784adb39b82be6df7fca73ed0206f67f4f5d1bb029f0f862a34a4feefa`. It was chosen by the run's DEV fitness policy, not by this review. `last.pt`, thresholds and labels were not substituted.

## Measured result and its population

There are 278 visible GT targets and two unknown GT slots across twenty DEV images. Nineteen images return one selected court instance with bbox IoU≥0.5; image16 returns none. The 264 finite, confident, matched visible keypoints give **94.96% measurement coverage**, not 94.96% correct localization. All fourteen missing targets remain in the recall denominator. The two unknown targets remain excluded.

Only **20/278 targets are within10px (7.19%)** and **74/278 within20px (26.62%)**. Conditional on an available, confident, bbox-matched point, median error is **31.5309px**, P95 **122.8922px**. No returned extra instances were present at the frozen settings; these positive court images cannot establish a negative-video false-alarm rate.

The twenty images belong to just two conservative event/context groups, with repeated stills from publisher videos. They are not twenty independent matches. Original FPS, PTS and verified match/session identity remain unknown. No video stability, metric ground geometry, line-call accuracy, independent human qualification or production promotion follows from this result.

## What is visibly wrong

The failure is often localization after a successful court-box match. Several blue and clay views place left semantic landmarks far inside the court toward image-right, while the near-right service point remains much closer. Several indoor views also put the far baseline too high and the near baseline too low, expanding the outer vertical extent around relatively good service-center points. These patterns are visible repeatedly within the same publisher videos.

The asymmetry is strong in the retained numbers. Slots0/2/4/5/8/10 on the left have **6/119 within20px**, no target within10px, and conditional median error **52.2544px**. Their mean signed horizontal error is **+45.3894px** (positive means image-right); 92/113 available points shift right and21 shift left. Right slots1/3/6/7/9/11 have **50/119 within20px**, conditional median **22.4349px**, and mean horizontal error **−7.7789px**. Service-center slots12/13 have **18/40 within20px**.

This is not one constant translation: image19 has severe left-side displacement toward image-left, opposite images3/6/8/20. A post-hoc global offset or label adjustment would be unsupported. The green GT intersections are visibly on court markings in this review; the overlays do not provide evidence for changing the frozen annotation convention to follow the predictions.

Image16 is a complete miss on an ordinary elevated clay singles frame with visible court lines and fourteen visible GT labels. It is not a blank frame or obvious close-up. The saved result alone does not identify why no court was returned.

## Every native overlay

The detailed, explicitly attributed observations are in `observations.json`. The table below summarizes the same twenty actual views. Medians are conditional on available points; misses stay in each visible denominator.

| # | Publisher image ID | ≤20px / visible GT | Median px | Observed pattern |
|---:|---|---:|---:|---|
| 01 | oxaylunY_hc_250 | 5/14 | 47.44 | Large image-right shift of left landmarks; right near-service point close. |
| 02 | V6NR0NAN_cA_3650 | 3/14 | 22.97 | Moderate up/left displacement; recognizable topology. |
| 03 | VtqY_T2_gI4_750 | 1/14 | 67.81 | Severe inward left-landmark deformation; far points low, near-left points high. |
| 04 | bQiC1RcZOe4_200 | 4/14 | 24.78 | Far baseline high, near baseline low; middle service region closer. |
| 05 | oxaylunY_hc_800 | 4/14 | 30.11 | Repeats image01's left deformation at lower magnitude. |
| 06 | 5QObSWGBQB8_1200 | 1/13 | 55.50 | Large image-right left-side displacement; GT slot2 remains unknown. |
| 07 | wMjFLZGBBho_450 | 11/14 | 16.78 | Relative success; residual far-baseline height and left-service error. |
| 08 | 5QObSWGBQB8_1250 | 1/13 | 63.73 | Repeats image06's left deformation after changed framing; GT slot3 unknown. |
| 09 | wMjFLZGBBho_700 | 9/14 | 10.35 | Relative success; right/center good, left endpoints retain offsets. |
| 10 | UuYiD9JWAgI_100 | 4/14 | 33.54 | Left-side inward displacement; right near-service close. |
| 11 | gyIQ4Zx7qKc_200 | 4/14 | 33.09 | Vertical extent expansion and inward far-baseline coordinates. |
| 12 | V-nCyiVkZpg_700 | 6/14 | 29.99 | Far-side inward displacement plus vertical extent expansion. |
| 13 | m_njKoEjGQQ_1000 | 3/14 | 30.65 | Outer baselines too far apart vertically; service region closer. |
| 14 | bQiC1RcZOe4_500 | 4/14 | 24.34 | Repeats image04's vertical expansion and far-left error. |
| 15 | gyIQ4Zx7qKc_500 | 4/14 | 31.84 | Repeats image11's expansion; near center almost coincident. |
| 16 | uXkAqALS0AA_100 | 0/14 | unavailable | No returned court despite visibly present primary-view court. |
| 17 | V-nCyiVkZpg_350 | 5/14 | 29.88 | Similar to image12, with stronger left-side inward error. |
| 18 | m_njKoEjGQQ_250 | 3/14 | 30.02 | Repeats image13's outer-extent deformation. |
| 19 | UuYiD9JWAgI_1850 | 1/14 | 70.96 | Large left-side error reverses toward image-left; far points too high. |
| 20 | VtqY_T2_gI4_700 | 1/14 | 67.61 | Strong repetition of image03's asymmetric deformation. |

Images06 and08 have respectively slot2 and slot3 unknown. Returned coordinates at those slots are not hidden-point accuracy evidence. Similar errors in paired stills demonstrate correlation in this saved development sample, not qualified temporal stability: the ordinals have no verified time base.

## Context and landmark grouping

| Conservative context | Visible GT | Available | ≤10px | ≤20px | Conditional median / P95 px | Review |
|---|---:|---:|---:|---:|---:|---|
| context_7cce172ce37d: visibly indoor ATP Finals blue courts | 140 | 140 | 8 | 42 | 29.2963 /85.7867 | All ten court instances matched; repeated vertical expansion and left-side errors. |
| context_e5bc79257ff1: visibly Monte Carlo clay courts | 138 | 124 | 12 | 32 | 33.2754 /136.9505 | Relative successes07/09 coexist with severe asymmetric errors03/06/08/19/20 and complete miss16. |

These observed event names describe the visible context only; they do not identify verified original matches or sessions.

| Slot | Landmark | ≤20px / visible GT | Conditional median / P95 px |
|---:|---|---:|---:|
| 0 | far_left_doubles | 0/20 | 62.2288 /136.3552 |
| 1 | far_right_doubles | 3/20 | 37.6088 /50.2811 |
| 2 | near_left_doubles | 1/19 | 39.8320 /160.1292 |
| 3 | near_right_doubles | 6/19 | 25.4632 /42.7842 |
| 4 | far_left_singles | 1/20 | 52.9607 /118.0711 |
| 5 | near_left_singles | 1/20 | 38.0816 /137.3084 |
| 6 | far_right_singles | 6/20 | 25.6427 /52.9608 |
| 7 | near_right_singles | 7/20 | 22.3930 /33.0573 |
| 8 | far_service_left | 1/20 | 51.1469 /118.4286 |
| 9 | far_service_right | 9/20 | 21.0037 /43.7672 |
| 10 | near_service_left | 2/20 | 42.7142 /124.4638 |
| 11 | near_service_right | 19/20 | 12.6217 /15.3611 |
| 12 | far_service_center | 9/20 | 23.2067 /74.4144 |
| 13 | near_service_center | 9/20 | 22.7362 /62.0611 |

The single complete miss contributes one failed visible target to every slot. Slots2/3 each exclude one unknown GT target. Slot11's19/20 result is therefore19 successful available points and one image miss, not an independent high-accuracy qualification result.

## Why investigate the input transform next

Training used `imgsz:1280`, `rect:false`, `multi_scale:0`; the three retained actual TRAIN previews were1280×1280. The diagnostic forward hook records the shape union **[1,3,1280,1280]** and **[1,3,736,1280]**. That receipt does not associate each hook call with an image or warmup. It does not prove that shape differences caused these errors, nor that simply forcing square inputs repairs them.

A small, frozen two-arm comparison on these same twenty DEV images is the most direct next engineering check: explicit rectangular input versus explicit square input, keeping the checkpoint, native full image, thresholds, semantic ordering and selection rule identical. Record actual shape, padding, dtype and native coordinate roundtrip for every image separately from warmup. Report both arms and all misses; do not choose an arm per image, search thresholds, apply GT-derived offsets or overwrite this original diagnostic. No new final-test or independent-accuracy claim is allowed.

There is evidence that optimization learned something: retained training pose loss fell7.78613→0.17276, and DEV pose loss fell11.2208→2.99646 across epochs1→20. DEV pose loss was lower at epoch16 (2.46563), then increased by epoch20 while training loss kept decreasing. This does not support a blind longer run. The final epoch CSV's pose mAP50/mAP50–95 (0.7898/0.42876) uses a different task-level metric and the same development population; it does not override the poor native-pixel localization of the retained best checkpoint.

If a predeclared transform comparison materially changes the deformation or missing instance, investigate that consistency issue before further optimization. If it does not, the input-shape hypothesis is weakened: prioritize reviewed source-diverse court data and a separately reviewed bounded optimization plan. The existing small, correlated development split still cannot qualify production, regardless of an improvement in either arm.

## Retained artifacts

- `observations.json`: explicit manual observation for each of the twenty actually viewed overlays.
- `retained_error_decomposition.json`: exact source bindings, all twenty overlay hashes, original aggregates, per-image observations, signed errors by landmark/context/side, and all280 retained semantic-slot records. SHA256 `644c0419bfe340c7fbb51cbe32404173abf11ef4a7bf953625cb1fb93d64cac3`.
- `derive_retained_review.ps1`: the executed saved-prediction-only derivation; refuses to overwrite its output and performs no model/image-runtime imports or inference.
- `proposed_next_diagnostic.json`: a concrete unexecuted paired-transform proposal, with both inference and training authorization explicitly false.

The source diagnostic and all original training, checkpoint, data and protocol receipts remain untouched. Review-derived signed means and error-pattern groupings are descriptive post-hoc analysis of existing predictions, not new qualification metrics or an experiment that was already executed.
