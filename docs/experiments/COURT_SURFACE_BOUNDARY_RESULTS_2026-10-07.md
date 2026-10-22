# Conditional court-surface boundary evidence — 7 October 2026

The development candidate is **not promoted**. It improves the visible Dimitrov boundary fit, but loses one correct selection landmark against the same boundary optimizer using original line evidence. The production pipeline, selected model/configuration and physical-measurement authority remain unchanged. The full website execution objective remains incomplete.

## Method and retained scope

One candidate was frozen before real-case execution in [protocol.json](../../artifacts/validation/product_execution/court_surface_boundary_2026-10-07/protocol.json). It preserves every passing original line proposal exactly. Previously rejected proposals can gain evidence only where the unchanged selected connected surface mask consistently straddles the line, with one inside orientation at at least 80% of samples. Existing bilateral brightness and minimum-saturation conditions remain mandatory. No full hull mask, per-case tuning, optimizer/support threshold relaxation, model inference or production change was introduced.

The comparison uses the same boundary optimizer, seed and admission gates for original and candidate line evidence. The older unconstrained optimizer is replayed separately as a historical check. All 64 previous development observations were retained: 32 selection, eight external-source checks and 24 pilot observations. Only 47 have valid initial geometry; the other 17 remain abstentions. Forty cases carry the existing 14-landmark labels, scored with the same inclusive seven-native-pixel threshold and out-of-image exclusion. Repeated frames/models are not independent sessions.

Nine automatic source views reuse the exact first frames of three Commons, three UVY and three CalTennis inputs. They are development controls, not a qualification split. The known incorrect UVY spectator hypothesis is a separate specificity check. No new recordings, labels, weights or training were acquired.

## Synthetic checks and trial failure

The first synthetic attempt passed 34/35 checks. Its colored-ridge rejection fixture failed: gray neighbors can satisfy the inherited central-minimum-saturation condition even when the bright ridge is colored. The candidate was not changed to hide this limitation. A fresh fixture with saturated surroundings checks actual saturation rejection, and an additional test explicitly records the gray-background limitation. The follow-up passed **36/36**, with no failures, errors or skips. The original failed test/source/receipt remain intact in `tests/`; the follow-up is in `tests_v2/`. Candidate SHA256 remains `0e0a6420432ca3866326d0fe6c5fa69494110c2719e6faa6b7a658931ebccfd0` and protocol SHA256 remains `6b6609c7743a5a5434f2a56951fdaf4f1b29febeb32b7d69782bd5ef2fc93421`.

The one real trial saved all **64 case records, nine automatic records and the spectator record**, then **failed finalization**. The local name `actual`, initially a process object, was reused for court-point arrays; the final `actual.memory_info()` call raised before `report.json` could be written. This is a harness error, not a successful benchmark exit. The frozen script, failed receipt, logs, all 73 flushed progress records and images are preserved. No candidate, source decode, optimizer or trial was rerun.

[recovery.json](../../artifacts/validation/product_execution/court_surface_boundary_2026-10-07/recovery.json) copies the flushed observations and explicitly retains the failed-run status. A separate [independent arithmetic/integrity audit](../../artifacts/validation/product_execution/court_surface_boundary_2026-10-07/audit/receipt.json) passed: 2,240 native landmark errors across four stages, gains/losses, support and boundary arithmetic, proposal preservation, saved PNG pixels, source bindings, sample identities and product preservation. It does not reconstruct missing loaded-library/version inventories or unrecorded actual historical replay arrays, and does not convert the original execution into a pass.

The retained resource receipt records 176.167 seconds of monitored wall time, 662 samples and a 280,555,520-byte peak sampled aggregate owned RSS. All owned identities retired; 205 frozen inputs and all ten saved jobs/319 analysis files remain unchanged. Sampling can miss peaks or double-count shared pages. These are development-script resource observations, not inference throughput or full-match capacity measurements.

## Recovered measured results

| Development group | Stage | TP | FP | FN |
|---|---|---:|---:|---:|
| Selection, 32 observations | Original unconstrained refinement | 370 | 64 | 77 |
| Selection, 32 observations | Same boundary optimizer, original evidence | 369 | 65 | 78 |
| Selection, 32 observations | Same boundary optimizer, conditional evidence | 368 | 66 | 79 |
| External-source check, eight observations | Original unconstrained refinement | 65 | 33 | 47 |
| External-source check, eight observations | Same boundary optimizer, original evidence | 65 | 33 | 47 |
| External-source check, eight observations | Same boundary optimizer, conditional evidence | 65 | 33 | 47 |

Five fits changed: cases 14, 22, 23, 24 and 27. Against the same optimizer/evidence baseline, there were **zero gained identities and one lost identity**. Case 27 (`ktiDOhLZIVs_3500`), landmark 2, moved from 3.7643 to 7.5279 native pixels from its existing label, crossing the fixed seven-pixel threshold. Every individual error and original-versus-boundary comparison remains in the recovered records; pooled totals do not hide this loss.

Dimitrov is the sole newly admitted automatic **image hypothesis**. The candidate adds 14 proposals; its rough mean distance to the 13 already exposed source-review marks falls from the prior seed's approximately 18.78 to **8.8019 native pixels**. These marks carry approximately eight-pixel uncertainty, with occluded landmark 8 null. Source-context review sees substantial outer-boundary alignment improvement, with remaining fine offsets and umpire occlusion. This is not independent ground truth or physical court validation.

Djokovic's cropped half-court and Schiavone's green-runoff hypotheses remain rejected. The six UVY/CalTennis views produce no eligible four-corner component, so no automatic geometry is available; their abstentions do not establish conditional-refinement specificity. The saved spectator hypothesis still overlays grandstands and remains rejected. [Visual review](../../artifacts/validation/product_execution/court_surface_boundary_2026-10-07/visual_review.json) covers all 15 panels and 14 native source images. Small shifts in the five labeled fits do not justify a blanket visual accuracy claim.

## Website restoration and pending worker check

The prior website processes were absent at fresh inspection. Production source/build bindings and all saved analyses matched the retained 6 October verification. A fresh absent-runtime preflight passed, followed by documented startup and [post-start compatibility verification](../../artifacts/validation/product_execution/court_surface_boundary_2026-10-07/runtime_restore_verification.json): three demos, 12 exact exports, seven index assets, loopback server/worker identities, and unchanged ten jobs/319 files. No accepted job was cancelled or saved analysis replaced.

One fresh real-worker smoke attempt was prepared after an independent available-RAM observation of 2,530,119,680 bytes. Its actual launch preflight later saw **2,435,448,832 bytes**, below the unchanged **2,500,000,000-byte** requirement. It terminated **NOT_STARTED**, with zero child launches, samples or model work. [Independent terminal review](../../artifacts/validation/product_execution/pose_racket_worker_smoke_2026-10-07/terminal_review.json) verifies the decline and all prior attempt bindings. The attempt was not retried, and raw pose/racket output, worker completion and full-match capacity checks remain pending.

The [current data recheck](../research/ACCURACY_DATA_RECHECK_2026-10-07.md) found no newly verified untouched recordings satisfying the qualification conditions. Independent singles sessions, physical speed/3D references and automatic participant accuracy remain unresolved. Continue the full execution prompt; do not treat this rejected development candidate, a restored website or recovered arithmetic checks as completion.
