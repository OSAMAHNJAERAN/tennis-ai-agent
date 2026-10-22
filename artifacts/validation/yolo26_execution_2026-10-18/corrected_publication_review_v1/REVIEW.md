# Correction publication review

This read-only review used the exact current source pins retained in both receipts. Product, implementation and test files were unchanged by the reviewer. All media and commits were isolated explicit synthetic test generations; no model or GPU inference ran.

## P1: database baseline is learned after the generation manifest is written

`publish_corrected_result.final_check` computes `database_record = _file_record(storage._path)` at callback entry. The native generation writer has already hashed the database into its new manifest before that callback. A database mutation in the intervening compatibility-copy phase becomes the callback's baseline and passes its self-comparison. The new current pointer then selects a manifest whose expected database size/hash differs from the actual database.

`probe_candidate_database.py` calls real native preparation, rendering, report production and publication, mutating only its isolated candidate database after manifest construction and before the final callback. It returned a generation token, changed the current pointer, and the actual `ResultDocument` rejected the current result with `generation database is missing or its size differs`. All nine source pins remained stable.

Recommended repair: capture a detached trusted database record immediately when owned result storage publishes its just-closed database; verify it against the complete candidate before and after the final callback. Reconstruct and verify the native manifest/token from that frozen record and already sealed file/result/summary records so an altered manifest cannot be committed either. A bounded streamed root-JSON digest check against the sealed report also closes a mutation before the initial database-byte freeze. Coordinate this source change after active frozen verification runs finish.

Evidence: `receipt.json`, SHA-256 `9d8b9d514b431bffee1ef878d9d7148562c1e58792fa9b2245870588996b5f67`.

## P2: ordinary lost acknowledgements and process interrupts share a catch

The outer `except BaseException` in `publish_corrected_result` returns the committed generation whenever `_committed` succeeds. Consequently KeyboardInterrupt and SystemExit raised after a successful pointer commit are suppressed before the store's `publication_guard` can save completion and rethrow the interrupt.

`probe_execution_interrupts.py` wraps the real `publish_result`, raises each interrupt after its actual atomic CAS, and observes the publisher return a token for both cases. Both committed results are readable. Reconcile ordinary Exception acknowledgements; preserve committed artifacts and rethrow process interrupts so the durable guard can acknowledge completion while honoring shutdown.

Evidence: `execution_interrupt_receipt.json`, SHA-256 `3ed2ec1c20c32255fe086a6ab0b9ad7e6ac9d0f18efdee778bc8abee9d5d08f3`.

## Execution closure passed

The latter probe also calls actual `_StrictPoseModel`, `_crop_options`, `SelectedCorrectionPose.record`, `close` and publisher `_execution` using an explicitly synthetic in-memory predictor and a stubbed source-admission check. Plain metadata survives provider close and is detached by publisher validation. One provider invocation with zero predictions retains `new_vision_inference=False`; one provider invocation with two software proxy predictions retains True. Counts, pinned crop settings and `DISABLED_BOUND_SOURCE_CROPS` remain exact. This proves the software metadata seam, not actual checkpoint/CUDA execution. No additional P1/P2 execution-schema mismatch was found.
