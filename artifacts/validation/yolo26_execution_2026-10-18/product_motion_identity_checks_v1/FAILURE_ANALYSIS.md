# Product profile movement regression: source identity in a legacy fixture

The final combined run had one failure in `test_profiles_do_not_merge_names_and_unconfirmed_motion_is_excluded`. Its legacy `result.json` fixture contained `summary.durationSeconds=7` and a nonempty player metric, but omitted `summary.id`. The expected movement row therefore could not pass the current result reader's source-identity contract.

This is not caused by a new physical-measurement qualification gate. The current `player_detail` handler requires an uploaded completed match, linked profile and confirmed whole-clip same/swapped role; it then calls `open_result(output, analysis_id)`, reads the matching `playerMetrics` entry and returns it as a `video_estimate`. The `ResultDocument` reader independently requires the saved summary's ID to match the requested analysis ID, including historical plain-JSON results. A role confirmation cannot replace that source binding. Invalid or foreign result identity is caught and excluded from the profile history.

The documented product behavior in `docs/PRODUCT_DASHBOARD.md` requires an explicit profile link and whole-clip role confirmation, and keeps movement estimates separate from personal identity guesses. The strict reader behavior is also independently covered by `tests/test_result_document.py::test_wrong_analysis_is_rejected`. No Git checkout is present here; retained source snapshots and receipts are the evidence used for this diagnosis.

## Exact failure and control

- `../product_motion_identity_failure_v1/receipt.json` and its original source snapshot preserve the **unmodified single-test failure**: one test, one expected failure, no errors/skips, unchanged five before/after pins, no Torch/Ultralytics imports. The original test file SHA is `66a06f43d4d9068dd1024e4387e5ff3742cd3fe4f980747a7b6a0c8ed7d3fe41`.
- Root explicitly authorized only a fixture repair and two negative source-identity controls. The existing positive test now adds `summary.id='match_test'` to its fixture and reuses the exact saved payload for its unchanged-result assertion. Its original identity/profile/role-clear assertions remain.
- Two parameterized negative cases retain the same nonempty metric payload but either omit summary ID or give a foreign ID. Even after an actual HTTP metadata update confirms the same role, movement stays empty; the profile still has its one linked match; direct `open_result` raises the specific different-analysis error; saved result bytes stay unchanged.
- `receipt.json` records **11 passed**, no failures/errors/skips, 7.34879339998588 runner seconds (7.00 pytest seconds), unchanged five source pins, no model imports. Current test SHA is `e1766b8c1f6ab18c282016cdabdf194d28b31885710b99f2137fb9016f3466ad`.

## Unchanged application sources

| Source | SHA256 in both the failed reproduction and passing controls |
| --- | --- |
| `src/api/product.py` | `9f0216fdbcba2f685e53d7abcd55642e874bfbe6a0b985ca93900bf99a709750` |
| `src/api/result_document.py` | `27595098ae059ab81b659b86e362af41cbe10ed53bf87be09dab0a963d3a0fc8` |
| `src/api/result_storage.py` | `4eb1d7a2dfeb33d93a6ca5fcee536ec34cbe17eaa793a55b9ac824cc02de1213` |

There is no application behavior change, relaxed source binding or measurement qualification change. These regressions establish profile linking and saved-source identity behavior only; they do not establish the underlying movement metric's accuracy or independent qualification. The broader combined test suite needs a new final run because its earlier receipt correctly recorded the stale fixture failure.
