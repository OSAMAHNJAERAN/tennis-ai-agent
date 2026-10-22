# Dashboard implementation and verification

Verified locally on 19–20 September 2026. Scope: the approved five-phase product redesign, preserving Astra's existing identity and the existing inference pipeline.

## Phase outcomes

| Phase | Completed | Verification |
| --- | --- | --- |
| 1 — Navigation and Home | Global Home and searchable Match Center; separate Examples; contextual analysis; obsolete destinations redirected; global pages independent of analysis loading | Frontend route-selection and library-filter tests; browser Home and mobile navigation checks |
| 2 — Upload and match information | Optional participant/profile, tournament and match date; transactional draft/job linkage; direct job status; editable metadata; unknown identities remain unknown | API migration/upload tests; real browser upload to completed workspace; metadata edit preserves artifact hashes |
| 3 — Analysis workspace | Replay, Statistics, Insights, Details & exports; source/annotated selector; shared replay and insight interval; mini court; specialist tools and exports; capability-based unavailable messages | Browser playback/source switching and return links; existing replay/event/export regression suites; production build |
| 4 — Manual live desk | Required initial server; best-of-three/five; scoring history; pause/resume, undo and corrections; revision conflicts and idempotent commands; persisted point/service/return statistics | Deterministic scoring tests through deuce, tiebreak and match completion; browser point/correction/undo/pause; restart persistence |
| 5 — Players, insights and settings | Explicit profiles and linked histories; confirmation-gated movement estimates; persistent notes; local observations/coaching adapter; prediction availability contract; service status and replay preferences | Duplicate-name and role-confirmation tests; browser unconfirmed-history exclusion, note persistence, mobile profile layout; coaching accessibility and adapter tests |

No implementation phases remain within the agreed scope. External sports feeds, OCR, predictions, streaming inference and doubles analysis remain deliberately unconnected or unsupported.

## Automated verification

- Full Python regression suite: **833 passed**, with one existing Starlette/httpx deprecation warning. No backend changes followed that full run.
- Final focused product API rerun: **29 passed** (`test_product_matches.py` and `test_product_api.py`).
- Full frontend regression suite after navigation/accessibility fixes: **78 passed across 16 files**.
- Final outage-state regression rerun: **8 passed** in `ProductFlows.test.tsx`, including two new checks that distinguish an unavailable library from an empty one.
- ESLint, TypeScript and the production build passed, including the final outage-state change.
- `git diff --check` passed; Git reported only the repository's automatic LF/CRLF conversion notices.

The Windows Node 26 environment had slow test startup. A focused run completed successfully, followed by the full 78-test suite. Node emitted an experimental localStorage warning; browser preferences explicitly fall back when storage is unavailable.

## Real upload journey

The browser selected `data/sample_videos/input_video.mp4`, selected a saved test profile, supplied an explicitly unverified second participant and tournament, and submitted analysis. The processing screen automatically opened the completed match workspace.

- Match: `session_860985ec562b4c1a96ee038520525129`
- Analysis: `match_f2a47cbd9ab14ddfa3acedbf497d3ea0`
- Result: **214 frames, 30 FPS, 7.13 seconds**, completed inference, playable source and annotated video, court and player trajectories, movement estimates and downloadable artifacts.
- Original file SHA-256: `5f2584a2010864d5d8624c325d7f0571d48bd0756164657ef603dc8a97bab22d`
- Normalized result SHA-256: `c8090d2bdb7037c0914fa6d69996c0841472379c1d2cefd428d2fe7ff9bcdd06`

Both hashes remained identical after saving notes and correcting tournament metadata. Verification reports: [initial upload](../../artifacts/validation/product_integration/dashboard_upload.json) and [after metadata correction](../../artifacts/validation/product_integration/dashboard_after_metadata.json).

These are workflow/artifact checks, not evidence that the models meet professional measurement accuracy. The pipeline continues to report `production_qualified: false`. Unsupported ball speed, authoritative bounce/shot events, rally statistics and inferred scoring remain unavailable.

## Browser checks

- Home displays the two primary actions and persisted records without an example selected.
- Mobile Home and player screens checked at **375 × 812**; desktop workspace checked at **1440 × 1000**. Navigation drawer opens, focuses its close control, follows a destination and closes. Temporary viewport overrides are reset after testing.
- Manual verification match `session_e1017d6880e44a85a6234269adc3e06e`: point to A, correct winner to B, undo to 0–0, pause. It remains paused after the server restart.
- Notes saved, reloaded and remained present. Match metadata corrected independently of inference.
- Saved profile `player_811295372c2244328df47328d3536970` lists its uploaded match but has **no confirmed movement history**, because whole-clip role identity remains unconfirmed.
- Replay starts with the source video. Source/annotated switching works; the actual annotated video loaded successfully. Keyboard seeking to frame 213, switching to Insights and returning through its link preserved frame 213, annotated selection and interval 30–213. Statistics' return link likewise preserved frame and interval.
- Insights for frames 30–213 displayed **368 player-frame samples** and a **6.1-second interval**, matching the selected range. Saved notes remain separate from these observations. Predictions show unavailable.
- The visible coaching question retains its accessible label while the replay's coaching panel remains mounted. Collapsed sidebar links have explicit accessible names.
- Match details precede the collapsed analysis export/diagnostic section. Doubles is visibly unavailable and whole-clip identity confirmation is explicit.

## Boundaries

This is a local, single-host application. Full-match capacity, multi-user deployment, external-provider availability and cross-camera identity continuity are not established by this short video check. Match statistics use full-clip evidence; the selected interval applies to replay density, positioning observations, coaching and interval exports. User-created test records are clearly labeled and remain available for inspection.

See [product workflows, storage and APIs](../PRODUCT_DASHBOARD.md) and [local setup](../PRODUCT_WORKFLOW.md).
