# Frontend QA Record

Date: 2026-08-22  
Branch: `codex/frontend-dashboard`  
Starting SHA: `7e92a765624301a9b4951a10f12283ff4c3ce519`

## Automated frontend verification

| Check | Result |
|---|---|
| pnpm peer dependency check | Pass |
| ESLint | Pass, zero warnings |
| TypeScript project references | Pass |
| Vitest | Pass, 7 files and 26 tests |
| axe-core component scan | Pass, zero violations with color contrast excluded because jsdom cannot calculate rendered contrast |
| Vite production build | Pass |
| Browser console | Pass, zero warning or error entries during route and interaction QA |

The test suite covers the seven current run generations, snapshot and run contracts, scientific enums, historical missing artifacts, invalid run IDs, number formatting, coverage derivation, coordinate mapping, event stepping, review-required behavior, navigation, filtering, event drawers, responsive event records, keyboard video playback, overlay toggles, loading, empty, error, and retry states.

## Route smoke checks

The following production routes rendered their expected level-one heading with no alert state and no horizontal document overflow:

- `/`
- `/matches`
- `/analysis`
- `/court`
- `/players`
- `/ball`
- `/events`
- `/line-calls`
- `/reports`
- `/system`

## Responsive matrix

Each required viewport was exercised against the production overview. The audit checked document width, visible workspace elements outside the viewport, navigation and main-content boundaries, and mobile interactive target sizing.

| Viewport | Navigation | Horizontal overflow | Off-screen workspace elements |
|---|---|---:|---:|
| 1920 × 1080 | 224px desktop | None | None |
| 1600 × 900 | 224px desktop | None | None |
| 1440 × 900 | 224px desktop | None | None |
| 1366 × 768 | 224px desktop | None | None |
| 1280 × 800 | 224px desktop | None | None |
| 1024 × 768 | 224px desktop | None | None |
| 834 × 1194 | Mobile drawer | None | None |
| 768 × 1024 | Mobile drawer | None | None |
| 430 × 932 | Mobile drawer | None | None |
| 390 × 844 | Mobile drawer | None | None |
| 360 × 800 | Mobile drawer | None | None |

Representative visual inspection was completed at 1600 × 900, 1440 × 900, 834 × 1194, and 390 × 844. The mobile court, event drawer, and video player were inspected separately.

## Interaction checks

- Mobile navigation opened as a 320px accessible dialog, locked background scrolling, closed after navigation, and did not cover the destination content.
- Navigation retains the selected `run` and `event` search parameters.
- Event filtering reduced the Phase 5 record set from six events to three bounces.
- Opening a mobile event record created an accessible detail dialog and updated the URL to `event=2`.
- The replay drawer exposed the frame, time, player, tracker state, coordinates, evidence fields, and replay action.
- The video ball overlay toggled from on to off.
- Event stepping updated the URL and synchronized the player to the selected event.
- Mobile video controls wrapped into two measured rows without overlap.
- The invalid-run state displayed the normalized repository error and a functioning retry action.
- The current line-call data displayed the reusable no-review state; synthetic component coverage verifies `REVIEW_REQUIRED` produces an alert workflow.

## Backend regression baseline

No Python dependencies or backend files were changed.

- Full command: `python -m pytest tests -q -p no:cacheprovider`
- Result: collection stopped with 9 errors because pandas, OpenCV (`cv2`), and Torch are not installed in the current Python environment.
- Dependency-light command: `python -m pytest tests/test_bbox_utils.py tests/test_court_geometry.py -q -p no:cacheprovider`
- Result: 9 passing tests.

This matches the recorded pre-frontend baseline.

## Token and copy scan

- No occurrence of the three retired accent hex values remains in active frontend, documentation, or design files.
- `#dbfe00` is the only chromatic interface signal.
- Remaining legacy color-name strings are limited to preserved token identifiers required for compatibility.
- No visible application copy contains an em dash or en dash.

## Verdict

**PASS: READY FOR FRONTEND HANDOFF**

The production snapshot SPA meets the requested frontend-only scope and passes the automated, responsive, interaction, and accessibility checks recorded above.
