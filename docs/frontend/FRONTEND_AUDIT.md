# T88J709 Frontend Audit

## Audit baseline

- Branch at audit start: `master`
- Starting commit: `7e92a765624301a9b4951a10f12283ff4c3ce519`
- Implementation branch: `codex/frontend-dashboard`
- Existing untracked user files: `DESIGN.md`, `theme.css`, `variables.css`, `tokens.json`
- Frontend status: no existing JavaScript application, package manifest, route tree, component library, browser test setup, or build pipeline

The repository is a Python computer-vision project. The new frontend must remain a self-contained consumer of generated artifacts. It must not alter inference, tracking, court geometry, line calling, scoring, training, datasets, or model configuration.

## Existing technology

| Area | Current state | Frontend implication |
| --- | --- | --- |
| Runtime | Python 3 project with Pytest | Add an isolated Node workspace under `frontend/` |
| Package management | `requirements.txt`; no Node manifest | Use the bundled pnpm runtime and commit a lockfile |
| Frontend framework | None | Greenfield Vite, React, and TypeScript SPA |
| Routes and screens | None | Create task-focused analytics routes |
| State management | None | Keep URL state for run/event selection and local state for interaction |
| API/backend | No web API | Use a typed static artifact repository and documented future adapter boundary |
| Styling | Four supplied design reference files | Import their variables into the app and preserve their non-color tokens |
| Charts and tables | None | Add Recharts and TanStack Table with custom presentation |
| Browser testing | None | Use component tests plus in-app browser visual QA |
| CI/CD | None found | Provide deterministic local scripts; do not add deployment scope |

## Current visual and UX state

There is no current dashboard to preserve. The source design reference is a light developer-tool aesthetic with white and off-white surfaces, graphite text, hairline borders, subtle shadows, 8px utility radii, 16px card radii, pill controls, Suisse typography, Geist Mono telemetry, and a legacy chromatic accent family.

The user-mandated migration changes only the chromatic identity of that system. The legacy accent triplet now resolves to signal lime `#dbfe00`. The current neutral palette, typography scale, spacing scale, radii, and shadow recipes remain unchanged.

Because the reference was extracted from a marketing site, its centered hero, numbered section labels, three-card feature rows, world-map decoration, and Firecrawl-specific content are not suitable dashboard patterns. The frontend will reuse the tokens and material language, not the marketing information architecture.

## Accessibility and responsive gaps

No frontend currently exists, so all accessibility and responsive capabilities are missing:

- no semantic app landmarks or heading hierarchy;
- no keyboard navigation, focus treatment, dialogs, or tooltips;
- no responsive navigation or mobile data strategy;
- no chart descriptions or table semantics;
- no loading, empty, error, disabled, or review-required states;
- no reduced-motion behavior;
- no safeguards against bright-lime contrast failures;
- no overflow testing at the required viewport matrix.

## Existing analytics artifacts

The ignored `outputs/` directory contains seven real analysis runs:

1. `baseline_run_1`
2. `phase2_ball_1`
3. `phase2_yolo11_final`
4. `phase3_events_1`
5. `phase4_line_calls_1`
6. `phase4_1_line_calls`
7. `phase5_scoring_1`

The latest run contains:

- a 214-frame, 30 FPS trajectory and detection sequence;
- Player 1 and Player 2 court positions for every frame;
- six semantic match events;
- three line calls with signed margin, uncertainty, confidence, contact model, line geometry, and explanatory evidence;
- player distance, average speed, and coverage metrics;
- ball speed overview and seven flight segments;
- canonical court keypoints, homography, and reprojection error;
- match state, scoring events, pipeline metrics, and configuration metadata.

The input video is H.264 (`avc1`) and browser-compatible. Generated annotated videos use `mp4v`, which is not a reliable browser delivery codec. The frontend will therefore bundle the H.264 source and draw synchronized analytical overlays from the real artifacts.

## Schema variation

Artifact shapes evolve across pipeline generations:

- Baseline and Phase 2 use `detected_keypoints_14`, `canonical_keypoints_14`, and nested metrics summaries.
- Phase 3 changes court, event, trajectory, and metric field names.
- Phase 4 adds line-call evidence and player court-position series.
- Phase 5 adds match state, score history, and scoring events.

A frontend normalization boundary is required. Historical missing fields must map to explicit capability flags or `null`, never fabricated values.

## Safe integration points

- Read artifacts only through a frontend sync script.
- Commit a versioned normalized snapshot so the app builds without ignored output directories.
- Expose repository methods for listing runs, reading one normalized run, and resolving its video URL.
- Preserve raw scientific terminology and store source metadata alongside normalized values.
- Derive only presentation metrics such as counts and coverage ratios from existing arrays.
- Keep client-side downloads limited to already loaded JSON/CSV data.

## Existing test baseline

`python -m pytest tests -q` currently stops during collection because the local Python environment lacks pandas, OpenCV, and Torch. This is a pre-existing environment issue and must not be addressed by frontend changes.

The dependency-light baseline succeeds:

```text
python -m pytest tests/test_bbox_utils.py tests/test_court_geometry.py -q -p no:cacheprovider
9 passed
```

The same commands will be repeated after frontend work. Full-suite collection errors must remain attributable only to the unchanged missing Python dependencies.

## Audit conclusion

The correct implementation is a greenfield static analytics SPA with a strict artifact adapter. All planned pages are justified by existing data, with Reports and unsupported actions constrained to frontend-only behavior. No backend or scientific code change is required.
