# Astra Tennis: 3D intelligence upgrade

## Repository analysis (6 September 2026)

The Python pipeline in `src/pipeline/` composes YOLO11 detection, player/ball tracking, court homography, semantic events, line calling, and scoring. The existing numerical pipeline and its evaluation artifacts remain authoritative. This upgrade does not retrain models or change their predictions.

The frontend is React 19 + TypeScript + Vite 8 with Radix controls, Phosphor icons, native CSS, Zod contracts, and Vitest. `scripts/sync-demo-artifacts.mjs` assembles persisted JSON into `src/data/generated/analysis-snapshot.json`; `StaticAnalysisRepository` validates and serves it. There is no existing application API. Seven analysis runs exist; the default `phase5_scoring_1` has 214 frames at 30 FPS, player positions, a ball trajectory, six events, court calibration, and a local video. Earlier snapshots lack frame-level trajectories and must have useful empty states.

`SynchronizedTelemetryView` currently hosts video, SVG court, and SVG binned density. Only the video owns playback, so analytical views lack independent transport. `CourtDiagram` clamps players to court boundaries and joins gaps; the heatmap discards all behind-baseline samples. The Match Center also mixes genuine telemetry with hardcoded professional player names, scores, spin, and millimetric line claims. These are replaced on the upgraded workspace with selected-run evidence.

The incoming working tree has pre-existing modifications to shell, pages, video, court, styles, snapshot, and sync script. Preserve this work; do not reset or broadly stage it.

## Architecture and implementation plan

1. Create a shared metric geometry and normalization module. Map `(court_x, court_y)` to centered `(x, 0, z)` in meters; preserve nulls, tracking states, and frame indices. Include run-off space; never clamp observations to sidelines.
2. Introduce one replay transport with play/pause, frame steps, rates, loop, event seeking, and a frame range. Use the source FPS, bounded interpolation, and one shared playhead for video, 2D, 3D, and event inspector.
3. Add a lazy React Three Fiber scene with a regulation court, service lines inside singles boundaries, sagging net, posts, stylized articulated players, ground shadows, movement paths, and tracked ball. Camera presets: tactical, broadcast, top, cinematic; orbit, zoom, reset. Dispose resources and provide a WebGL fallback.
4. Animate player translation from adjacent samples, with locomotion driven by displacement. Avatars are stylized visualizations, not recovered body poses or inferred stroke classifications. Ball ground projection is the default. An explicitly illustrative arc can explain event-to-event travel; it is never a measured height or speed source.
5. Replace binned glyphs with normalized Gaussian density, a consistent shared intensity scale for P1/P2, run-off-aware sample counts, range filtering, selectable zones, and baseline/center/side breakdowns. Offer 2D density and 3D density columns.
6. Add a coach panel that explains observed occupancy and compares player movement. A configurable FastAPI service connects Astra GPT through a server-side model/endpoint/key. Offline observations remain clearly labeled; no fictitious generated response or match outcome.
7. Recompose Match Center as an evidence-led dark workspace with a large spatial stage, adjacent insights, compact player metrics, and an event rail. Retain routes and selected-run/event URL state. Add a direct spatial entry in navigation.
8. Test mapping, missing data, interpolation, density, range, evidence serialization, transport, provider errors, and accessible controls. Run frontend lint/typecheck/tests/build and targeted Python tests. Inspect desktop/mobile views in a browser and document results.

## Design direction

Charcoal surfaces, chalk-white Manrope typography, tabular Geist Mono telemetry, electric lime P1 and ice-blue P2. Lime is the action accent; the second color encodes player identity. A restrained court grid and directional light create depth. Use tight 8–16px control spacing and broader 24–32px panel spacing. The court dominates the visual hierarchy; details remain adjacent or progressive. Transitions use opacity and transforms; respect reduced motion. Mobile stacks the inspector below the court, wraps view controls, and keeps touch targets at least 44px.

## Data limitations

Single-camera homography yields court-plane estimates; airborne ball projection is not true 3D position. Court validity gates spatial rendering. Missing samples stay missing, large jumps break trails, and unmatched events never manufacture trajectories. Short clips support descriptive observations, not conclusions about why a match was lost. Speed labels retain court-plane semantics. No measured accuracy, spin, height, or real-time inference claim is added.

## Verified implementation references

- React Three Fiber v9 pairs with React 19: https://r3f.docs.pmnd.rs/
- Canvas fallback, DPR, render loop: https://r3f.docs.pmnd.rs/api/canvas
- Orbit controls and disposal: https://threejs.org/docs/pages/OrbitControls.html
- ITF dimensions corroborating `src/court/court_geometry.py`: https://www.itftennis.com/media/15604/atp-2026-rulebook.pdf

## Follow-on refactoring

Consolidate other legacy court renderers onto the same geometry contract; extend validated snapshots with camera segments and measured pose/height only if the pipeline supports them; replace illustrative content on unrelated scouting/tournament pages with real records in a separate data-product pass. Keep model qualification separate from presentation changes.
