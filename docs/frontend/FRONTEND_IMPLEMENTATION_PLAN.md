# T88J709 Frontend Implementation Plan

## Product direction

- Audience: researchers, technical reviewers, and tennis analysts
- Theme: light only, following the supplied design source
- Accent: signal lime `#dbfe00`, used only for active and selected states
- Design variance: 6
- Motion intensity: 3
- Visual density: 7
- Interaction model: compact workstation, not marketing site, betting UI, broadcast clone, or gaming HUD

## Foundation

Create a Vite, React, and TypeScript application under `frontend/` using Tailwind v4 for composition and the root design variables as the styling source of truth. Use Radix primitives for dialogs, menus, tabs, tooltips, and selects; Phosphor for icons; Recharts for responsive analytical charts; TanStack Table for the event explorer; and Zod for runtime artifact validation.

The shell uses a 224px collapsible sidebar and a 64px topbar. Below 1024px, navigation moves into an accessible modal drawer. Page content uses a dashboard-specific maximum width while preserving the source design system's spacing and component geometry.

## Data flow

`pnpm data:sync` reads ignored outputs and creates a versioned normalized snapshot plus the browser-compatible source-video asset. It never writes to the Python output directories.

The runtime consumes an `AnalysisRepository` interface:

```ts
interface AnalysisRepository {
  listRuns(): Promise<AnalysisRunSummary[]>;
  getRun(id: string): Promise<AnalysisRun>;
  getVideoUrl(id: string): string | null;
}
```

The committed static implementation is replaceable by a future API implementation without changing route components. Run and event selections use URL search parameters; playback time remains local to the video component.

## Information architecture

| Route | Primary task |
| --- | --- |
| `/overview` | Understand the selected run and current match state |
| `/matches` | Browse real analysis runs and available capabilities |
| `/analysis` | Inspect synchronized video, overlays, and event timeline |
| `/court` | Explore canonical court trajectories and line evidence |
| `/players` | Compare Player 1 and Player 2 movement analytics |
| `/ball` | Inspect tracking quality, confidence, speed, and flight segments |
| `/events` | Filter events and open technical evidence details |
| `/line-calls` | Review line decisions, uncertainty, and court-line geometry |
| `/reports` | Export loaded data and prepare a printable summary |
| `/system` | Inspect model, configuration, schema, and artifact metadata |

## Core components

- App shell, sidebar, topbar, run selector, mobile drawer, breadcrumbs
- Button, icon button, card, badge, tabs, segmented control, input, select, tooltip, dialog, drawer
- Metric display, technical label, status treatment, empty state, error state, skeleton, notice
- Tennis court SVG, movement trail, occupancy map, trajectory layer, line-evidence zoom
- Video player, synchronized canvas overlay, playback controls, overlay toggles, event timeline
- Responsive chart frame, tracking-state distribution, speed/confidence chart, player comparison
- Event data table, mobile event records, detail drawer, review-required panel

## State and behavior

- Lazy-load route modules and show shape-matched skeletons.
- Validate data on repository load and route validation failures to a calm retryable error state.
- Treat missing capabilities as unavailable, not zero.
- Use text, icon, border style, and geometry for scientific decisions; never rely on red/green alone.
- Draw continuous video overlays outside React rendering through `requestVideoFrameCallback`, with cleanup and a `timeupdate` fallback.
- Disable unsupported PDF generation and manual-review submission with an explanatory status.
- Enable JSON, CSV, and browser print actions using only loaded frontend data.

## Responsive rules

- 1024px and above: persistent sidebar and desktop analytical grids.
- 768px to 1023px: navigation drawer, two-column layouts where useful, reduced chart ticks.
- Below 768px: single-column pages, stacked event records, two-row video controls, full-width touch targets, no page-level horizontal scrolling.
- Tables may scroll only inside a labeled table region on intermediate widths; phone layouts use record cards.
- Court and video preserve their aspect ratios at every width.

## Verification

- ESLint, TypeScript, Vitest, production build, route smoke tests, and output-size report.
- Component accessibility checks with axe-core and keyboard interaction tests.
- Visual QA at every user-specified viewport using the in-app browser.
- Mechanical scans for old accent values, accidental chromatic colors, visible em dashes, overflow, duplicate styling, and missing states.
- Repeat the full and dependency-light Python regression commands without altering Python dependencies.

## Commit sequence

1. `docs(frontend): record audit and implementation plan`
2. `style(frontend): migrate accent tokens to signal lime`
3. `feat(frontend): add application foundation and artifact adapter`
4. `feat(frontend): implement analytics workspace`
5. `test(frontend): verify dashboard and document handoff`
