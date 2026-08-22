# Frontend Delivery Notes

## Delivered boundary

The frontend is a greenfield Vite, React, and TypeScript application in `frontend/`. It uses a replaceable offline repository with these methods:

```ts
listRuns(): Promise<AnalysisRunSummary[]>
getRun(id: string): Promise<AnalysisRun>
getVideoUrl(id: string): string | null
```

`pnpm data:sync` reads existing ignored output artifacts and the authorized source video. It does not write to, modify, or delete source artifacts. It produces a committed snapshot with summaries for all seven discovered generations, normalized historical metadata, full Phase 5 frame details, and a browser-compatible H.264 demo video.

## Implemented views

- Overview
- Matches
- Match analysis
- Court view
- Players
- Ball tracking
- Events
- Line calls
- Reports
- System

Every route is lazy loaded. The player draws current-frame evidence and a short ball tail through `requestVideoFrameCallback`, with cancellation during cleanup. Selected run and event state are shareable URL search parameters; continuous playback time stays local.

## Known frontend-only limitations

- The snapshot is static and does not update while the app is open.
- Only the Phase 5 source video is bundled for synchronized replay.
- Historical frame arrays are intentionally omitted from the committed browser snapshot to keep the production payload bounded; their summaries, normalized metadata, and artifact availability remain visible.
- Ball speed is an experimental two-dimensional court-plane estimate and does not account for monocular vertical elevation.
- Server PDF generation, backend search, live processing, and database-backed workflows are disabled because no such frontend integration exists.
- The full Python regression suite requires the separate ML environment with pandas, OpenCV, and Torch.

## Commands

From `frontend/`:

```text
pnpm install
pnpm data:sync
pnpm dev
pnpm check
pnpm bundle:report
```

No deployment or push is part of this delivery.

