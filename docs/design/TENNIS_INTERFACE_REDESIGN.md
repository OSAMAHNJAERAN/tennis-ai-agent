# Astra Tennis interface redesign

Design handoff · 6 October 2026

The frontend now uses a paper and electric-blue editorial direction across the home page, navigation, upload flow, saved analyses, replay workspace and analytics. The existing tennis analysis service, data contracts and saved results remain the source of product behavior and measurements.

## Reference and interpretation

The primary reference was [The Design Society](https://www.thedesignsociety.fr/), examined across its home, events and about pages. The useful principles were generous editorial spacing, oversized display typography, bracketed navigation, fine rules, strong blue fields, lime accents, scanline lettering and deliberate scroll motion.

Those principles were adapted to a tennis product with original copy and court artwork. The reference site's branding, commercial fonts and media were not incorporated. The design uses locally bundled, openly licensed Geist Variable, DM Serif Display and Geist Mono Variable. `CourtEmblem` is an original decorative SVG, separate from measured court geometry and replay evidence.

| Token | Value | Use |
| --- | --- | --- |
| Paper | `#efeeec` | Page canvas and chart tooltip |
| Ink | `#1e1e1e` | Primary text and structural rules |
| Blue | `#350aff` | Primary action, court artwork and Player 1 |
| Lime | `#bcf015` | Completed states and graphic accents |
| Muted | `#62615e` | Supporting interface copy |
| Line | `#cbc9c4` | Dividers and form boundaries |
| Player 2 | `#9b482b` | A distinct analytics series with a dashed line |

Display headings use DM Serif Display; interface text uses Geist; metadata uses Geist Mono. Controls have square edges, clear focus outlines and restrained hover transitions. Wide headings, full rows and responsive gallery counts avoid vacant grid cells.

## Implemented experience

- **Home:** An editorial split hero, original court illustration, moving capability strip, real annotated-video previews, inline court artwork, scroll-revealed evidence statement, horizontal workflow accordions, private-library entry and a large upload action.
- **Navigation and footer:** A compact horizontal navigation with bracketed labels, upload action and service settings. Small screens use a menu button with expanded state, route-aware closure and Escape dismissal that returns focus to the toggle. A blue footer carries the original Astra wordmark treatment.
- **Upload:** A large two-line introduction, blue court-line drop zone, prominent file chooser, optional match metadata, processing feedback and session archive. Supporting guidance moves below the upload area on small screens.
- **Saved analyses and workspace:** Editorial list rows, square filters, clear completed/processing states, revised tabs and consistent replay surfaces. Existing section navigation and downloads remain connected to the saved analysis.
- **Analytics:** Large player comparison rows, a custom speed chart with elapsed-video labels and tooltip, proportional ball tracking-state distribution, retained availability messages and the existing spatial heatmap controls.

## Architecture and preserved behavior

The app remains React 19, TypeScript and Vite, connected to the existing FastAPI service through `/api`. Active product routes remain `/overview`, `/upload`, `/match-center`, `/matches/:matchId`, `/examples`, `/examples/:exampleId` and `/settings`; existing aliases continue to resolve through the router.

The design is implemented in `Society.tsx`, `ProductHome.tsx`, `AppShell.tsx`, `Upload.tsx` and `LiveRunAnalytics.tsx`, with shared overrides in `society.css` and scoped upload/analytics styling. The shared stylesheet is loaded after the existing product styles. Replay logic, video synchronization, Three.js court rendering, heatmap computation and the repository/API adapters remain in their existing modules.

Upload still validates the actual file and service limit, creates or updates the draft match with revision checking, reports actual transfer progress, supports cancellation, polls processing status and opens a completed result. No artificial progress or placeholder analysis data were added. Public demonstration analyses are filtered out of private upload history and the user's analysis library.

Analytics continues to render unavailable measurements as unavailable. Null speed intervals remain gaps; interval aggregation requires contiguous observed values. Player numbers identify court roles rather than confirmed named identities. Tracking coverage is explicitly distinct from detection accuracy. Ball speed, bounces and shot statistics remain unavailable unless the pipeline supplies qualified measurements.

## Motion and accessibility

GSAP and `@gsap/react` drive masked hero entrances, section entrances, a scrubbed word-opacity reveal and desktop workflow-title pinning. The animation scope is stable across resource polling. A debounced resize observer refreshes ScrollTrigger after content-height changes; GSAP context, media queries, observer, timer and animation frame are cleaned up on teardown.

Reduced-motion preferences disable decorative CSS loops and GSAP motion. The moving capability strip exposes a Pause/Resume control for decorative graphics. Preview videos use visibility-aware playback, defer non-priority loading, stop when the page is hidden or the preview leaves view, honor reduced motion and retain a user's manual pause. Failed autoplay exposes an explicit play action.

Accordion controls expose expanded state and controlled panels. Navigation includes a skip link and visible keyboard focus. Upload errors use alerts, transfer progress uses a status region, and analytics tooltips announce changes politely. Chart series differ by color and line style, with a descriptive figure caption and accessibility support.

## Media and attribution

Existing source credits remain attached to the public previews and analysis views. Source metadata and transformations are recorded in `data/demos/sources.json`.

| Existing demo | Creator | License and source |
| --- | --- | --- |
| Djokovic point, 2012 US Open | JC | [CC BY-SA 2.0 · Wikimedia Commons](https://commons.wikimedia.org/wiki/File:2012_US_Open_Djokovic_point.webm) |
| Schiavone rally, 2012 US Open | JC | [CC BY-SA 2.0 · Wikimedia Commons](https://commons.wikimedia.org/wiki/File:2012_US_Open_Schiavone_rally.webm) |
| Dimitrov and Tiafoe, 2024 US Open | Oleg Yunakov | [CC BY-SA 4.0 · Wikimedia Commons](https://commons.wikimedia.org/wiki/File:2024_US_Open._Grigor_Dimitrov_vs._Frances_Tiafoe.webm) |

The first two analyses use the recorded source clips without trimming. The third uses the existing ten-second excerpt from 6.5 to 16.5 seconds. Analysis derivatives omit audio and add tracking/court/pose overlays as documented in the source registry. These are diagnostic product demonstrations, not model qualification evidence.

## Verification record

| Check | Result |
| --- | --- |
| Full frontend test suite | Passed: 23 files, 165 tests |
| Frontend lint | Passed |
| Production build and TypeScript | Passed: `pnpm build` completed with the final changes |
| Home, upload and analytics responsive inspection | No horizontal overflow at 1920, 1440, 1280, 1024, 768, 430 and 390 pixels |
| Saved-analysis browser checks | Frame stepping, 2D/3D court views, camera changes and heatmap filters checked against real saved results |
| Export and production console | Readable HTML report downloaded successfully; no warnings or errors logged during the production-browser checks |

The built application is available at [the local production preview](http://127.0.0.1:8000/overview). It is served by the existing product Python environment. Saved screenshot evidence includes the home page at desktop and mobile sizes, upload, replay, and analytics at desktop and mobile sizes.

Browser checks exercised existing results and product controls. They do not establish new ML accuracy, independently validate measurements or constitute a fresh end-to-end processing benchmark. No backend or ML implementation changes were made as part of this redesign. The repository had extensive pre-existing changes, which are outside this handoff's attribution.

## Visual evidence

The handoff screenshot set is stored alongside this document:

- [Home, desktop](screenshots/home-desktop.jpg)
- [Home, mobile](screenshots/home-mobile.jpg)
- [Upload, desktop](screenshots/upload-desktop.jpg)
- [Analytics, desktop](screenshots/analytics-desktop.jpg)
- [Replay, desktop](screenshots/replay-desktop.jpg)

For a local preview, start the existing API service and run `pnpm dev` from `frontend`, then open the `/overview` route. The frontend uses the Vite API proxy; processing and saved evidence depend on the configured local service and artifacts.
