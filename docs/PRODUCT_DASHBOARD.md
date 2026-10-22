# Tennis intelligence dashboard

The dashboard preserves Astra's existing colors, typography, icons and analytical components while separating user matches from historical examples.

## Main journeys

- **Home** (`/overview`) offers Live match analysis and Upload match video, with active sessions, recent results and actual API/worker status.
- **Match Center** (`/match-center`) searches user matches by participant, tournament or filename and filters live, processing, completed and failed records.
- **Match workspace** (`/matches/:matchId`) contains Replay/Live score, Statistics, Insights, and Details & exports. Analysis tools are scoped to the match. Existing analysis-ID links resolve to this workspace and retain selected events.
- **Players** contains explicitly created profiles, linked histories and comparisons. Identical names do not merge profiles. Video movement history requires both an explicit profile link and a whole-clip role confirmation.
- **AI Insights** leads to evidence-backed match observations and persistent notes. Coaching uses the existing server-configured adapter. Predictions remain unavailable.
- **Examples** is the separate collection of real historical pipeline artifacts. Examples do not contribute to user match statistics.
- **Settings** reports service configuration, upload limits and browser replay preferences. The original recorded pipeline remains experimental.

## Upload and identity

Choose an MP4, MOV or AVI, then optionally enter participant names, link or create profiles, enter a tournament and the actual match date. Singles is the supported analysis format. Doubles is visibly unavailable before submission.

The product creates a draft match before uploading and links the accepted job transactionally. Uploads from older clients also receive match records. Cancelled drafts do not appear as matches in the library. Reusing a linked draft cannot start a duplicate job.

No identity provider or OCR service is connected. The current source therefore supplies no automatic identity suggestions. Unknown names stay unknown; the product never interprets a filename as an identity. Metadata can be corrected without inference or artifact changes.

Near/far tracks are not personal identities. Participants in the match header are separate from tracked roles. Only confirm same/swapped roles if they remain consistent through the entire clip. Save participant changes before confirming their roles; changes clear earlier confirmation. Leave confirmation unset for end changes and camera/role switches. Such clips remain available as role-based analysis but cannot enter named movement trends.

## Manual live scoring

Manual matches require an initial server and start at 0–0. Best-of-three is the default; best-of-five is supported. Existing deterministic scoring rules provide advantage games, seven-point tiebreaks at 6–6, server rotation, set completion and match completion.

The operator records each point winner. Undo removes the latest point; correction changes an earlier winner and replays subsequent scoring. A correction that would leave extra points after match completion is rejected with instructions to undo those points first. Pause blocks new points until resumed. Point-history timestamps describe entry time, not video time.

Commands include an idempotency key and expected revision. An uncertain request can be retried with the same payload without awarding another point. A conflicting update requires refreshing. Score history and metadata persist in SQLite; live scoring is independent of video inference and survives web-server restarts.

Only supported statistics are shown: recorded points won and service/return points won against logged opportunities, games, sets and known results. No serve type, ace, rally length, shot pattern or causal tactical claim is inferred from a point-winner log.

## Storage and API compatibility

SQLite migration version 3 retains version 2 matches, player profiles and score-command history and adds ownership, attempt/heartbeat and cancellation fields. Existing uploads and profiles keep the local owner; signing up does not claim them. Accounts and resumable transfer/batch tables persist alongside the jobs. Back up `outputs/product` before moving to another host; use the updated server/worker together. There is no downgrade migration.

New routes:

| API | Purpose |
| --- | --- |
| `GET/POST /api/matches` | List user matches or create upload/manual records |
| `GET /api/matches/{id}` | Metadata, processing/score state, notes and availability |
| `GET /api/matches/by-analysis/{id}` | Resolve existing analysis links |
| `PATCH /api/matches/{id}/metadata` | Revision-checked metadata corrections |
| `PUT /api/matches/{id}/notes` | Revision-checked persistent user notes |
| `POST /api/matches/{id}/score-commands` | Idempotent point/undo/correct/pause/resume commands |
| `GET/POST /api/players` | Explicit profile listing and creation |
| `GET /api/players/{id}` | Linked matches and confirmed video movement estimates |

The existing upload endpoint accepts optional `match_id`; analysis status, result and media APIs remain compatible. Scientific output is immutable under metadata corrections. API health now indicates whether the worker lock is held, which is availability evidence rather than a promise of progress or accuracy.

## Deliberate limits

Single-host operation supports account ownership and persistent resumable batches; see [the workflow and access configuration](PRODUCT_WORKFLOW.md). Public deployment capacity, quotas and account recovery remain unqualified. There is no connected sports feed, genuine live video inference or OCR. Model qualification, full doubles tracking, long-match capacity and cross-camera identity continuity are not established by software tests or short sample verification. Browser preferences fall back to defaults if storage is unavailable.

Legacy illustrative page source is retained for compatibility/history but is no longer routed into the product. The reference-image carousel and fabricated claims are excluded from the production navigation and build entry points.
