# T88J709 Autonomous Agent Bundle

This folder is intended to be placed at the root of the T88J709 tennis-analysis repository or provided to a cloud coding agent with access to that repository.

## Files

- `MASTER_PROMPT.md` — copy/paste this as the primary instruction to the coding/cloud agent.
- `AGENT.md` — persistent engineering rules, research standards, architecture constraints, evaluation requirements, cloud-training rules, and definition of done.
- `PLAN.md` — phased execution checklist from repository audit through datasets, training, integration, evaluation, and handoff.
- `VIDEO_REFERENCE.md` — structured extraction of the supplied YouTube tutorial plus the specific weaknesses that the final system must improve.
- `PROJECT_REPORT.md` — full Markdown conversion of the supplied FYP DOCX report.
- `report_assets/` — images/media extracted from the DOCX and referenced by `PROJECT_REPORT.md`.

## Recommended usage

1. Copy the entire folder contents into the project repository root.
2. Give the agent `MASTER_PROMPT.md` as its primary prompt.
3. Tell it to obey the read order in `AGENT.md`.
4. Keep `PROJECT_REPORT.md` and `report_assets/` together so report image references remain valid.
5. Let the agent start at Phase 0 of `PLAN.md`; do not begin expensive GPU training before the repository/data audit gates pass.

## Design intent

The YouTube implementation is treated as a reproducible baseline, not the final architecture. The final agent is required to verify current datasets/licenses, prevent data leakage, benchmark modern YOLO/temporal ball tracking and court-registration approaches, use real video timestamps for speed, preserve uncertainty for interpolated/near-line events, and report only measured held-out metrics.
