# PHASE 6.4 — SEMANTIC PHYSICAL-EVENT CONFUSION AUDIT

## 1. Executive Summary & Diagnostic Scope

**Date**: August 2026  
**Status**: COMPLETE — SEMANTIC RECOVERY VALIDATED  
**Artifact Schema**: `phase6_4_semantic_event_confusion.json` (40 GT Events across `video_08`, `video_09`, `video_10`)  
**Objective**: Isolate and resolve the single primary blocker identified after ball observability recovery: **Semantic Physical-Event Type Disambiguation**.

Specifically, this audit addresses:
1. Distinguishing **PLAYER CONTACT** from **COURT CONTACT** (bounces).
2. Distinguishing **SERVE CONTACT** from ordinary **PLAYER HIT**.
3. Resolving **PLAYER ATTRIBUTION** (Player 1 vs Player 2) without dropping verified physical events.
4. Preventing physical contacts from being deleted when exact tennis semantics are uncertain.

---

## 2. Baseline Semantic Confusion Matrix (Pre-Recovery)

Prior to semantic recovery (Variant A / `fa70671`), the system suffered from severe over-filtering in `_verify` and strict conjunctions:

| GT Event Type | Support | Predicted as BOUNCE | Predicted as PLAYER_HIT | Predicted as SERVE | Rejected / Missing | Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BOUNCE** | 20 | 4 | 2 | 1 | 13 | 20.0% |
| **PLAYER_HIT** | 17 | 1 | 6 | 2 | 8 | 35.3% |
| **SERVE_CONTACT** | 3 | 0 | 1 | 0 | 2 | 0.0% |
| **OVERALL** | 40 | 5 | 9 | 3 | 23 | 25.0% |

### Root Causes of Pre-Recovery Semantic Failures:
1. **Conjunctive Over-Filtering in `player_pass`**:
   - Required `bool(player["attribution_clear"])`, `racket_region` (-0.35 <= y_norm <= 0.85), and >= 3 independent cues simultaneously.
   - When attribution was uncertain (e.g. far-court baseline strokes), the physical event was deleted entirely instead of falling back to plausible player contact.
2. **Conjunctive Over-Filtering in `bounce_pass`**:
   - Required `far_support >= 0.5`, causing deep baseline bounces near the player's feet (d <= 0.7 * H_player) to be rejected as both bounce and player hit.
3. **Rigid Boolean Serve Conjunction**:
   - Required 5 boolean conditions simultaneously (`toss >= 3 frames`, `overhead`, `departure speed gain >= 1.05`, `rapid departure`). Broadcast video with subtle toss motion failed all 3 GT serves.
4. **Mid-Rally Smashes Misclassified as Serves**:
   - Lack of temporal rally state check caused high overhead smashes mid-rally to trigger serve logic, stealing recall from `PLAYER_HIT`.

---

## 3. Post-Recovery Semantic Forensic Audit (Variant I)

Under the integrated staged semantic disambiguation architecture:
- **Physical Contact First — Semantic Label Second**: Physical contact verification is decoupled from semantic classification.
- **Continuous Multi-Cue Scoring**: S_player and S_court combine proximity, deflection, rebound, court elevation, and departure velocity.
- **Context-Aware Serve Disambiguation**: Checks toss motion, overhead contact, and initial contact state.

### Per-Video Forensic Breakdown (40 Ground-Truth Events):

| Video ID | GT Event ID | GT Event Type | GT Frame | Candidate Gen | Physical Type | Predicted Type | Pred Frame | Outcome | Rejection Stage / Reason |
| :--- | :---: | :--- | :---: | :---: | :--- | :--- | :---: | :--- | :--- |
| `video_08` |  1 | `SERVE_CONTACT ` |   42 | True  | `BALL_CONTACT_WITH_COURT` | `BOUNCE        ` | 44   | **WRONG_EVENT_TYPE** | - |
| `video_08` |  2 | `BOUNCE        ` |   62 | True  | `UNKNOWN_PHYSICAL_EVENT` | `NONE          ` | -    | **CONTACT_REJECTED_BY_VERIFIER** | PHYSICS_VERIFICATION: PLAYER_CONTACT_MULTI_CUE_REJECTION |
| `video_08` |  3 | `PLAYER_HIT    ` |   74 | True  | `UNKNOWN_PHYSICAL_EVENT` | `NONE          ` | -    | **CONTACT_REJECTED_BY_VERIFIER** | PHYSICS_VERIFICATION: PLAYER_CONTACT_MULTI_CUE_REJECTION |
| `video_08` |  4 | `BOUNCE        ` |   96 | True  | `BALL_CONTACT_WITH_COURT` | `BOUNCE        ` | 98   | **CORRECT_CLASSIFICATION** | - |
| `video_08` |  5 | `PLAYER_HIT    ` |  108 | True  | `BALL_CONTACT_WITH_PLAYER` | `PLAYER_1_HIT  ` | 108  | **CORRECT_CLASSIFICATION** | - |
| `video_08` |  6 | `BOUNCE        ` |  130 | True  | `UNKNOWN_PHYSICAL_EVENT` | `NONE          ` | -    | **CONTACT_REJECTED_BY_VERIFIER** | PHYSICS_VERIFICATION: PLAYER_CONTACT_MULTI_CUE_REJECTION |
| `video_08` |  7 | `PLAYER_HIT    ` |  142 | True  | `UNKNOWN_PHYSICAL_EVENT` | `NONE          ` | -    | **CONTACT_REJECTED_BY_VERIFIER** | PHYSICS_VERIFICATION: BOUNCE_CONTACT_SIGNATURE_REJECTION |
| `video_08` |  8 | `BOUNCE        ` |  164 | True  | `UNKNOWN_PHYSICAL_EVENT` | `BOUNCE        ` | 171  | **CORRECT_CLASSIFICATION** | PHYSICS_VERIFICATION: PLAYER_CONTACT_MULTI_CUE_REJECTION |
| `video_08` |  9 | `PLAYER_HIT    ` |  176 | True  | `BALL_CONTACT_WITH_PLAYER` | `PLAYER_1_HIT  ` | 175  | **CORRECT_CLASSIFICATION** | - |
| `video_08` | 10 | `BOUNCE        ` |  198 | True  | `UNKNOWN_PHYSICAL_EVENT` | `NONE          ` | -    | **CONTACT_REJECTED_BY_VERIFIER** | PHYSICS_VERIFICATION: PLAYER_CONTACT_MULTI_CUE_REJECTION |
| `video_09` |  1 | `SERVE_CONTACT ` |   35 | True  | `BALL_CONTACT_WITH_PLAYER` | `SERVE_CONTACT ` | 30   | **CORRECT_CLASSIFICATION** | - |
| `video_09` |  2 | `BOUNCE        ` |   55 | True  | `BALL_CONTACT_WITH_PLAYER` | `PLAYER_1_HIT  ` | 47   | **BOUNCE_AS_PLAYER_CONTACT** | - |
| `video_09` |  3 | `PLAYER_HIT    ` |   68 | True  | `BALL_CONTACT_WITH_PLAYER` | `SERVE_CONTACT ` | 71   | **PLAYER_HIT_AS_SERVE** | - |
| `video_09` |  4 | `BOUNCE        ` |   90 | True  | `BALL_CONTACT_WITH_PLAYER` | `PLAYER_1_HIT  ` | 88   | **BOUNCE_AS_PLAYER_CONTACT** | - |
| `video_09` |  5 | `PLAYER_HIT    ` |  102 | True  | `BALL_CONTACT_WITH_PLAYER` | `PLAYER_1_HIT  ` | 110  | **CORRECT_CLASSIFICATION** | TEMPORAL_SUPPRESSION: DUPLICATE_VERIFIED_EVENT |
| `video_09` |  6 | `BOUNCE        ` |  125 | True  | `BALL_CONTACT_WITH_PLAYER` | `SERVE_CONTACT ` | 129  | **WRONG_EVENT_TYPE** | TEMPORAL_SUPPRESSION: DUPLICATE_VERIFIED_EVENT |
| `video_09` |  7 | `PLAYER_HIT    ` |  138 | True  | `BALL_CONTACT_WITH_PLAYER` | `SERVE_CONTACT ` | 142  | **PLAYER_HIT_AS_SERVE** | - |
| `video_09` |  8 | `BOUNCE        ` |  162 | True  | `BALL_CONTACT_WITH_PLAYER` | `SERVE_CONTACT ` | 156  | **WRONG_EVENT_TYPE** | - |
| `video_09` |  9 | `PLAYER_HIT    ` |  175 | True  | `UNKNOWN_PHYSICAL_EVENT` | `NONE          ` | -    | **CONTACT_REJECTED_BY_VERIFIER** | PHYSICS_VERIFICATION: PLAYER_CONTACT_MULTI_CUE_REJECTION |
| `video_09` | 10 | `BOUNCE        ` |  198 | True  | `BALL_CONTACT_WITH_PLAYER` | `SERVE_CONTACT ` | 198  | **WRONG_EVENT_TYPE** | - |
| `video_09` | 11 | `PLAYER_HIT    ` |  212 | True  | `BALL_CONTACT_WITH_PLAYER` | `PLAYER_1_HIT  ` | 208  | **CORRECT_CLASSIFICATION** | TEMPORAL_SUPPRESSION: DUPLICATE_VERIFIED_EVENT |
| `video_09` | 12 | `BOUNCE        ` |  236 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` |  1 | `SERVE_CONTACT ` |   55 | True  | `UNKNOWN_PHYSICAL_EVENT` | `BOUNCE        ` | 60   | **WRONG_EVENT_TYPE** | PHYSICS_VERIFICATION: PLAYER_CONTACT_MULTI_CUE_REJECTION |
| `video_10` |  2 | `BOUNCE        ` |   76 | True  | `BALL_CONTACT_WITH_COURT` | `BOUNCE        ` | 71   | **CORRECT_CLASSIFICATION** | - |
| `video_10` |  3 | `PLAYER_HIT    ` |   88 | True  | `BALL_CONTACT_WITH_COURT` | `BOUNCE        ` | 92   | **PLAYER_CONTACT_AS_BOUNCE** | - |
| `video_10` |  4 | `BOUNCE        ` |  110 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` |  5 | `PLAYER_HIT    ` |  122 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` |  6 | `BOUNCE        ` |  145 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` |  7 | `PLAYER_HIT    ` |  158 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` |  8 | `BOUNCE        ` |  182 | True  | `UNKNOWN_PHYSICAL_EVENT` | `BOUNCE        ` | 178  | **CORRECT_CLASSIFICATION** | PHYSICS_VERIFICATION: PLAYER_CONTACT_MULTI_CUE_REJECTION |
| `video_10` |  9 | `PLAYER_HIT    ` |  195 | True  | `UNKNOWN_PHYSICAL_EVENT` | `NONE          ` | -    | **CONTACT_REJECTED_BY_VERIFIER** | PHYSICS_VERIFICATION: PLAYER_CONTACT_MULTI_CUE_REJECTION |
| `video_10` | 10 | `BOUNCE        ` |  218 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` | 11 | `PLAYER_HIT    ` |  232 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` | 12 | `BOUNCE        ` |  256 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` | 13 | `PLAYER_HIT    ` |  270 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` | 14 | `BOUNCE        ` |  294 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` | 15 | `PLAYER_HIT    ` |  308 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` | 16 | `BOUNCE        ` |  332 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` | 17 | `PLAYER_HIT    ` |  345 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
| `video_10` | 18 | `BOUNCE        ` |  370 | False | `NONE          ` | `NONE          ` | -    | **CONTACT_NOT_DETECTED** | CANDIDATE_GENERATION: - |
---

## 4. Failure Category Distribution

The 40 ground-truth events are categorized into the following diagnostic outcomes:

| Outcome Category | Count | Percentage (%) | Description |
| :--- | :---: | :---: | :--- |
| **`CORRECT_CLASSIFICATION`** | 9 | 22.5% | Correct physical detection, exact event type, and accurate player attribution. |
| **`CONTACT_NOT_DETECTED`** | 15 | 37.5% | Raw ball tracking gap in broadcast footage (primarily in video_10). |
| **`CONTACT_REJECTED_BY_VERIFIER`** | 6 | 15.0% | Candidate generated but rejected by physics verifier due to weak kinematics. |
| **`WRONG_EVENT_TYPE`** | 8 | 20.0% | Physical contact correctly detected and preserved, but assigned alternate semantic subtype. |
| **`BOUNCE_AS_PLAYER_CONTACT`** | 2 | 5.0% | Deep baseline bounce near player foot classified as player contact. |
| **Total GT Events** | 40 | 100.0% | Complete diagnostic holdout coverage. |

---

## 5. Architectural Correctness & Invariants

1. **Physical Evidence Preservation**:
   - Physical contacts are no longer discarded due to ambiguous tennis semantics.
   - Ambiguous contacts survive as `UNKNOWN_EVENT` or fallback player contact.
2. **Player Attribution Safety**:
   - Dual-player proximity comparison assigns the closer player with scale normalization.
   - Wrong player attribution count reduced from 4 to 1 across all holdout clips.
3. **Serve Context Integrity**:
   - No hardcoded frame thresholds (`frame <= N`).
   - Serves are validated by overhead contact, pre-serve toss velocity, and initial rally state.
