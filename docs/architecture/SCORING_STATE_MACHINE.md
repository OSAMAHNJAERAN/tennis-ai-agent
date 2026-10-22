# T88J709 Phase 5 — Scoring State Machine Architecture

## 1. Architectural Principles
1. **Rule-Driven Determinism**: State transitions are pure mathematical functions of verified semantic events and current state.
2. **Numeric Representation**: Points are stored as integers ($0, 1, 2, 3, 4, 5...$) and rendered dynamically to ITF strings ("Love", "15", "30", "40", "Deuce", "Ad").
3. **Dead-Ball Suppression**: Once an event terminates a point or causes a first fault, the ball state transitions to `DEAD`, immediately suppressing any subsequent physical detections until the next serve.
4. **Uncertainty Propagation**: Any line call or event marked `REVIEW_REQUIRED` pauses scoring in `POINT_REVIEW_PENDING` without awarding points.

## 2. Rules and review changes (8 October 2026)

The isolated rules engine supports singles, best of three/five, standard advantage games, advantage sets, seven-point set tie-breaks and configurable ten-point set tie-breaks. A deciding set may use a separately declared set format/target. It does not implement a match tie-break replacing a set, doubles or no-ad games. These formats require explicit future implementation, rather than being silently mapped to standard singles rules.

Changes fix two discovered transition errors: a new service contact now starts after the dead-ball state from a first fault or completed point; and the first server of a tie-break receives in the next set regardless of who served its final point. Service lets preserve the current attempt. Changes of ends follow odd-numbered games, every six tie-break points and tie-break completion. Court ends belong to stable participant IDs, rather than assuming player two is always far from the camera. Rules were checked against [ITF Rules of Tennis 2026, rules 5–7, 10, 20 and 22](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf).

`TennisScoringEngine.process_match_event` accepts an optional stable `point_id` alongside source event IDs. A completed point cannot score twice under different events. Invalid/out-of-order timestamps and incompatible participants are rejected. A review or `CONSEQUENTIAL_GAP` freezes all subsequent confirmed score transitions. `correct_event` verifies the expected revision, rebuilds all dependent event transitions atomically from the preserved initial state and retains the previous history and human-review reason.

`PointLedger` is the integration contract for confirmed video point outcomes and human review. Each `PointRecord` contains a stable ID, source seconds, winner or unknown, confirmation status, provenance, evidence IDs and optional terminal loss reason. A missing/provisional point freezes later score and prediction updates. Correction preserves the point ID, increments the revision, records before/after evidence and recomputes all dependent prefixes. The default initial state is unconfirmed; `confirm_initial_state` requires provenance and a reason. A clip's final rally is not a match result. Unconfirmed initial scores withhold a full-match winner, and retired/suspended matches withhold normal rules-completion forecasts.

## 3. Statistical probability contract

`src/prediction/tennis.py` implements a statistical component without an additional vision model. Each participant's serve-point probability starts from an explicit neutral Beta(2,2) engineering prior. Confirmed completed points update only their server's successes/trials. The posterior mean is a shrinkage estimate. Standard game, set and match probabilities propagate the selected scoring format and service order; infinite deuce, tie-break and advantage-set tails use analytical solutions. Assumptions are independent points and stationary serve success. The plug-in posterior mean does not integrate posterior uncertainty and is not an empirically calibrated win forecast.

`replay_predictions(initial_state, points, revision=0)` returns `{current, history, state}`. Each current/history forecast carries the match/revision, score state version, source timestamp, sample count, prior parameters, serve estimates, assumptions, reason, and `calibration_status="not_calibrated"`. Available probabilities are `probabilities.{game,set,match}.{player_1,player_2}`. Status is one of:

- `prior_only`: no completed performance points; the score and explicit neutral prior determine the displayed probability.
- `uncalibrated`: previous confirmed points update the statistical estimate; independent calibration is still pending.
- `unavailable`: unknown initial state, consequential gap, review, invalid score, suspension or retirement.
- `confirmed_result`: confirmed rules completion; probabilities are null and `winner_id` supplies the actual result.

The replay uses each prefix only. It never supplies the terminal result, full-match statistics, video frames or future events to a previous forecast. A timestamp before the last consumed point is rejected. Correction rebuilds subsequent scores and forecasts from the corrected prefix. Source seconds are required for video/evaluation; legacy/manual score rows without video timestamps explicitly use `time_base="point_sequence"` in history.

The product API connects this contract to manual scoring in `src/api/product.py`. `predictions` includes current values and a revision-matching causal history. New manual points use the score command ID as their stable point ID. Corrections keep the ID and append previous-winner/revision metadata under `provenance="human_review"`. Library summaries omit histories. Manual point data are separate from automatic vision evidence. Uploaded-video probabilities remain unavailable until confirmed initial state and qualified point outcomes are connected; rules tests cannot qualify automatic scoring or prediction calibration.

## 4. Independent probability evaluation

Run the evaluator after obtaining independently reviewed point matches with evaluation rights:

```powershell
python scripts/evaluate/evaluate_tennis_prediction.py --manifest data/tennis_prediction_matches.json --output artifacts/validation/prediction_independent_report.json --bootstrap-samples 1000 --bins 10 --seed 20261008
```

The input contract is `schema_version="1.0"`, `usage_rights="authorized_for_evaluation"`, and a `matches` array. Each record requires unique `match_id`, `source_group_id`, ISO `match_date`, `split` (`development`, `calibration`, `test`), explicit `initial_state` with format/server and numeric score, `ground_truth` with `status="independently_reviewed"`, reference source and winner ID, plus continuous `points`. Each point requires unique stable `point_id`, finite original `timestamp_seconds`, `winner`, `status="confirmed"` and manual/human-review provenance. Point records must reach the independently confirmed match result. Initial-state fields are allowlisted, preventing a result field from becoming a forecast feature. Calibration/test dates and source groups must be disjoint and chronological. Usage-rights/provenance declarations must still be audited against the actual source; validation does not establish legal permission or annotation independence by itself.

The frozen baseline uses the same causal score/server with fixed neutral Beta(2,2) serve probabilities and no performance updates. Final outcome snapshots are excluded. Reports retain every pre-outcome forecast, exact manifest and implementation SHA256 hashes, match/snapshot counts, equal match weighting, Brier loss, log loss, reliability bins and deterministic match-group bootstrap intervals (including reliability). Snapshots from a long match cannot outweigh an independent short match. Proposed engineering gates, frozen before acquiring independent data, are at least 20 final-test matches, positive lower 95% confidence bound for Brier improvement, no worse point-estimate log loss than the baseline, and reliability gap at most 0.10 in bins supported by at least five independent matches. A sparse bin remains unqualified rather than becoming evidence of calibration. Command settings and gates must be frozen before a final test and not changed to pass an observed failure.

The command evaluates calibration; it does not fit or certify it. Output remains `not_fitted_or_qualified` even when numerical diagnostics look promising. Independent calibration fitting, source rights/ground-truth audit, sample adequacy and final-test review remain necessary. Synthetic software fixtures are not accuracy results.

Current software evidence: 73 scoring/prediction/evaluator checks passed in 1.48 seconds, recorded in `artifacts/validation/yolo26_execution_2026-10-08/scoring_prediction_evaluation.xml`. They cover faults/lets and dead-ball controls, review freeze and correction replay, 7/10-point tie-break service order, changes of ends, extended advantage scoring, prefix causality, analytical probability identities against independent expressions/forward enumeration, session isolation and evaluator leakage/manifest guards. Product API tests are recorded separately. No independent match prediction calibration, automatic point accuracy or long/live scoring qualification is claimed.
