# Ball candidate coverage and lower-threshold experiment

Status: validation diagnosis and experiment, not independent qualification.

## Measured bottleneck

On the twelve-clip expansion at the selected original checkpoint, overlap step one and heatmap threshold .20, the correct location appears among any candidates on 508/543 visible labeled frames. The selected stationary filter retains those same 508 possibilities; top-one ranking selects 505 correctly. Thus perfect label-assisted ranking could recover only three additional detections, with a 93.55% recall ceiling. Thirty-five visible labels have no candidate within the four-reference-pixel tolerance. No stationary-filter removal accounts for these misses in this particular subset.

There are 57 explicit absent labels. Raw candidates exist on 30; filtered candidates remain on 21. Candidate oracle recall cannot establish precision or recognize absence. Oracle access is limited to this diagnostic and is never used to produce pipeline predictions.

Of the 35 missing-candidate frames, 11 have no raw candidates at all. Only one has any candidate within eight reference pixels; the remainder are larger detection failures. Thirteen occur in `match148_000`, six in `match150_000`, five in `match144_000`, and eleven across five other clips. Replacing mass ranking with peak confidence on the same filtered candidates gives TP=506, FP=40, FN=37 (P=92.67%, R=93.19%, F1=92.93%). The one-detection gain is insufficient to resolve the measured bottleneck and is not promoted.

Evidence: `artifacts/validation/vision_upgrade/expansion12_original_candidate_coverage.json`. Its source report hash, evaluator/filter/metric source hashes and every visible-frame category are retained. Three synthetic regression cases verify ranking versus detector misses, explicit absence and real-candidate suppression by the stationary filter.

## Next experiment, specified before its results

Keep the official checkpoint, the same twelve clips, native frame rates, overlap step one and four-pixel/512x288 matching rule. Lower the heatmap threshold to .05 to test whether missed candidates emerge. Save full-frame candidates and labeled-frame heatmaps. Score raw top one, then replay the already selected stationary filter at .25 seconds with all other settings unchanged. No gap filling or ground-truth-assisted candidate selection is allowed.

The .05 threshold is motivated by the earlier six-clip recall/precision tradeoff and the .20 expansion candidate ceiling. This is further selection on reused validation, not a frozen final test. A gain in recall alone is insufficient for promotion: compare precision, F1, explicit-absence specificity, per-clip errors and the availability of correct candidates. If the result loses precision, report that failure; do not call additional candidate coverage deployed accuracy.

Planned raw report: `artifacts/validation/vision_upgrade/racketvision_expansion12_original005.json`. Runtime with compressed heatmap writes includes that extra work and must not be compared as pure inference throughput to a run without caching.

## Bounded recovery hypothesis, specified before replay

The completed .05 run raises candidate coverage to 526/543 (96.87%), but its filtered top one gives TP=500, FP=96, FN=43, P=83.89%, R=92.08%, F1=87.80%; 54/57 absent labels become false detections. It is rejected as a replacement. There are 26 correct lower-ranked candidates and 17 visible labels without any correct candidate.

Test only short missing intervals in the existing .20/filter output. Both bounding observations must have peak confidence at least .50, their total span must be at most .20 s, and their average image motion must be 20–1200 reference pixels/s. Inside such gaps, choose a real .05 candidate within six reference pixels of the linear anchor guide. Never replace an existing .20 observation, extrapolate at the ends, or output the interpolated guide. Keep .25 s stationary filtering on both streams. These fixed settings are a conservative recovery hypothesis, not physical ball-speed limits or a tuned grid.

This uses future anchors and is offline. False anchors can cause false recovery, and curved flight/contact may fail the guide. Both input reports must match checkpoint, dataset, frame ordering and labels. The predicted coordinates never use labels. Results remain validation selection; no deployment is authorized by an oracle ceiling or a synthetic recovery test.

## Completed recovery result

At the predeclared settings, zero candidate observations are recovered across all 6,082 frames. Counts remain exactly TP=505, FP=41, FN=38, TN=36: P=92.49%, R=93.00%, F1=92.75%. This conservative rule is ineffective on these clips and is not integrated into the pipeline. Its first replay attempt rejected a nonexistent report field before writing results; the evaluator was corrected to compare actual candidate-sequence lengths. The successful immutable report is `artifacts/validation/vision_upgrade/expansion12_anchored_recovery.json`.

Ten targeted tests pass across candidate coverage and anchored recovery. They verify actual-candidate identity, no synthesized output, rejection of long gaps/weak or stationary anchors, ranking/missing/absence accounting and duplicate-label rejection. They do not establish that recovery succeeds on real video. The lower-threshold and recovery hypotheses have not improved deployed measurements; the selected .20 original-model configuration remains unchanged.
