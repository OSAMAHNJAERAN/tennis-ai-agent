# Visual rejection with preserved detector ordering

Protocol fixed before internal reselection or external scoring. Pilot05's global context head produces 61 wrong visible localizations, 48 with an available correct proposal. Some wrong crops contain the actual ball away from their center. A context classifier score therefore need not identify the best localized candidate.

Reuse all eight frozen pilot05 head checkpoints and the exact training/external feature caches. Change only the selection policy: scan candidates in the original WASB mass order and choose the first whose visual score meets the threshold; emit missing if none passes. Do not rank accepted candidates by the visual score, synthesize coordinates, or change labels. The original array order/rank must be contiguous and consistent.

Select checkpoint and threshold afresh on the same four internal-selection training clips only, using F1 then precision then recall, eight epochs and thresholds .3/.5/.7/.9. Save the frozen selection before external evaluation. This is another experiment on reused development evidence, not an independent holdout or externally selected threshold rescue. Report paired changes against pilot05 and the previous CNN control. No automatic integration or production promotion.


## Completed result

Internal reselection chooses epoch 8 / threshold 0.3, with precision 95.29%, recall 89.01% and F1 92.05%. The selected checkpoint and threshold happen to match pilot05. External candidate scores were verified exactly equal between the two reports, so their external difference is solely the candidate-order decision rule.

External counts are **476 TP, 57 FP, 67 FN and 35 TN**, precision **89.31%**, recall **87.66%**, F1 **88.48%**. Relative to visual-score ranking, 28 visible frames become correct and two lose a correct detection, for a net 26 gain; wrong visible localizations fall from 61 to 35. Explicit-absence outcomes remain unchanged, with 22 false detections and 35 correct absences. Compared with the matched small-CNN control, five more true positives come with 29 more false positives. This is not a successful production replacement.

The experiment confirms that global context scores should not automatically rank precise candidate centers, but original detector ordering still makes errors and the context model still misses balls. Both pretrained-context variants are rejected for production. No labels, default configuration or physical/event outputs changed.

Nine targeted tests passed in 4.58 seconds for accepted ordering, rejection, explicit missing output, score/order validation and point metrics. Nineteen earlier feature/cache/verifier tests passed. All eight head checkpoints, both complete feature-cache hashes, source/encoder code hashes and saved selection provenance were reverified.

Frozen selection and external report: `artifacts/validation/vision_upgrade/verifier_accepted_order_pilot06/selection.json` and `external.json`. Exact paired changes, source hashes and final verification: `outputs/vision_upgrade_audit/verifier_context_pilot05/final_provenance.json`. This is reused development evidence, not independent qualification.
