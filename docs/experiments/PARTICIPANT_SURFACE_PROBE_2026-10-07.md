# Participant surface-support diagnostic — 7 October 2026

Decision: keep the fixed surface-only filter and its intersection with the rejected motion selector out of production. The actual overlays admit on-grass officials/crew, exclude a valid boundary player and sometimes provide no usable mask. This is development evidence, not scientific qualification.

The single completed run used the thirteen already reviewed native stills from the UVY motion failure experiment. It reused cached tracked detections and the unchanged surface extractor; it did not decode video, run models, train, alter thresholds, rasterize a convex hull, merge masks or assign participant identities. Every pixel decision was saved before publisher labels were parsed. The fixed fifteen-point bbox-foot patch used nearest-integer reduced pixels, required all samples in bounds, and required at least eight supported pixels. No eligible component and out-of-bounds patches abstained.

| Fixed diagnostic group | Proposals | Original class1 matches | Publisher-unmatched | Original class1 misses |
| --- | ---: | ---: | ---: | ---: |
| Cached detector | 128 | 13 | 115 | 11 |
| Cached rejected motion selector | 14 | 4 | 10 | 20 |
| Surface supported detector proposals | 24 | 12 | 12 | 12 |
| Motion and surface intersection | 6 | 4 | 2 | 20 |

These are counts on thirteen retrospective frames, not full-clip metrics. The fixed matcher uses IoU 0.5. Publisher-unmatched includes visibly valid players whose boxes miss that threshold, unlabeled people and officials; it does not establish false-positive counts or participant precision. Known V01 frame604 missing labels and V03 frame140 chair-umpire class1 labeling are retained.

Actual visual review found the one lost match at V01 frame409, where the right player falls outside the selected surface. V01 frame818 supports a player plus four nearby crew/official fragments; two crew boxes also match original class4 labels. V02 frame99 supports both players plus three other people, including two original class4 matches. V03 frame140 supports the near player and a boundary official matching class4 identity5. V03 frame0 has no eligible component despite both players being visible. V03 frame442 supports both players while motion selects neither. These failures show why surface context can contribute evidence but cannot establish participant identity or repair missing detector proposals.

Of 128 proposals, 24 were supported, 25 rejected, 75 unknown because a sampling patch crossed image bounds, and four unknown because no surface component existed. Four proposals repeat reduced-grid pixels; multiplicity is retained and does not imply fifteen independent pixel samples. Roles remain null and participant status UNKNOWN throughout.

The actual guarded trial completed in 5.8877039 seconds including process overhead. Its twenty monitor samples observed peak owned RSS 140,779,520 bytes and minimum available system memory 2,564,644,864 bytes. The workload and its exact owned native console were retired. This small still-image diagnostic does not measure model throughput or full-match capacity.

An independent stdlib audit decoded the 0/1 grayscale masks without rerunning the extractor, reconstructed every sampling location/status, independently matched all groups, and checked thirteen native images, 181 input bindings, 643 loaded module files, staged semantic read order, resource acknowledgements and process retirement. Ten saved product jobs and 319 product files were preserved. Audit receipt: artifacts/validation/product_execution/participant_surface_probe_2026-10-07/audit/receipt.json. Root visual notes and image bindings: visual_review.json in the same unit. All attempts and source evidence are retained.

The original execution objective remains incomplete. Accurate automatic player selection, independent qualified camera/session data, full-match capacity and physical ball speed/3D validation remain separate outstanding requirements. The explicitly permitted assisted court-correction feature is being prepared separately and cannot replace automatic accuracy evaluation.
