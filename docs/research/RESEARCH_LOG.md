# Research log

This index records decisions and links the detailed source audits. Reported research metrics are not project measurements. Earlier audits remain in this directory and retain their original dates and evidence.

## 22 October 2026: NKUST landing-reference measurement semantics

- **Sources and date:** [publisher dataset v4](https://data.mendeley.com/datasets/75m8vz7jr2/4), released 1 May 2024, and [author article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11068545/), reviewed 22 October. The article cites v2; v4 requires its own file correspondence. No publisher implementation was executed.
- **Rights:** dataset CC BY 4.0; retained whole-session reservation separates outdoor reference TEST from indoor reference DEV. Neither session is a singles-match qualification source.
- **Confirmed and unresolved:** signed distances to a doubles sideline and baseline can support a separate practice-landing comparison. Exact CSV-to-clip identity, baseline/line convention, global axes, calibration, contact timing and measurement uncertainty remain unresolved. Paired clip numbers do not establish synchronization.
- **Metrics, hardware and reproduction:** no project accuracy, model execution, TEST frame/target inspection or GPU use. Source text and existing acquisition metadata were inspected only; actual measurement coverage is still absent.
- **Decision:** `REFERENCE_ONLY` pending indoor DEV convention review and a frozen measurement protocol. Do not derive global XY, speed truth or line-officiating qualification from these labels.

See [the source audit](../data/INDEPENDENT_SOURCE_ACQUISITION.md#22-october-2026--nkust-landing-reference-semantics-before-test-exposure) for exact methods and limitations.

Subsequently, the official metadata inventory enabled actual acquisition of one indoor DEV view pair and its 99-row CSV, with all publisher hashes verified. Native PTS show 139 frames per view at `60000/1001`; six independently agent-reviewed image anchors do not establish the measured outgoing landing or regulation geometry. The [retained source manifest](../../data/external/nkust_landing_reference_indoor_development_v4/source_manifest.json) and source audit distinguish this new DEV inspection from the earlier text-only phase. No model, outdoor TEST inspection, calibrated metric or qualification was produced.

## 17 October 2026: CalTennis camera/export conventions

- **Source title and date:** CalTennis paper/project/dataset and a candidate `caltech-tennis-benchmark` implementation; audited 17 October 2026. The candidate code's relationship to the publisher's released assets is unverified; its release date is not established by this audit.
- **Paper/project:** [CalTennis paper](https://arxiv.org/html/2606.20542v1), [publisher project](https://ilonadem.github.io/caltennis-website/).
- **Code:** [candidate commit 4a24b98](https://github.com/selinananana/caltech-tennis-benchmark/tree/4a24b98ead5a000a2597770e617bb31225f8f122). The pinned publisher website still has an unassigned code link.
- **Dataset and license:** [publisher dataset](https://huggingface.co/datasets/demalenk/caltennis), CC BY-NC 4.0 as recorded in the existing acquisition audit. Dataset rights do not establish candidate-code rights or public-example rights.
- **Tennis relevance and assumptions:** court-coordinate and timestamp conventions could support independently calibrated player measurements, subject to verified export linkage, camera geometry and cross-camera synchronization. The candidate code defines meters and relative milliseconds; released-array offsets and the 1080-to-1088 pixel mapping remain unresolved.
- **Claimed metrics and hardware:** none reproduced or adopted in this follow-up. This was bounded text-only inspection, with no model or GPU execution; the source workflow's production hardware requirements were not established.
- **Reproduction and limitations:** no code execution, media inspection or reserved-test reads. Local reservations remain unchanged. Publisher linkage, geographic axes, alignment offsets and anonymizer mapping remain unverified.
- **Decision:** `REFERENCE_ONLY`. Retain the definitions as a conditional lead; do not promote them to synchronized ground truth or a speed-accuracy result.

Detailed source links, earlier findings and exact limitations: [camera-convention audit](CALTENNIS_CAMERA_CONVENTIONS.md). Retained request/error/hash receipt: `artifacts/validation/yolo26_execution_2026-10-17/caltennis_conventions_followup_v1/followup_receipt.json`, SHA256 `93213a55f8f210f0398f900f147fc35e5fefbbd7747b89c8d86757a099477d92`.
