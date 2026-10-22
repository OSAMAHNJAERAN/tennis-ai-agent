# Bounded court data repair wave

The fixed metadata plan selects 64 publisher images from 64 publisher video IDs absent from all 88 previously cached IDs. All selected splits, event contexts, real match IDs and camera sessions remain unassigned. No image was downloaded or opened by the planner; no label, model inference or training was produced.

`source_wave_plan.json` SHA256: `27d72148f4173ab476e9dbaaa5df6b64ce392864fd7150e6d7c0b59e893af65a`.

| Inventory or budget | Actual value |
| --- | ---: |
| Combined original annotation rows | 8,841 |
| Combined publisher video IDs | 498 |
| IDs shared by publisher train/validation | 434 |
| Existing images / excluded complete video IDs | 165 / 88 |
| Eligible new video IDs | 410 |
| Fixed selected images / video IDs | 64 / 64 |
| Original publisher-train / publisher-val provenance | 44 / 20 |
| Compressed selected PNG member bytes | 52,918,160 |
| Decompressed PNG file bytes | 56,517,686 |
| Maximum planned HTTP range bytes, including local-header allowance | 52,987,635 |
| Fixed range ceiling | 80,000,000 |
| Maximum inflight member requests | 1 |
| Media/image/model/GPU operations performed | 0 |

Selection is predeclared seed 709 with SHA256 ranking of canonical video IDs, followed by SHA256 ranking of image IDs within each selected video. Both original annotation splits are combined before excluding whole cached videos. The 64 selected rows and 410-candidate ordering are byte-equivalent to metadata plan v1. Version 2 additionally verifies the archive index and annotations against the original, hash-bound existing review protocol.

Two selected proposals retain out-of-image coordinates: `rhHgYUly2Fc_550` slots 0/1/4/6 and `wOHdAcE1ZLo_2550` slot 3. These are unresolved review cases, not valid supervised labels. All fourteen original semantic slots are retained for every row; visibility is unknown, no coordinates are clamped, and no source FPS/PTS is invented.

All 165 previous accepted **and rejected** video/event links and the complete prior context graph are retained. New native image review must merge compilation/event links transitively. Every component touching LondonFinals or MonteCarlo stays DEV. Distinct publisher videos do not establish independent matches, new event diversity, valid court geometry or untouched final tests. No final test is selected here.

The [author's dataset release and training instructions](https://github.com/yastrebksv/TennisCourtDetector#dataset) were checked on 15 October 2026. They describe 8,841 broadcast-derived 1280×720 images, fourteen keypoints, manually filtered classical proposals, a dataset download and training instructions. This wave retains the existing private local academic development basis. An explicit raw dataset license and broadcast redistribution clearance remain absent; code or mirror licenses are not substituted for those rights.

A separate fresh **256-byte non-image ZIP-tail** probe verified HTTP 206, the pinned 7,255,696,316-byte archive size and exact cached suffix bytes. The receipt is `../court_source_wave_metadata_v1/remote_archive_tail_probe.json`. This establishes current range access and suffix consistency; it does not establish a whole-archive checksum. No image member was requested.

Actual metadata execution took 0.152 seconds, peak owned RSS 55,513,088 bytes. Available physical RAM was 949,428,224 bytes before execution and 924,684,288 afterward. The metadata-only 128 MiB admission floor was enforced; media guards were untouched. All image, model and network imports were blocked with zero attempted forbidden imports. The dedicated 29 invariant tests passed in 0.15 seconds; results are under `../court_source_wave_metadata_checks_v2/`.

Next release is limited to these frozen member ranges, after root inspection of the concrete source hashes, rights and budget. Member acquisition must check ZIP local header/name/method, exact sizes, publisher CRC, image SHA256, native PNG geometry and complete final count. Native image/context/intersection review and root second pass must precede split freezing or supervised export.
