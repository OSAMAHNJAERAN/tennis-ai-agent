# Tennis person and pose validation label sources

Research date: 2026-09-08. Documentation-only follow-up to `VISION_UPGRADE_RESEARCH_2026_09_07.md`. No dataset or model downloads, inference, tests, or benchmark processes were started by this research task. Endpoint publication was verified; archive transfer, extraction and subset counts were not.

## Practical first source: COCO 2017 validation

Use the official **val2017 images containing a tennis racket**, retaining their independently supplied person boxes and person keypoints. This is a tennis-context still-image validation slice. It cannot establish continuous player identity, identity switches, tracking recovery, or accuracy on this project's match videos. A racket annotation does not identify which nearby person is a player.

**Category mapping correction:** official COCO sparse `category_id=43` is `tennis racket`; `category_id=39` is `baseball bat`. Person is category 1. Ultralytics uses contiguous zero-based indices: person 0, tennis racket 38. Resolve names from each source's categories instead of assuming the number systems match. [COCO category metadata](https://raw.githubusercontent.com/cocodataset/panopticapi/master/panoptic_coco_categories.json), [Ultralytics COCO configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

| Asset | Official endpoint | Published size and intended use |
|---|---|---|
| 2017 validation images | [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) | COCO advertises 5K images / 1 GB, rounded. Full validation images, not a tennis-only archive. |
| 2017 train/validation annotations | [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) | COCO advertises 241 MB, rounded. Use `annotations/instances_val2017.json` and `annotations/person_keypoints_val2017.json`; do not train from this validation slice. |
| Selected images only | Each selected image record's `coco_url` | The official API downloads exactly these URLs for requested image IDs. Prefer these or already cached images after selecting the subset. Aggregate tennis-subset size is unknown until the manifest is built. |

Archive URLs and rounded sizes are from the [official download-page source](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm). Selective downloads are supported by the [official COCO API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py). The person-keypoint annotation path is also used in the [official MMPose COCO validation configuration](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py). No exact byte count or checksum was established here.

## Selection and scoring protocol

Recommended protocol, to run after the current timed benchmark:

1. Reuse any verified local copies first. Load the original instance and keypoint JSONs. Resolve `tennis racket` and `person` by name and assert sparse IDs 43 and 1. Build sorted, unique image IDs from **ground-truth racket annotations**, never candidate predictions.
2. Freeze all racket-image IDs as the tennis-context detection slice. Retain **all person annotations** in these images, including the metadata needed for COCO crowd/ignore handling. Do not score a player-only selector against all people, or count correctly detected spectators as false positives against a player-only target list. Player/non-player roles require an additional reviewed annotation layer.
3. Use the same image IDs for pose evaluation and preserve the original keypoint annotations and COCO ignore rules. Report how many images have eligible poses and how many non-crowd people have `num_keypoints > 0`; a separately reported eligible-pose subset may use that filter, but must not silently become the detection denominator. Preserve image and annotation IDs, original image dimensions, boxes, areas and visibility flags. Keypoint annotations include the corresponding instance ID and box. [COCO data format](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm).
4. Write a manifest containing source URLs, SHA-256 of source JSONs and images, ordered image IDs, selection rule, missing-image failures, counts, checkpoint hash, preprocessing and package revisions. Use the complete slice if feasible; if capped, select once using a recorded seed before observing predictions. Do not silently discard failures or unfavorable frames.
5. Evaluate person box AP/AR with COCO evaluation on the frozen image list. Evaluate pose with OKS AP/AR, retaining standard visibility semantics: 0 is unlabeled, 1 is labeled but not visible, 2 is labeled and visible. OKS scores labeled joints (`v > 0`); visible-only joint errors can be an additional clearly named diagnostic. Report sample counts and errors for wrists, elbows, knees and ankles, plus size strata. [Official keypoint evaluation](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/keypoints-eval.htm).
6. Compare end-to-end pose predictions using detected boxes. A second run using ground-truth crops can isolate pose localization quality, but label it as a diagnostic with oracle boxes; it does not measure detector misses. Keep model selection/tuning separate from a later match-disjoint project test set.

Coverage limit: COCO keypoints annotate mostly medium/large, non-crowd people; many small people have no labeled keypoints. Consequently, this slice offers limited evidence for small far-court players. Its instance schema also supplies no temporal identity supervision. [COCO data format](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm).

## Pretrained-model independence

Ultralytics documents YOLO11 detection weights as COCO-pretrained. Its standard detection configuration separates 118,287 `train2017` images from 5,000 `val2017` images; the pose configuration lists 56,599 training and 2,346 validation images. Thus **COCO-pretrained does not itself mean that val2017 was a training split**. [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11), [detection split configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), [pose split configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml).

However, these configurations do not audit every checkpoint's complete training history. Record exact checkpoint provenance and any additional fine-tuning. Public validation is also used for development and model selection: independent human annotations are not proof of an untouched blind test, novel-domain generalization, or absence of pretraining overlap for every candidate. Treat this as a reproducible validation baseline and retain separate independently labeled, match-disjoint footage for production claims.

## One temporal tennis candidate checked: Pose2Trajectory

The authors publish [Pose2Trajectory](https://github.com/alshami52/Pose2Trajectory), with an [annotation directory](https://github.com/alshami52/Pose2Trajectory/tree/main/dataset/annotations) containing `p1_and_p2_keyjoins_with_ball.csv`. The standard GitHub raw endpoint derived from that verified file path is [p1_and_p2_keyjoins_with_ball.csv](https://raw.githubusercontent.com/alshami52/Pose2Trajectory/main/dataset/annotations/p1_and_p2_keyjoins_with_ball.csv); the raw endpoint and its content were not fetched in this task.

The [authors' paper, sections 3â€“4](https://arxiv.org/html/2411.04501v1) describes Vienna Open singles clips from TennisTV, Faster R-CNN player detection with selected poor boxes manually relabeled, ViTPose body joints, TrackNet balls, and polynomial filling of missing ball coordinates. Therefore it is a plausible temporal trajectory resource, **not established independently annotated pose or continuous-identity ground truth**. The paper's claim that COCO lacks sports/tennis images should not be adopted: COCO explicitly includes a tennis-racket category.

Unknowns requiring inspection before any use: per-row manual-versus-generated provenance, complete frame-to-video mapping and source-video availability, human-verified persistent identity labels and switches, complete bounding-box availability in the CSV, annotation audit coverage, archive/data size, and video/data reuse terms. A repository MIT software license alone does not establish source-video rights. This candidate does not currently close the project's player-tracking validation gap; independently reviewed temporal box/identity labels are still needed.


## Additional temporal candidate: UVY-Track (checked2026-09-09)

The authors' [UVY record](https://zenodo.org/records/21303900), published July11,2026, describes sports videos collected under CC BY and annotations produced with YOLO-World followed by CVAT manual review/correction. The [publisher sequence table](https://zenodo.org/records/21303900/files/metrics_per_video.csv?download=1) lists three tennis sequences:819,968 and443 frames, with1,445,1,936 and822 player boxes respectively. This is a promising source of temporal player labels from user-generated footage. It does not establish pose, court or physical-speed ground truth.

The archive preview contains MOT-style `gt/gt.txt`, `gt/labels.txt` and `img1` images. Before scoring, inspect class IDs, frame-number origin, ignored/visibility semantics, annotation completeness and visual identity consistency. Some non-tennis rows report image/GT length discrepancies; do not infer that missing annotations mean no people. The table's tennis FPS entries are30, but source timing still needs verification. Semiautomatic annotations require quality checks; do not claim independent perfect labels.

The full archive is listed at3.3GB with publisher MD5 `f99594a1bd9f627ebe21219db86317a6`. Prefer a bounded tennis-only extraction if the public archive supports byte ranges, while preserving member CRC and source metadata; otherwise check storage before full acquisition. The public record API returned HTTP503 during metadata inspection, and one web JSON fetch returned429. No UVY image or GT acquisition has completed and no UVY accuracy is claimed. This server issue does not block ongoing ball training.
