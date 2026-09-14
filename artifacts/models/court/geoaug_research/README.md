---
license: other
license_name: research-use-only
license_link: https://github.com/yastrebksv/TennisCourtDetector
tags:
  - computer-vision
  - keypoint-detection
  - tennis
  - sports-analytics
---

# Tennis court keypoint detector — geometrically fine-tuned

ResNet-50 regression head predicting the **14 standard tennis court keypoints** from a
single broadcast frame. Used by [Tennis-Vision](https://github.com/HarshTomar1234/Tennis-Vision)
to fit the court homography that every real-world measurement depends on.

## Why this fine-tune exists

The base model is accurate on footage resembling its training distribution but
degrades on real broadcast clips with different camera framing. Measured on a 9-clip
YouTube evaluation suite, only about 4 of 9 clips produced a usable court fit — and
the failures were not obvious, because the model returns a tidy quadrilateral even
when it lands on the stands.

Per-surface accuracy was measured first and ruled out the obvious hypothesis:

| surface | median keypoint error | within 25 px |
|---|---|---|
| hard (blue) | 3.90 px | 98.5 % |
| clay | 4.58 px | 98.0 % |
| grass / green | 4.65 px | 98.2 % |

All three within 0.75 px of each other, so surface was never the weakness. Every
observed real-world failure was the predicted court displaced **vertically**.
Augmentation is therefore **geometric** — translation (wider vertically), scale, mild
perspective warp, horizontal flip with keypoint remapping — and deliberately **not**
photometric. Colour jitter would have trained hard and fixed nothing.

## Results

Held-out validation split of the TennisCourtDetector dataset (2,211 images):

| metric | base | fine-tuned |
|---|---|---|
| median keypoint error | 4.03 px | **2.90 px** |
| mean keypoint error | 5.71 px | **3.92 px** |
| keypoints within 10 px | 93.7 % | **97.2 %** |
| images with all 14 keypoints within 25 px | 96.8 % | **98.3 %** |

On the real-clip evaluation suite, clips passing the court-validity gate went from
**4/9 to 8/9**. Validated on Wimbledon grass footage the model had never seen
(line-support 0.65 / 0.64 / 0.63). The one remaining failure is a near-ground-level
camera where the court is extremely foreshortened; the pipeline's validity gate flags
that case rather than reporting confident wrong numbers.

Both numbers are reproducible:

```bash
python eval/court_keypoint_accuracy.py --model models/keypoints_model_geoaug.pth
python eval/court_validity_calibration.py --model models/keypoints_model_geoaug.pth
```

## Usage

```python
import torch, cv2
from torchvision import models

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2)
model.load_state_dict(torch.load("keypoints_model_geoaug.pth", map_location="cpu"))
model.eval()
# Input: 224x224 RGB, ImageNet normalisation.
# Output: 28 values (x0, y0, ... x13, y13) in 224x224 space — rescale by
# original_width / 224 and original_height / 224.
```

Keypoint order (index pairs forming court lines): `(0,2)` left outer sideline,
`(1,3)` right outer sideline, `(0,1)` far baseline, `(2,3)` near baseline,
`(4,5)` left singles sideline, `(6,7)` right singles sideline, `(8,9)` far service
line, `(10,11)` near service line, `(12,13)` centre service line.

## Limitations

- Trained and evaluated on **broadcast and elevated fixed-camera** footage. Ground-level
  cameras fail.
- Doubles and amateur footage are untested.
- The validation split shares lineage with the base model's training data, so 2.90 px
  is an in-distribution figure. The real-clip result (8/9) is the out-of-distribution
  evidence.

## Provenance and credit

Fine-tuned from the pretrained weights in
**[yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)**,
on that project's dataset (8,841 annotated images across hard, clay and grass). All
credit for the original architecture, dataset and base weights belongs to that author.

The upstream licence is not explicitly stated; the dataset was released for research
reproduction. This derivative is published on the same terms — **research use only** —
and should not be assumed to carry any broader grant.
