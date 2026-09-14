"""Record actual inspection of the nine preselected source-context examples."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/vision_upgrade_audit/ball_spaced_error_audit01'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review = json.loads((OUT/'visual_review.json').read_text())
    assert review['report_sha256'] == digest(OUT/'report.json')
    for name,checksum in review['image_hashes'].items():
        assert digest(OUT/name) == checksum
    observations = {
        ('match143_000',126): 'Visible yellow ball descends against spectator seats; selected point is elsewhere. Stride-one candidate coverage is not saved.',
        ('match143_000',155): 'Yellow ball is visible near the far player legs/racket across adjacent frames; final abstains. Stride-one candidate coverage is not saved.',
        ('match143_000',174): 'Selected position falls on white PERTH floor lettering; the three local crops show static lettering.',
        ('match143_000',202): 'Selected position again falls on white PERTH floor lettering; adjacent crops show the same text.',
        ('match144_000',447): 'Labeled center is near the far player lower leg/shoe. These small native crops do not establish a clear ball at the target; labels remain unchanged.',
        ('match144_000',459): 'Pale moving object crosses the baseline across adjacent frames; current frame is faint at the line. Pixel-motion filtering removes the raw correct selection.',
        ('match144_000',530): 'Target is beside the near player hand/racket during contact preparation; a pale object and motion blur are visible, but separation is difficult in the current crop.',
        ('match155_000',234): 'Visible elongated pale ball on the purple court; raw selection is elsewhere although a correct candidate is available. Final abstains.',
        ('match155_000',246): 'Visible moving pale ball near the net/advertisement texture; a correct candidate is available but the final selected point is displaced.'}
    assert set(observations) == {(r['clip'],r['frame']) for r in review['entries']}
    for row in review['entries']:
        row.update(visually_inspected=True,observation=observations[row['clip'],row['frame']])
    review.update(visually_inspected=True,inspected_examples=9,
                  scope='All nine predetermined examples inspected; not every one of the 55 final-error frames.',
                  observation_recorder_sha256=digest(Path(__file__)))
    (OUT/'visual_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    print(json.dumps(dict(inspected=9,report_sha256=review['report_sha256'],visual_review_sha256=digest(OUT/'visual_review.json'))))


if __name__ == '__main__':
    main()
