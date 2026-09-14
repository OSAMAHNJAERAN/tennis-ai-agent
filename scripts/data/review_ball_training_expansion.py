"""Independently verify counts and record the completed source-image inspection."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


# Written after inspecting every generated board. These are qualitative notes,
# not replacement targets and not a claim of exhaustive label correctness.
NOTES={
 'match208_000':('clay','elevated_baseline','singles','Moving bright streak crosses the net near the labeled point.'),
 'match209_000':('indoor_hard','low_baseline','singles','Label lies in far-player hand/racket contact clutter; the exact ball cannot be isolated confidently.'),
 'match21_000':('hard','elevated_baseline','singles','Fast pale streak moves through the net/background at the visible label.'),
 'match210_000':('clay','elevated_baseline','singles','Moving bright streak crosses service-line markings.'),
 'match211_000':('indoor_hard','elevated_baseline','singles','Pale moving streak is visible just above the net near the target.'),
 'match212_000':('grass','elevated_baseline','singles','Yellow moving ball is visible over the net against grass.'),
 'match213_000':('hard','elevated_baseline','doubles','Moving ball is visible over the net; four active players appear in the scene.'),
 'match214_000':('hard','elevated_baseline','singles','Faint elongated motion streak appears beside the center service line.'),
 'match215_000':('hard','elevated_baseline','singles','Moving green/yellow ball appears against the turquoise far run-off area.'),
 'match216_000':('hard','elevated_baseline','singles','Yellow ball passes in front of dark net and lettering.'),
 'match217_000':('hard','low_baseline','singles','Pale elongated ball moves against dark advertising and court edge.'),
 'match218_000':('clay','elevated_baseline','singles','Bright moving ball appears against clay near the service box.'),
 'match219_000':('indoor_hard','elevated_baseline','singles','Bright descending motion streak appears near the center service line.'),
 'match22_000':('clay','elevated_baseline','doubles','Blurred moving ball appears on near-right clay; four active players are visible.'),
 'match220_000':('hard','elevated_baseline','singles','Yellow moving streak appears over the near service box.'),
 'match221_000':('hard','elevated_baseline','singles','Pale ball moves across net/line intersection; central frame is blurred.'),
 'match222_000':('hard','elevated_baseline','singles','Large elongated yellow streak moves across net lettering.'),
 'match223_000':('clay','elevated_baseline','singles','Distinct yellow moving ball is visible against clay.'),
 'match224_000':('indoor_hard','elevated_baseline','singles','Bright ball stretches into a vertical streak across the net-top region.'),
 'match225_000':('hard','elevated_baseline','singles','Pale moving ball crosses the service-line and net region.')
}


def main():
    out=ROOT/'outputs/vision_upgrade_audit/ball_training_expansion60'
    destination=out/'review.json'
    if destination.exists():
        raise FileExistsError(destination)
    report=read(out/'report.json')
    assert report['complete'] and len(report['clips'])==60
    assert digest(ROOT/'scripts/data/audit_ball_training_expansion.py')==report['script_sha256']
    assert digest(ROOT/'docs/experiments/BALL_TRAINING_EXPANSION60_PROTOCOL.md')==report['protocol_sha256']
    clips,manifests=[],{}
    for path,checksum in report['dataset_manifest_hashes'].items():
        folder=ROOT/path
        assert digest(folder/'manifest.json')==checksum
        m=read(folder/'manifest.json')
        assert m['split']=='TRAINING_ONLY' and m['revision']=='85157ca21faa2abca96d837dd2b963738029bcc8'
        for f in m['files']:
            assert digest(folder/f['path'])==f['sha256']
        manifests[path]=m
        clips.extend(tuple(c) for c in m['selected_clips'])
    assert len(clips)==len(set(clips))==60
    first=ROOT/next(iter(manifests))
    publisher_train=read(first/'tennis/info/train.json')
    publisher_val=read(first/'tennis/info/val.json')
    assert clips==[tuple(c) for c in publisher_train[:60]]
    assert not {m for m,_ in clips}&{m for m,_ in publisher_val}
    unique=list(dict.fromkeys(m for m,_ in clips))
    held=set(unique[4::5])
    summary={name:Counter() for name in ('train','selection')}
    images,observations=[],[]
    new_frames=0
    for clip in report['clips']:
        role='selection' if clip['clip'].rsplit('_',1)[0] in held else 'train'
        assert clip['split']==role
        folder=ROOT/clip['dataset']
        match,rally=clip['clip'].rsplit('_',1)
        source=folder/f'tennis/all/{match}/csv/{rally}_ball.csv'
        assert digest(source)==clip['labels_sha256']
        assert digest(folder/f"tennis/videos/{clip['clip']}.mp4")==clip['video_sha256']
        with source.open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        indices=[int(r['Frame']) for r in rows]
        assert len(indices)==len(set(indices)) and all(0<=i<clip['frames'] for i in indices)
        assert all(r['Visibility'] in ('0','1') for r in rows)
        visible=sum(r['Visibility']=='1' for r in rows)
        absent=len(rows)-visible
        assert (len(rows),visible,absent)==(clip['label_count'],clip['visible_labels'],clip['absent_labels'])
        for row in rows:
            if row['Visibility']=='1':
                x,y=float(row['X']),float(row['Y'])
                assert math.isfinite(x) and math.isfinite(y) and 0<=x<1920 and 0<=y<1080
        stride=max(1,int(math.floor(clip['fps']/30+.5)))
        assert stride==clip['temporal_stride']
        complete_context=sum(any(i-slot*stride>=0 and i+(2-slot)*stride<clip['frames'] for slot in (0,1,2)) for i in indices)
        assert complete_context==clip['labels_with_real_spaced_context']==len(rows)
        summary[role].update(clips=1,labels=len(rows),visible=visible,absent=absent)
        if not clip['new_acquisition']:
            continue
        assert clip['fully_decoded_this_audit']
        new_frames+=clip['frames']
        image=out/clip['review_image']
        assert digest(image)==report['image_hashes'][image.name]
        images.append(image.name)
        expected=[]
        for vis in ('1','0'):
            eligible=sorted([r for r in rows if r['Visibility']==vis],key=lambda r:int(r['Frame']))
            if eligible:
                expected.append(int(eligible[0]['Frame']))
        assert expected==[r['frame'] for r in clip['review']]
        surface,camera,format_,note=NOTES[clip['clip']]
        for r in clip['review']:
            observations.append(dict(clip=clip['clip'],frame=r['frame'],split=role,image=image.name,
                                     visible_label=r['target_xy'] is not None,visually_inspected=True,
                                     visual_note=note if r['target_xy'] is not None else
                                     'Full-frame adjacent context inspected. At this review scale, absence of a tiny or occluded active ball is not conclusively certified; preserve publisher label.',
                                     annotation_certified=False))
    assert {k:dict(v) for k,v in summary.items()}==report['summary']
    assert len(images)==20 and len(observations)==35 and new_frames==6751
    review=dict(complete=True,qualification_evidence=False,report_sha256=digest(out/'report.json'),
                reviewer_sha256=digest(Path(__file__)),independent_counts_exact=True,source_hashes_verified=True,
                new_images_inspected=20,preselected_examples_inspected=35,observations=observations,
                scene_tags={name:dict(surface=n[0],camera=n[1],format=n[2],source='ASSISTANT_VISUAL_TRIAGE') for name,n in NOTES.items()},
                labels_changed=False,training_started=False,
                remaining_uncertainty='35 sampled labels do not certify all 1000 new targets; absence thumbnails cannot prove invisibility. Source broadcast grouping and original model pretraining overlap remain unknown.',
                next_gate='DECLARE_BOUNDED_TRAINING_PROTOCOL_WITH_UNCHANGED_EVALUATION_LABELS')
    destination.write_text(json.dumps(review,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in review.items() if k not in ('observations','scene_tags')},indent=2))


if __name__=='__main__':
    main()
