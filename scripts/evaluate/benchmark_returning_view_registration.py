"""Controlled source-video cut/return challenge for original-anchor recovery."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.court.calibration import CourtCalibration
from src.court.court_geometry import TennisCourtGeometry
from src.court.camera_registration import CourtCameraRegistration
from src.court.returning_view_registration import ReturningViewRegistration
from src.utils.video_frame_sequence import VideoFrameSequence
from src.utils.bbox_utils import BBox


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    out=ROOT/'outputs/vision_upgrade_audit/returning_view_registration01'
    if out.exists():raise FileExistsError(out)
    sources={}
    def load(path):
        sources[str(path.relative_to(ROOT))]=digest(path)
        return read(path)
    records={}
    for name in ('match148','match143'):
        run=ROOT/f'outputs/vision_upgrade_audit/phase6_spaced_{name}'
        detections=load(run/'detections.json')
        geometry=load(run/'court_geometry.json')
        video=ROOT/detections['metadata']['video']
        manifest_path=video.parents[2]/'manifest.json'
        manifest=load(manifest_path)
        expected=next(f['sha256'] for f in manifest['files'] if f['path'].replace('\\','/')==f'tennis/videos/{name}_000.mp4')
        assert digest(video)==expected
        sources[str(video.relative_to(ROOT))]=expected
        records[name]=(video,detections,geometry)
    reference=records['match148']
    initial=CourtCalibration(np.array(reference[2]['homography_matrix']))
    canonical=TennisCourtGeometry.get_canonical_keypoints().astype(np.float32)
    points=cv2.perspectiveTransform(canonical[None],np.linalg.inv(initial.image_to_court))[0]
    fps=reference[1]['metadata']['fps']
    assert fps==60
    maps={
        'healthy':[('match148',i) for i in range(120)],
        'unrelated_cut_return':[('match148',i) for i in range(60)]+[('match143',i) for i in range(30)]+[('match148',i) for i in range(60,120)],
        'blank_cut_return':[('match148',i) for i in range(60)]+[('blank',i) for i in range(30)]+[('match148',i) for i in range(60,120)],
        'permanent_unrelated':[('match148',i) for i in range(60)]+[('match143',i) for i in range(60)],
        'brief_return':[('match148',i) for i in range(60)]+[('blank',i) for i in range(30)]+[('match148',i) for i in range(60,62)]+[('blank',i) for i in range(30)],
    }
    out.mkdir(parents=True)
    report=dict(complete=False,qualification_evidence=False,source_hashes=sources,
                protocol_sha256=digest(ROOT/'docs/experiments/COURT_RETURNING_VIEW_PROTOCOL.md'),
                code_hashes={p:digest(ROOT/p) for p in ['scripts/evaluate/benchmark_returning_view_registration.py','src/court/returning_view_registration.py','src/court/camera_registration.py','src/court/calibration.py','src/utils/video_frame_sequence.py']},
                anchor_landmark_source='CANONICAL_LANDMARKS_PROJECTED_THROUGH_SAVED_INITIAL_HOMOGRAPHY; NOT_NEW_GROUND_TRUTH',
                mask_scope='SAVED_TWO_SELECTED_PLAYERS; OTHER_PEOPLE_NOT_MASKED',fps=fps,
                challenge_timing='SPLICED_SOURCE_FRAMES_ON_60_FPS_GRID; NOT_NATURAL_CUTS_OR_SOURCE143_PLAYBACK_TIMING',cases=[])
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    frames={name:VideoFrameSequence(str(record[0])) for name,record in records.items()}
    image_hashes={}
    try:
        for name,mapping in maps.items():
            started=time.perf_counter()
            baseline=CourtCameraRegistration(initial,points)
            recovery=ReturningViewRegistration(initial,points,fps=fps)
            entries=[]
            writer=None
            if name=='unrelated_cut_return':
                writer=cv2.VideoWriter(str(out/'cut_return_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,400))
                assert writer.isOpened()
            for index,(source,fi) in enumerate(mapping):
                if source=='blank':
                    frame=np.zeros((1080,1920,3),np.uint8);people=[]
                else:
                    frame=frames[source][fi]
                    det=records[source][1]['frames'][fi]
                    people=[BBox(*det[p]['bbox']) for p in ('player_1','player_2') if det[p] is not None]
                cv2.setRNGSeed(7);a=baseline.update(frame,people)
                cv2.setRNGSeed(7);b=recovery.update(frame,people)
                entries.append(dict(frame=index,source=source,source_frame=fi,original=a.audit,recovery=b.audit))
                if writer is not None:
                    panel=np.full((400,1280,3),22,np.uint8)
                    for col,(result,title) in enumerate(((a,'Existing latched registration'),(b,'Original-view recovery'))):
                        tile=cv2.resize(frame,(640,360))
                        if result.calibration.is_valid:
                            projected=cv2.perspectiveTransform(canonical[None],np.linalg.inv(result.calibration.image_to_court))[0]
                            for point in projected:
                                cv2.circle(tile,tuple(np.rint(point*np.array([640/1920,360/1080])).astype(int)),3,(50,240,70),-1)
                        state='VALID' if result.calibration.is_valid else 'WITHHELD'
                        panel[40:400,col*640:(col+1)*640]=tile
                        cv2.putText(panel,f'{title}: {state}',(col*640+8,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)
                    cv2.putText(panel,f'Controlled splice frame {index} / {source}:{fi}; recovered {b.audit["recovery_count"]}',(8,36),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
                    writer.write(panel)
                    if index in (59,60,89,90,95,119,149):
                        path=out/f'frame_{index:03}.jpg';assert cv2.imwrite(str(path),panel)
                        image_hashes[path.name]=digest(path)
            if writer is not None:writer.release()
            restore=[e['frame'] for e in entries if e['recovery']['recovered_this_frame']]
            false_accepts=[e['frame'] for e in entries if e['source']!='match148' and e['recovery']['is_valid']]
            healthy_exact=all(e['original']['image_to_court']==e['recovery']['image_to_court'] for e in entries if e['frame']<60 or name=='healthy')
            entry=dict(name=name,frames=len(entries),original_valid=sum(e['original']['is_valid'] for e in entries),
                       recovery_valid=sum(e['recovery']['is_valid'] for e in entries),recovered_frames=restore,
                       accepted_unrelated_frames=false_accepts,healthy_calibrations_exact=healthy_exact,
                       seconds=time.perf_counter()-started,entries=entries)
            report['cases'].append(entry);save()
            print(json.dumps({k:v for k,v in entry.items() if k!='entries'}),flush=True)
    finally:
        for source in frames.values():source.close()
    case={c['name']:c for c in report['cases']}
    healthy=case['healthy']['entries']
    errors=[]
    for name in ('unrelated_cut_return','blank_cut_return'):
        for e in case[name]['entries'][90:]:
            if e['recovery']['is_valid'] and healthy[e['source_frame']]['recovery']['is_valid']:
                got=np.array(e['recovery']['image_to_court']);expected=np.array(healthy[e['source_frame']]['recovery']['image_to_court'])
                source_points=cv2.perspectiveTransform(canonical[None],np.linalg.inv(expected))
                mapped=cv2.perspectiveTransform(source_points,got)[0]
                errors.append(float(np.linalg.norm(mapped-canonical,axis=1).max()))
    passed=(all(not c['accepted_unrelated_frames'] and c['healthy_calibrations_exact'] for c in report['cases']) and
            all(case[n]['recovered_frames'] for n in ('unrelated_cut_return','blank_cut_return')) and
            not case['brief_return']['recovered_frames'] and not case['permanent_unrelated']['recovered_frames'] and
            bool(errors) and max(errors)<.1)
    report.update(complete=True,advance_to_opt_in_integration=passed,image_hashes=image_hashes,
                  maximum_returned_vs_healthy_landmark_ground_difference_m=max(errors) if errors else None,
                  geometric_difference_scope='AGREEMENT_WITH_UNINTERRUPTED_ANCHOR_REGISTRATION; NOT_PHYSICAL_ACCURACY')
    save()
    print(json.dumps(dict(complete=True,advance=passed,maximum_agreement_error_m=max(errors) if errors else None)))


if __name__=='__main__':main()
