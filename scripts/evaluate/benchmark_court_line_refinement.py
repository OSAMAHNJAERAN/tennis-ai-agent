"""Frozen local court refinement across new failure cases and earlier controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.evaluate.refine_court_line_geometry import refine, LINES
from scripts.evaluate.probe_court_line_segments import propose
from src.court.calibration import calibrate_court
from src.court.court_geometry import TennisCourtGeometry


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    root=ROOT/'outputs/vision_upgrade_audit'
    sources={}
    def read(path):
        sources[str(path)]=digest(path)
        return json.loads(path.read_text(encoding='utf-8'))
    cases=[]
    probe=read(root/'spaced_pipeline_court_line_probe/report.json')
    for name,checksum in probe['code_hashes'].items():assert digest(ROOT/name)==checksum
    for row in probe['cases']:
        camera=root/f"phase6_spaced_{row['clip'].split('_')[0]}"/'camera_registration.json'
        assert digest(camera)==row['camera_report_sha256']
        sources[str(camera)]=digest(camera)
        video=ROOT/f"data/external/racketvision_validation_expansion12/tennis/videos/{row['clip']}.mp4"
        assert digest(video)==row['input_sha256']
        cases.append(dict(model='pipeline',identity=row['clip'],frame=row['frame'],video=video,
                          reference_points=row['projected_canonical_points_reference_px'],segments=row['selected_segments']))
    for name in ['uvy_player_tracking_baseline','uvy_player_tracking_geoaug']:
        source=read(root/name/'report.json')
        manifest=read(ROOT/'data/external/uvy_tennis_videos/manifest.json')
        for identity,row in source['sequences'].items():
            cases.append(dict(model=name,identity=identity,frame=0,
                              video=ROOT/'data/external/uvy_tennis_videos'/manifest['sequences'][identity]['video'],points=row['keypoints']))
    source=read(root/'caltennis_court_probe/report.json')
    for identity,row in enumerate(source['clips'],1):
        for name,prediction in row['samples'][0]['predictions'].items():
            cases.append(dict(model='caltennis_'+name,identity=str(identity),frame=0,
                              video=ROOT/'data/external/caltennis_diagnostic'/row['video'],points=prediction['keypoints']))
    source=read(root/'court_heatmap_broadcast_control_verified/report.json')
    for row in source['frames']:
        cases.append(dict(model='broadcast_heatmap_control',identity=row['video'],frame=0,
                          video=ROOT/'data/external/racketvision_validation'/row['video'],points=row['points']))
    report={'complete':False,'qualification_evidence':False,'source_reports':sources,
            'code_hashes':{name:digest(ROOT/name) for name in (
                'scripts/evaluate/benchmark_court_line_refinement.py','scripts/evaluate/refine_court_line_geometry.py',
                'scripts/evaluate/audit_court_line_support.py','scripts/evaluate/probe_court_line_segments.py',
                'src/court/calibration.py','src/court/court_geometry.py')},
            'protocol_sha256':digest(ROOT/'docs/experiments/COURT_PROJECTIVE_LINE_REFINEMENT.md'),
            'scope':'REUSED_DEVELOPMENT_IMAGE_SUPPORT; NOT_INDEPENDENT_COURT_ACCURACY', 'cases':[]}
    canonical=TennisCourtGeometry.get_canonical_keypoints().astype(float)
    cache={}
    for index,case in enumerate(cases):
        key=str(case['video']),case['frame']
        if key not in cache:
            cap=cv2.VideoCapture(key[0]);cap.set(cv2.CAP_PROP_POS_FRAMES,key[1]);ok,image=cap.read();cap.release()
            assert ok
            scale=960/image.shape[1]
            image=cv2.resize(image,(960,round(image.shape[0]*scale)))
            segments=[r['segment'] for r in propose(image) if r['support_fraction']>=.5]
            cache[key]=(image,scale,segments)
        image,scale,segments=cache[key]
        result={'model':case['model'],'identity':case['identity'],'frame':case['frame'],
                'video':str(case['video']),'video_sha256':digest(case['video'])}
        if 'reference_points' in case:
            points=np.asarray(case['reference_points'])
            assert segments==case['segments']
        else:
            predicted=case['points']
            available=[i for i,p in enumerate(predicted) if p is not None]
            calibration=calibrate_court(np.asarray([predicted[i] for i in available]),canonical[available])
            result['initial_calibration']=calibration.to_dict()
            if not calibration.is_valid:
                result.update(accepted=False,reason='INITIAL_GEOMETRIC_CALIBRATION_INVALID')
                report['cases'].append(result)
                print(json.dumps({k:result[k] for k in ('model','identity','accepted','reason')}),flush=True)
                continue
            points=cv2.perspectiveTransform(canonical[None],np.linalg.inv(calibration.image_to_court))[0]*scale
        refined,evidence=refine(points,segments,image.shape[1],image.shape[0])
        result.update(evidence,original_points=points.tolist(),refined_points=refined.tolist())
        panels=[]
        for candidate,title in ((points,'Before'),(refined,'After' if evidence['accepted'] else 'Unchanged / rejected')):
            panel=image.copy()
            for a,b in LINES:
                cv2.line(panel,tuple(np.rint(candidate[a]).astype(int)),tuple(np.rint(candidate[b]).astype(int)),(30,30,255),2)
            cv2.rectangle(panel,(0,0),(960,32),(20,20,20),-1)
            cv2.putText(panel,f'{index}: {title} {case["model"]} frame{case["frame"]}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            panels.append(panel)
        name=f'case_{index:02}.jpg'
        assert cv2.imwrite(str(args.output/name),np.concatenate(panels),[cv2.IMWRITE_JPEG_QUALITY,85])
        result.update(image=name,image_sha256=digest(args.output/name),visually_inspected=False)
        report['cases'].append(result)
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'case':index,'model':case['model'],'identity':case['identity'],
                          'accepted':evidence['accepted'],'reason':evidence['reason'],
                          'before':evidence['before']['mean_visible_line_support'],
                          'after':evidence.get('proposed_after',{}).get('mean_visible_line_support')}),flush=True)
    assert len(report['cases'])==24
    report['complete']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
