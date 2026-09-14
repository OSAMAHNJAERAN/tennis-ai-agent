"""Replay one frozen first-frame court correction through saved camera motion."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.evaluate.benchmark_court_line_refinement import digest
from scripts.evaluate.audit_court_line_support import support, LINES
from scripts.evaluate.probe_court_line_segments import propose
from src.court.court_geometry import TennisCourtGeometry


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    base=ROOT/'outputs/vision_upgrade_audit'
    source_path=base/'court_line_refinement01/report.json'
    source=json.loads(source_path.read_text(encoding='utf-8'))
    assert source['complete']
    for name,checksum in source['code_hashes'].items():assert digest(name)==checksum
    result={'complete':False,'qualification_evidence':False,'source_report_sha256':digest(source_path),
            'scope':'ONE_FIRST_FRAME_CORRECTION_PROPAGATED_WITH_EXISTING_CAMERA_TRANSFORMS; NO_PER_FRAME_REFIT',
            'image_support_sampling':'frame0, each whole native second, final frame; same fitting segments are diagnostic only',
            'script_sha256':digest(__file__),'clips':[]}
    canonical=TennisCourtGeometry.get_canonical_keypoints().astype(float)
    for name in ('match143_000','match148_000'):
        case=next(c for c in source['cases'] if c['model']=='pipeline' and c['identity']==name and c['frame']==0)
        assert case['accepted']
        run=base/f"phase6_spaced_{name.split('_')[0]}"
        camera_path=run/'camera_registration.json'
        camera=json.loads(camera_path.read_text(encoding='utf-8'))
        validation=json.loads((run/'run_validation.json').read_text(encoding='utf-8'))
        video=Path(validation['input_video']);assert digest(video)==validation['input_sha256']==case['video_sha256']
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
        width,height=cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scale=width/960
        anchor,_=cv2.findHomography(canonical,np.asarray(case['refined_points'])*scale,0)
        assert np.linalg.norm(cv2.perspectiveTransform(canonical[None],anchor)[0]/scale-np.asarray(case['refined_points']),axis=1).max()<1e-4
        count=len(camera['frames']);sampled={0,count-1}|{round(t*fps) for t in range(1,int(count/fps))}
        writer=cv2.VideoWriter(str(args.output/f'{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,400))
        assert writer.isOpened()
        rows=[];samples=[];snapshots={}
        for index,entry in enumerate(camera['frames']):
            ok,image=cap.read();assert ok and entry['frame_index']==index
            if entry['is_valid']:
                old=cv2.perspectiveTransform(canonical[None],np.linalg.inv(np.asarray(entry['image_to_court'])))[0]
                transform=np.asarray(entry['anchor_to_frame_px']) @ anchor
                refined=cv2.perspectiveTransform(canonical[None],transform)[0]
                assert np.isfinite(refined).all() and np.isfinite(np.linalg.inv(transform)).all()
                rows.append({'frame':index,'image_to_court':np.linalg.inv(transform).tolist(),'points':refined.tolist()})
            else:
                old=refined=None;rows.append({'frame':index,'image_to_court':None,'points':None})
            if index in sampled and old is not None:
                reference=cv2.resize(image,(960,round(height/scale)))
                segments=[r['segment'] for r in propose(reference) if r['support_fraction']>=.5]
                samples.append({'frame':index,'before':support(old/scale,segments,960,reference.shape[0]),
                                'after':support(refined/scale,segments,960,reference.shape[0])})
            canvas=np.full((400,1280,3),22,np.uint8)
            for column,(points,title) in enumerate(((old,'Original calibration'),(refined,'Refined first-frame anchor'))):
                tile=cv2.resize(image,(640,360))
                if points is not None:
                    xy=points*np.array([640/width,360/height])
                    for a,b in LINES:cv2.line(tile,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),(20,230,250),1)
                left=column*640;canvas[40:,left:left+640]=tile
                cv2.putText(canvas,title,(left+8,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
                cv2.putText(canvas,f'{name} frame {index} | research geometry / no physical accuracy claim',(left+8,34),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            writer.write(canvas)
            if index in (0,count//2,count-1):
                filename=f'{name}_{index}.jpg';assert cv2.imwrite(str(args.output/filename),canvas)
                snapshots[filename]=digest(args.output/filename)
        cap.release();writer.release()
        output= args.output/f'{name}.mp4'
        check=cv2.VideoCapture(str(output));decoded=0
        assert abs(check.get(cv2.CAP_PROP_FPS)-fps)<.01
        while True:
            ok,frame=check.read()
            if not ok:break
            assert frame.shape[:2]==(400,1280);decoded+=1
        check.release();assert decoded==count
        result['clips'].append({'clip':name,'camera_report_sha256':digest(camera_path),'input_sha256':digest(video),
                                'decoded_frames':count,'fps':fps,'video_sha256':digest(output),
                                'snapshots':snapshots,'frames':rows,'support_samples':samples,'visually_inspected':False})
        print(json.dumps({'clip':name,'frames':count,'samples':[(r['frame'],r['before']['mean_visible_line_support'],r['after']['mean_visible_line_support']) for r in samples]}),flush=True)
        (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    result['complete']=True
    (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
