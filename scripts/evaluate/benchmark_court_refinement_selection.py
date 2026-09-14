"""Evaluate frozen court refinement on all preselected internal controls."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.court_geometry import TennisCourtGeometry
from src.court.calibration import calibrate_court
from scripts.evaluate.refine_court_line_geometry import refine, LINES
from scripts.evaluate.probe_court_line_segments import propose
from scripts.evaluate.benchmark_court_line_refinement import digest


def score(predictions,labels):
    details=[];tp=fp=fn=0
    for identity,(prediction,label) in enumerate(zip(predictions,labels,strict=True)):
        eligible=0<=label[0]<1280 and 0<=label[1]<720
        distance=None if prediction is None else float(np.linalg.norm(np.asarray(prediction)-label))
        correct=eligible and distance is not None and distance<=7
        if eligible:
            tp+=int(correct);fn+=int(not correct);fp+=int(prediction is not None and not correct)
        details.append({'identity':identity,'label':label,'prediction':prediction,'eligible':eligible,
                        'distance_native_px':distance,'correct':bool(correct)})
    return {'tp':tp,'fp':fp,'fn':fn,'landmarks':details}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    dataset=ROOT/'data/external/court_heatmap_pilot';manifest_path=dataset/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));assert manifest['complete']
    for row in manifest['files']:assert digest(dataset/row['path'])==row['sha256']
    selected=[r for r in manifest['samples'] if r['partition']=='selection'];assert len(selected)==32
    refinement_source=ROOT/'outputs/vision_upgrade_audit/court_line_refinement01/report.json'
    prior=json.loads(refinement_source.read_text(encoding='utf-8'));assert prior['complete']
    for path,checksum in prior['code_hashes'].items():assert digest(ROOT/path)==checksum
    checkpoint=ROOT/'models/keypoints_model.pth'
    detector=CourtKeypointDetector(model_path=str(checkpoint))
    args.output.mkdir(parents=True)
    report={'complete':False,'qualification_evidence':False,'scope':'REUSED_INTERNAL_SELECTION; PUBLISHER_TRAINING_EXPOSED; RAW_LABEL_AGREEMENT; RESNET_TRAINING_INDEPENDENCE_UNKNOWN',
            'original_protocol_sha256':digest(ROOT/'docs/experiments/COURT_REFINEMENT_RAW_LABEL_PROTOCOL.md'),
            'manifest_sha256':digest(manifest_path),'checkpoint_sha256':digest(checkpoint),
            'refinement_source_sha256':digest(refinement_source),
            'protocol_sha256':digest(ROOT/'docs/experiments/COURT_REFINEMENT_SELECTION_PROTOCOL.md'),
            'code_hashes':{p:digest(ROOT/p) for p in ('scripts/evaluate/benchmark_court_refinement_selection.py','scripts/evaluate/benchmark_court_refinement_labels.py',
                'scripts/evaluate/refine_court_line_geometry.py','scripts/evaluate/probe_court_line_segments.py',
                'src/court/court_keypoint_detector.py','src/court/calibration.py','src/court/court_geometry.py')},
            'images':[]}
    canonical=TennisCourtGeometry.get_canonical_keypoints().astype(float)
    for row in selected:
        path=dataset/'images'/f"{row['id']}.png";image=cv2.imread(str(path));assert image.shape==(720,1280,3)
        raw=detector.predict(image)
        calibration=calibrate_court(raw,canonical)
        projected=final=[None]*14
        evidence={'accepted':False,'reason':'INITIAL_GEOMETRIC_CALIBRATION_INVALID'}
        if calibration.is_valid:
            projected=cv2.perspectiveTransform(canonical[None],np.linalg.inv(calibration.image_to_court))[0]
            reference=cv2.resize(image,(960,540))
            segments=[r['segment'] for r in propose(reference) if r['support_fraction']>=.5]
            refined,evidence=refine(projected*.75,segments)
            final=(refined/.75).tolist();projected=projected.tolist()
        # Labels enter only below this point, after all prediction decisions.
        variants={'raw':raw.tolist(),'projected':projected,'refined':final}
        scored={name:score(points,row['kps']) for name,points in variants.items()}
        panels=[]
        for name,points in variants.items():
            panel=cv2.resize(image,(640,360))
            for a,b in LINES:
                if points[a] is not None and points[b] is not None:
                    cv2.line(panel,tuple(np.rint(np.asarray(points[a])*.5).astype(int)),tuple(np.rint(np.asarray(points[b])*.5).astype(int)),(30,30,230),1)
            for label,point in zip(row['kps'],points,strict=True):
                cv2.circle(panel,tuple(np.rint(np.asarray(label)*.5).astype(int)),4,(30,255,30),1)
                if point is not None:cv2.circle(panel,tuple(np.rint(np.asarray(point)*.5).astype(int)),2,(30,30,255),-1)
            panel=cv2.copyMakeBorder(panel,32,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
            cv2.putText(panel,f"{row['id']} {name} TP={scored[name]['tp']}/14",(6,13),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            cv2.putText(panel,'Green publisher label / red prediction; labels unverified',(6,27),cv2.FONT_HERSHEY_SIMPLEX,.36,(240,240,240),1)
            panels.append(panel)
        output=args.output/f"{row['id']}.jpg";assert cv2.imwrite(str(output),np.concatenate(panels),[cv2.IMWRITE_JPEG_QUALITY,88])
        entry={'id':row['id'],'input_sha256':digest(path),'initial_calibration':calibration.to_dict(),
               'refinement':evidence,'scores':scored,'image':output.name,'image_sha256':digest(output),'visually_inspected':False}
        report['images'].append(entry)
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'id':row['id'],'geometric_valid':calibration.is_valid,'refined':evidence['accepted'],
                          'tp':{name:s['tp'] for name,s in scored.items()}}),flush=True)
    pooled={}
    for name in ('raw','projected','refined'):
        counts={k:sum(r['scores'][name][k] for r in report['images']) for k in ('tp','fp','fn')}
        errors=[x['distance_native_px'] for r in report['images'] for x in r['scores'][name]['landmarks'] if x['eligible'] and x['distance_native_px'] is not None]
        pooled[name]={**counts,'precision':counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,
                      'recall':counts['tp']/(counts['tp']+counts['fn']),
                      'mean_native_error_available_px':float(np.mean(errors)) if errors else None,
                      'median_native_error_available_px':float(np.median(errors)) if errors else None,
                      'available_errors':len(errors)}
    report.update(complete=True,pooled=pooled)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(pooled),flush=True)


if __name__=='__main__':main()
