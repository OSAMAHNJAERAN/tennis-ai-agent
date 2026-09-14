"""Replay one fixed calibration ratio change across all existing court controls."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from src.court.calibration import calibrate_court
from src.court.court_geometry import TennisCourtGeometry
from scripts.evaluate.refine_court_boundary_geometry import refine, LINES
from scripts.evaluate.probe_court_ridge_ablation import propose
from scripts.evaluate.benchmark_court_refinement_labels import score
from scripts.evaluate.benchmark_court_line_refinement import digest


def main():
    base = ROOT/'outputs/vision_upgrade_audit'
    out = base/'court_calibration_support01'
    if out.exists():
        raise FileExistsError(out)
    sources, cases = {}, []
    def read(path):
        sources[str(path.relative_to(ROOT))] = digest(path)
        result = json.loads(path.read_text())
        assert result.get('complete', True)
        return result
    for group, folder in [('train','court_remaining_pilot01'), ('selection','court_refinement_selection01'), ('external','court_refinement_raw_labels01')]:
        report = read(base/folder/'report.json')
        for name, checksum in report['code_hashes'].items():
            assert digest(ROOT/name) == checksum
        for row in report['images']:
            cases.append(dict(group=group, id=row['id'], model='resnet',
                              points=[p['prediction'] for p in row['scores']['raw']['landmarks']],
                              labels=[p['label'] for p in row['scores']['raw']['landmarks']],
                              expected=row['initial_calibration'],
                              path=ROOT/'data/external/court_heatmap_pilot/images'/f"{row['id']}.png",
                              input_sha256=row['input_sha256'], frame=None))
    prior = read(base/'court_line_refinement01/report.json')['cases'][6:]
    controls = []
    manifest = read(ROOT/'data/external/uvy_tennis_videos/manifest.json')
    for model in ('uvy_player_tracking_baseline','uvy_player_tracking_geoaug'):
        report = read(base/model/'report.json')
        for identity, row in report['sequences'].items():
            controls.append(dict(model=model, id=identity, points=row['keypoints'],
                                 path=ROOT/'data/external/uvy_tennis_videos'/manifest['sequences'][identity]['video']))
    report = read(base/'caltennis_court_probe/report.json')
    for identity, row in enumerate(report['clips'],1):
        for name, prediction in row['samples'][0]['predictions'].items():
            controls.append(dict(model='caltennis_'+name, id=str(identity), points=prediction['keypoints'],
                                 path=ROOT/'data/external/caltennis_diagnostic'/row['video']))
    report = read(base/'court_heatmap_broadcast_control_verified/report.json')
    for row in report['frames']:
        controls.append(dict(model='broadcast_heatmap_control', id=row['video'], points=row['points'],
                             path=ROOT/'data/external/racketvision_validation'/row['video']))
    for case, old in zip(controls, prior, strict=True):
        assert (case['model'],case['id']) == (old['model'],old['identity'])
        case.update(group='camera_control',labels=None,expected=old['initial_calibration'],
                    input_sha256=old['video_sha256'],frame=0)
        cases.append(case)
    assert len(cases) == 183
    canonical = TennisCourtGeometry.get_canonical_keypoints().astype(float)
    out.mkdir(parents=True)
    report = dict(complete=False,qualification_evidence=False,alternate_ratio=5/7,sources=sources,
                  protocol_sha256=digest(ROOT/'docs/experiments/COURT_CALIBRATION_SUPPORT_PROTOCOL.md'),
                  code_hashes={p:digest(ROOT/p) for p in [
                      'scripts/evaluate/benchmark_court_calibration_support.py','src/court/calibration.py',
                      'src/court/court_geometry.py','scripts/evaluate/refine_court_boundary_geometry.py',
                      'scripts/evaluate/probe_court_ridge_ablation.py','scripts/evaluate/benchmark_court_refinement_labels.py']},cases=[])
    for index, case in enumerate(cases):
        assert digest(case['path']) == case['input_sha256']
        available = [i for i,p in enumerate(case['points']) if p is not None]
        points = np.array([case['points'][i] for i in available])
        default = calibrate_court(points,canonical[available])
        alternate = calibrate_court(points,canonical[available],min_inlier_ratio=5/7)
        assert default.is_valid == case['expected']['is_valid']
        assert default.inlier_count == case['expected']['inlier_count']
        if default.is_valid:
            np.testing.assert_allclose(default.image_to_court,case['expected']['homography_matrix'],atol=1e-9,rtol=0)
            np.testing.assert_array_equal(default.image_to_court,alternate.image_to_court)
        row = dict(index=index,id=case['id'],model=case['model'],group=case['group'],
                   input_sha256=case['input_sha256'],available_ids=available,
                   default=default.to_dict(),alternate=alternate.to_dict(),
                   newly_accepted=alternate.is_valid and not default.is_valid)
        predictions = {}
        for name, result in [('default',default),('alternate',alternate)]:
            predictions[name] = cv2.perspectiveTransform(canonical[None],np.linalg.inv(result.image_to_court))[0].tolist() if result.is_valid else [None]*14
        if row['newly_accepted']:
            if case['frame'] is None:
                image = cv2.imread(str(case['path']))
            else:
                capture = cv2.VideoCapture(str(case['path']))
                ok,image = capture.read()
                capture.release()
                assert ok
            scale = 960/image.shape[1]
            reference = cv2.resize(image,(960,round(image.shape[0]*scale)))
            segments = [p['segment'] for p in propose(reference,'no_white') if p['support_fraction'] >= .5]
            corrected,evidence = refine(np.array(predictions['alternate'])*scale,segments,reference.shape[1],reference.shape[0])
            predictions['refined'] = (corrected/scale).tolist()
            row.update(refinement=evidence,native_size=list(image.shape[:2][::-1]))
            panels=[]
            for name in ('alternate','refined'):
                panel=reference.copy()
                for a,b in LINES:
                    cv2.line(panel,tuple(np.rint(np.array(predictions[name][a])*scale).astype(int)),tuple(np.rint(np.array(predictions[name][b])*scale).astype(int)),(30,30,230),2)
                if case['labels'] is not None:
                    for label in case['labels']:
                        cv2.circle(panel,tuple(np.rint(np.array(label)*scale).astype(int)),5,(30,255,30),1)
                cv2.rectangle(panel,(0,0),(960,32),(20,20,20),-1)
                cv2.putText(panel,f"{index} {case['model']} {case['id']} {name}",(5,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                panels.append(panel)
            target=out/f'new_acceptance_{index:03}.jpg'
            assert cv2.imwrite(str(target),np.concatenate(panels),[cv2.IMWRITE_JPEG_QUALITY,70])
            row.update(image=target.name,image_sha256=digest(target),visually_inspected=False)
        row['predictions']=predictions
        if case['labels'] is not None:
            row['scores']={name:score(p,case['labels']) for name,p in predictions.items()}
        report['cases'].append(row)
        if row['newly_accepted']:
            print(json.dumps(dict(index=index,id=case['id'],model=case['model'],inliers=alternate.inlier_count,
                                  refined=row['refinement']['accepted'],tp={k:v['tp'] for k,v in row.get('scores',{}).items()})),flush=True)
    report['complete']=True
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(cases=len(cases),default_accepted=sum(x['default']['is_valid'] for x in report['cases']),
                          alternate_accepted=sum(x['alternate']['is_valid'] for x in report['cases']))),flush=True)


if __name__ == '__main__':
    main()
