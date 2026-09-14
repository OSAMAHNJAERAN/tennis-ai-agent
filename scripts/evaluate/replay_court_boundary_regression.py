"""Development regression harness: accepted refinements must retain prior matches."""
import argparse
import importlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scripts.evaluate.probe_court_line_segments import propose

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--implementation', default='scripts.evaluate.refine_court_line_geometry')
    args=parser.parse_args()
    refine=importlib.import_module(args.implementation).refine
    report=json.loads((ROOT/'outputs/vision_upgrade_audit/court_refinement_selection01/report.json').read_text())
    failures=[]
    for identity in ('5QObSWGBQB8_1200','ktiDOhLZIVs_3500'):
        row=next(r for r in report['images'] if r['id']==identity)
        previous=row['scores']['projected']['landmarks']
        points=np.array([p['prediction'] for p in previous])*.75
        image=cv2.imread(str(ROOT/'data/external/court_heatmap_pilot/images'/f'{identity}.png'))
        segments=[r['segment'] for r in propose(cv2.resize(image,(960,540))) if r['support_fraction']>=.5]
        refined,evidence=refine(points,segments)
        lost=[i for i,p in enumerate(previous) if p['correct'] and np.linalg.norm(refined[i]/.75-p['label'])>7]
        print(json.dumps(dict(id=identity,accepted=evidence['accepted'],lost=lost)),flush=True)
        failures.extend((identity,i) for i in lost)
    assert not failures, f'Previously correct boundary landmarks regressed: {failures}'

if __name__=='__main__':main()
