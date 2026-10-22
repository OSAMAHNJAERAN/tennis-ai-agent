"""Actual line-call method with an explicit synthetic metric point, no media."""
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
class RefuseModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('torch', 'ultralytics'):
            raise RuntimeError('Rule probe refuses model import: ' + fullname)
sys.meta_path.insert(0, RefuseModels())

import numpy as np
from src.line_calling.line_call_engine import TennisLineCallEngine, LineCallContext
from src.line_calling.line_geometry import ServiceBoxType
from src.line_calling.contact_refinement import RefinedContactPoint
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState

path = ROOT / 'src/line_calling/line_call_engine.py'
digest = lambda value: hashlib.sha256(value.read_bytes()).hexdigest()
before = digest(path)
class SyntheticRefiner:
    def refine_bounce_contact(self, frame, trajectory, raw_frames):
        return RefinedContactPoint(0, 0., 0., 3., 6., 1., 'EXPLICIT_SYNTHETIC_METRIC_POINT', 3., 6., 0., 0.)
engine = TennisLineCallEngine(refiner=SyntheticRefiner())
points = [TemporalBallPoint(0, 0., 3., 6., state=BallState.DETECTED)]
observed = {}
for name, context, target in (
    ('missing_target', LineCallContext.SERVE, None),
    ('explicit_correct_target', LineCallContext.SERVE, ServiceBoxType.FAR_DEUCE),
    ('rally_context', LineCallContext.RALLY, None)):
    observed[name] = engine.evaluate_bounce(1, 0, points, np.eye(3), context, target).to_dict()
assert digest(path) == before
assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
receipt = {'status': 'MISSING_SERVICE_TARGET_SILENT_DEFAULT_REPRODUCED',
    'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': before,
    'source_unchanged': True, 'observed': observed,
    'fixture_point_m': [3., 6.], 'fixture_correct_box': 'FAR_DEUCE',
    'source_behavior': 'target_service_box or ServiceBoxType.NEAR_DEUCE',
    'expected_without_eligible_identity_end_or_target': 'UNKNOWN_OR_REVIEW_REQUIRED_NOT_INVENTED_FAULT',
    'synthetic_identity_homography_and_refinement_not_geometric_GT': True,
    'torch_imported': False, 'ultralytics_imported': False, 'model_used': False,
    'media_used': False, 'training_performed': False, 'qualification_evidence': False,
    'script_sha256': digest(Path(__file__))}
out = Path(__file__).with_name('missing_service_target_receipt.json')
with out.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'receipt': str(out), 'decisions': {key: value['decision'] for key, value in observed.items()}}))
assert observed['missing_target']['decision'] in ('UNKNOWN', 'REVIEW_REQUIRED'), 'Missing target fabricates a service decision'
