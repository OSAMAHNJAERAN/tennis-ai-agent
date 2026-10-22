"""Execute real native crop/classifier/window code with no inference runtime.

Only the crop method is extracted from its heavyweight module. Its body is
unchanged; the import of torch receives an isolated no_grad context stub. Real
native window admission, box validation and classifier dispatch remain active.
"""
import ast
import builtins
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
names = ['src/shot_analysis/pose_feature_extractor.py', 'src/shot_analysis/shot_classifier.py',
         'src/scoring/corrected_shot_rebuild.py', 'src/scoring/corrected_shot_sources.py',
         'src/utils/bbox_utils.py', 'src/shot_analysis/shot_types.py']
def pins():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
before = pins()
from src.scoring.corrected_shot_rebuild import _WindowBoxes
from src.scoring.corrected_shot_sources import _box
from src.shot_analysis.shot_classifier import TennisShotClassifier
from src.shot_analysis.shot_types import PlayerHandedness, ShotClassificationSource, ShotType

tree = ast.parse((ROOT/names[0]).read_text(encoding='utf-8'))
klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PoseFeatureExtractor')
method = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
              and node.name == 'extract_hit_window_features')
method.returns = None
for arg in [*method.args.args, *method.args.kwonlyargs]: arg.annotation = None
stub_imports = []
def no_inference_import(name, *args, **kwargs):
    assert name == 'torch', 'Unexpected import in extracted crop method: '+name
    stub_imports.append(name)
    return SimpleNamespace(no_grad=nullcontext)
env = {'PlayerHandedness':PlayerHandedness,
       '__builtins__':{**vars(builtins), '__import__':no_inference_import}}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
             str(ROOT/names[0]),'exec'), env)
forbidden_forwards = []
def forbidden(image, **kwargs):
    forbidden_forwards.append(True)
    raise AssertionError('Missing/tiny windows must not invoke the model')
extractor = SimpleNamespace(model=SimpleNamespace(predict=forbidden), device='cuda:0')
frames_read = []
class Frames:
    def __len__(self): return 7
    def __getitem__(self,index):
        frames_read.append(index)
        return SimpleNamespace(shape=(960,1280,3))
clock = [{0:{'analysis_id':'fixture','analysis_allowed':True,'segment_id':'camera-0',
    'anchor_frame_index':0,'measurement_transition_allowed':index>0,
    'source_frame_index':index,'source_time_base':'1/30','source_pts':str(index)},
    1:{'analysis_id':'fixture','epoch':0,'revision':0,'observations':[
        {'track_key':'track-A','player_id':'A','state':'CONFIRMED'},
        {'track_key':'track-B','player_id':'B','state':'CONFIRMED'}]}} for index in range(7)]
observations = []
for case in ('all_boxes_absent','all_expanded_crops_too_small'):
    frames_read.clear()
    box = _box(None if case == 'all_boxes_absent' else {
        'bbox':[1.,1.,2.,2.], 'source_track_id':2, 'confidence':.9})
    boxes = _WindowBoxes([box]*7, clock, 3, 3)
    provider_calls = []
    def features(**kwargs):
        provider_calls.append(True)
        return env['extract_hit_window_features'](extractor, **kwargs)
    classifier = TennisShotClassifier(pose_extractor=SimpleNamespace(extract_hit_window_features=features),
        handedness_map={2:PlayerHandedness.RIGHT_HANDED}, court_orientation_map={2:1.})
    result = classifier.classify_shot('PLAYER_2_HIT',3,2,box,None,
        frames=Frames(),all_player_boxes=boxes,is_dead_ball=False)
    assert len(provider_calls) == 1
    assert result[0] == ShotType.UNKNOWN and result[2] == ShotClassificationSource.ABSTENTION_UNKNOWN
    assert not forbidden_forwards
    assert frames_read == ([] if case == 'all_boxes_absent' else list(range(7)))
    observations.append({'case':case,'actual_native_provider_calls':len(provider_calls),
        'actual_model_predict_invocations':len(forbidden_forwards),
        'frames_read':list(frames_read),'native_classification':result[0].value,
        'classification_source':result[2].value,'allowed_window_indices':sorted(boxes.allowed)})
assert stub_imports == ['torch','torch']
assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
after = pins()
assert before == after
receipt = {'status':'PASS','scope':'ACTUAL_NATIVE_CLASSIFIER_WINDOW_BOX_VALIDATION_AND_EXTRACTED_CROP_METHOD_WITH_ONLY_NO_GRAD_STUB; NO_MODEL_OR_GPU_INFERENCE',
    'source_before':before,'source_after':after,'source_unchanged':before==after,
    'observations':observations,'stub_imports':stub_imports,
    'torch_imported':'torch' in sys.modules,'ultralytics_imported':'ultralytics' in sys.modules}
(ART/'zero_forward_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'observations':observations,'source_unchanged':before==after}))
