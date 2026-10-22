"""Read-only pure argument resolution; never import Torch or load a model.

Compile only the pinned runtime's real Model.predict argument dispatcher and
checkpoint-key filter. A fake predictor records final arguments; no inference,
Ultralytics imports, checkpoint deserialization or tracker construction occur.
"""
import ast
import hashlib
import json
from pathlib import Path
import pickletools
import sys
import sysconfig
from types import SimpleNamespace
import zipfile

import yaml

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
PACKAGE = Path(sysconfig.get_paths()['purelib'])/'ultralytics'
model_source = PACKAGE/'engine/model.py'
defaults_source = PACKAGE/'cfg/default.yaml'
checkpoint = ROOT/'models/yolo26/yolo26s-pose.pt'
sources = [model_source, defaults_source, PACKAGE/'engine/predictor.py',
    PACKAGE/'trackers/track.py', PACKAGE/'nn/tasks.py',
    ROOT/'src/api/correction_worker.py', ROOT/'src/shot_analysis/pose_feature_extractor.py',
    ROOT/'src/pipeline/phase6_pipeline.py', ROOT/'src/detection/yolo26_adapters.py']
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
before = {str(path): sha(path) for path in sources}
tree = ast.parse(model_source.read_text(encoding='utf-8'))
model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model')
def extract(name):
    node = next(item for item in model_class.body if isinstance(item, ast.FunctionDef) and item.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in [*node.args.args, *node.args.posonlyargs, *node.args.kwonlyargs]:
        arg.annotation = None
    if node.args.vararg: node.args.vararg.annotation = None
    if node.args.kwarg: node.args.kwarg.annotation = None
    return node
default = yaml.safe_load(defaults_source.read_text(encoding='utf-8'))
env = {'ARGV':['pure-check'], 'get_cfg': lambda base, override:
       SimpleNamespace(**{**vars(base), **override})}
for name in ('predict', '_reset_ckpt_args'):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[extract(name)], type_ignores=[])),
                 str(model_source), 'exec'), env)

# Inspect the relevant literal train_args entry as inert pickle opcodes only.
with zipfile.ZipFile(checkpoint) as archive:
    entry = next(name for name in archive.namelist() if name.endswith('/data.pkl'))
    operations = list(pickletools.genops(archive.read(entry)))
train_index = next(i for i, (_, arg, _) in enumerate(operations) if arg == 'train_args')
img_index = next(i for i in range(train_index+1, len(operations)) if operations[i][1] == 'imgsz')
literal = next(item for item in operations[img_index+1:] if item[0].name not in ('BINPUT','LONG_BINPUT','MEMOIZE'))
assert literal[0].name in ('BININT','BININT1','BININT2','INT')
checkpoint_imgsz = literal[1]
assert checkpoint_imgsz == 640
overrides = env['_reset_ckpt_args']({**default, 'imgsz': checkpoint_imgsz})
overrides.update(model=str(checkpoint), task='pose')
records = []
class Predictor:
    def __init__(self, overrides, _callbacks):
        self.args = SimpleNamespace(**{**default, **overrides})
        self.trackers = []
    def setup_model(self, model, verbose):
        pass
    def __call__(self, source, stream):
        records.append(dict(vars(self.args)))
        return []
fake = SimpleNamespace(overrides=overrides, predictor=None, model=object(), callbacks={},
                       _smart_load=lambda _: Predictor)
# Original full-frame track's supplied kwargs, before the original crop call.
env['predict'](fake, 'full-frame', imgsz=960, conf=.4, classes=[0],
               end2end=True, nms=False, batch=1, mode='track')
env['predict'](fake, 'original-crop', verbose=False, device='cuda:0')
native = records[-1]
keys = ['imgsz','conf','batch','rect','half','max_det','iou','augment','agnostic_nms',
        'classes','end2end','nms','device','mode']
assert {name:native[name] for name in ('imgsz','conf','batch','rect','half','max_det')} == {
    'imgsz':640, 'conf':.25, 'batch':1, 'rect':True, 'half':False, 'max_det':300}
env['predict'](fake, 'current-corrected-crop', imgsz=960, conf=.4, verbose=False,
               device=0, end2end=True, nms=False)
corrected = records[-1]
assert corrected['imgsz'] == 960 and corrected['conf'] == .4
after = {str(path): sha(path) for path in sources}
assert before == after
assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
receipt = {'status':'PASS', 'scope':'PURE_PINNED_RUNTIME_ARGUMENT_DISPATCH_WITH_FAKE_PREDICTOR_NOT_MODEL_INFERENCE',
    'checkpoint_sha256':sha(checkpoint), 'inert_checkpoint_opcode':{
        'archive_entry':entry,'key_offset':operations[img_index][2],
        'integer_offset':literal[2],'imgsz':checkpoint_imgsz},
    'original_crop_args':{key:native[key] for key in keys},
    'current_correction_args':{key:corrected[key] for key in keys},
    'effective_original_end2end':True,
    'end2end_explanation':'Original crop args inherit default None; the already selected model head remains end2end=True.',
    'tracking_callback_limit':'Argument dispatch only. Original .track callbacks persist on crop .predict and may reorder/filter its Results. Fresh correction model has no tracker callbacks.',
    'source_pins':before,'source_unchanged':before==after,
    'torch_imported':'torch' in sys.modules,'ultralytics_imported':'ultralytics' in sys.modules}
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:receipt[key] for key in ('status','original_crop_args','current_correction_args','source_unchanged')}))
