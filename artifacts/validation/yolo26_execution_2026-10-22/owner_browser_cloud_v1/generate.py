"""Prepare an explicitly synthetic owner session for the real browser workflow.

Only existing test fixture factory AST is cloned. Real product parsers, codecs,
rules, result storage, API/auth and immutable generation writers are unchanged.
No server or worker is started by this preparation program.
"""
import ast
import builtins
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
MATCH_ID = 'match_7a26c0de7a26c0de7a26c0de7a26c0de'
LABEL = 'SYNTHETIC SOFTWARE TEST - correction workflow.mp4'
OWNER = {'username': 'synthetic_review_owner_v1', 'password': 'Synthetic-only-review-2026!'}
PORT = 0  # Server binds a fresh task-owned ephemeral loopback port.
WORK = ART/'work-v2'
STORE = ART/'service-v2'
TEMP = ART/'temp-v2'
CONFIG = ART/'unused-original-inference.yaml'
os.chdir(ROOT)
TEMP.mkdir(exist_ok=False)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(ART/'bootstrap'),
    PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
    TENNIS_REQUIRE_AUTH='1', TENNIS_SECURE_COOKIES='0')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
PATHS = sorted({*ROOT.joinpath('src').rglob('*.py'), *ROOT.joinpath('tests').rglob('*.py'),
                *(p for p in ROOT.joinpath('frontend/src').rglob('*') if p.is_file()),
                ROOT/'frontend/package.json', ROOT/'frontend/pnpm-lock.yaml', Path(__file__).resolve()})
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()
def pins():
    return {str(path.relative_to(ROOT)): sha(path) for path in PATHS}
before = pins()
attempts = []
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'torchvision', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Synthetic browser fixture must not load inference: '+name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
started = time.perf_counter()
factories, encoding = [], []

def browser_source(path):
    """Actually encode generated toy imagery before its source SHA is bound."""
    path = Path(path)
    temporary = path.with_name(path.stem+'.browser-source.mp4')
    executable = shutil.which('ffmpeg')
    assert executable, 'FFmpeg is required for actual browser-playable source'
    original_sha = sha(path)
    args = [executable, '-nostdin', '-n', '-v', 'error', '-xerror', '-i', str(path),
        '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, result.stderr
    assert temporary.is_file() and temporary.stat().st_size > 0
    # Both paths belong to this newly generated, private work directory. No
    # immutable generation has referenced these source bytes yet.
    assert path.parent == WORK/'analysis' and temporary.parent == path.parent
    temporary.replace(path)
    probe = subprocess.run([shutil.which('ffprobe'), '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_read_frames',
        '-of', 'json', str(path)], capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert probe.returncode == 0, probe.stderr
    metadata = json.loads(probe.stdout)['streams'][0]
    assert metadata == {'codec_name': 'h264', 'width': 1280, 'height': 960,
                        'avg_frame_rate': '30/1', 'nb_read_frames': '12'}
    encoding.append({'input_sha256': original_sha, 'bound_h264_source_sha256': sha(path),
        'command': args, 'probe': metadata})

class FixtureTransform(ast.NodeTransformer):
    def __init__(self, function):
        self.function = function
        self.literal_changes = []
        self.inserted_h264 = 0
        self.browser_copy_slot = 0

    def visit_Constant(self, node):
        replacement = MATCH_ID if node.value == 'fixture' else LABEL if node.value == 'Tennis <script>alert(1)</script> é 🎾.mp4' else None
        if replacement is None:
            return node
        self.literal_changes.append({'line': node.lineno, 'column': node.col_offset,
                                     'before': node.value, 'after': replacement})
        return ast.copy_location(ast.Constant(value=replacement), node)

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if (self.function == 'build_shot_base' and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == 'fixture' and node.targets[0].attr == 'source_sha'):
            self.inserted_h264 += 1
            call = ast.Expr(value=ast.Call(func=ast.Name(id='_browser_source', ctx=ast.Load()),
                args=[ast.Name(id='replacement_source', ctx=ast.Load())], keywords=[]))
            return [ast.copy_location(call, node), node]
        return node

    def visit_Tuple(self, node):
        node = self.generic_visit(node)
        if (self.function == 'build_complete_base' and
                all(isinstance(item, ast.Constant) for item in node.elts) and
                tuple(item.value for item in node.elts) == ('annotated.mp4', 'original.mp4')):
            node.elts.append(ast.Constant(value='annotated_web.mp4'))
            self.browser_copy_slot += 1
        return node

def clone_factory(module_name, function_name, overrides):
    module = importlib.import_module(module_name)
    source_path = Path(module.__file__)
    source = source_path.read_text(encoding='utf-8')
    module_ast = ast.parse(source)
    original = next(node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    original_ast = ast.dump(original, include_attributes=False)
    transformed = copy.deepcopy(original)
    transformer = FixtureTransform(function_name)
    transformed = transformer.visit(transformed)
    transformed.decorator_list = []
    ast.fix_missing_locations(transformed)
    assert transformer.literal_changes
    assert transformer.inserted_h264 == (1 if function_name == 'build_shot_base' else 0)
    assert transformer.browser_copy_slot == (1 if function_name == 'build_complete_base' else 0)
    namespace = dict(vars(module))
    namespace.update(overrides)
    namespace['_browser_source'] = browser_source
    tree = ast.Module(body=[transformed], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(ART/(function_name+'.private_ast.py')), 'exec'), namespace)
    code = ast.unparse(transformed)+'\n'
    output = ART/'factory_ast'/(function_name+'.py')
    output.parent.mkdir(exist_ok=True)
    output.write_text(code, encoding='utf-8')
    factories.append({'source_module': module_name, 'source_file': str(source_path.relative_to(ROOT)),
        'source_sha256': sha(source_path), 'function': function_name,
        'original_ast_sha256': hashlib.sha256(original_ast.encode()).hexdigest(),
        'private_ast_sha256': hashlib.sha256(ast.dump(transformed, include_attributes=False).encode()).hexdigest(),
        'private_source_sha256': sha(output), 'literal_changes': transformer.literal_changes,
        'injected_owned_h264_encoding_calls': transformer.inserted_h264,
        'added_original_browser_copy_slots': transformer.browser_copy_slot,
        'private_namespace_overrides': sorted(overrides)})
    return namespace[function_name]

try:
    resources = {'source': 'explicitly synthetic software fixture; no host admission probe',
                 'ram': 8*1024**3, 'disk': 8*1024**3, 'rss': 0, 'peak_rss': 0, 'vram': None}
    # No actual host/GPU resource probe or inference is authorized by this fixture.
    assert not STORE.exists() and not WORK.exists(), 'Never overwrite an existing browser fixture'
    WORK.mkdir()
    base = clone_factory('tests.test_corrected_rule_rebuild', 'build_sealed_base', {})
    shot = clone_factory('tests.test_corrected_shot_rebuild', 'build_shot_base', {'build_sealed_base': base})
    complete = clone_factory('tests.test_corrected_preparation', 'build_complete_base', {'build_shot_base': shot})
    fixture = complete(WORK)
    assert fixture.snapshot.analysis_id == MATCH_ID and fixture.snapshot.frames == 12
    assert fixture.snapshot.file_sha256('source.mp4') == encoding[0]['bound_h264_source_sha256']
    assert fixture.snapshot.resolve_file('annotated_web.mp4') is not None
    CONFIG.write_text('# EXPLICIT SYNTHETIC BROWSER FIXTURE; no original inference is queued.\n', encoding='utf-8')
    (ART/'empty-demos.json').write_text('{"schemaVersion":1,"sources":[]}\n', encoding='utf-8')
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    from src.api.jobs import Settings
    from src.api.result_document import ResultDocument
    from src.scoring.rule_reproduction import reproduce_rule_timeline
    from src.api.corrections import CorrectionStore
    settings = Settings(STORE, CONFIG, require_auth=True, secure_cookies=False)
    app = create_app(settings, start_worker=False, demos_manifest=ART/'empty-demos.json', frontend_root=ROOT/'frontend/dist')
    jobs = app.state.store
    with TestClient(app) as owner, TestClient(app) as guest, TestClient(app) as other:
        registered = owner.post('/api/auth/register', json=OWNER)
        assert registered.status_code == 201, registered.text
        owner_id = registered.json()['user']['id']
        foreign = other.post('/api/auth/register', json={'username': 'synthetic_other_owner_v1',
            'password': 'Synthetic-only-other-2026!'})
        assert foreign.status_code == 201, foreign.text
        jobs.create(MATCH_ID, LABEL, '.mp4', fixture.snapshot.resolve_file('original.mp4').stat().st_size,
            CONFIG, owner_id=owner_id)
        jobs.finish(MATCH_ID)
        output = jobs.output_dir(MATCH_ID)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fixture.directory, output)
        base_url = '/api/analyses/'+MATCH_ID
        response = owner.get(base_url+'/result')
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['summary']['id'] == MATCH_ID and result['summary']['label'] == LABEL
        assert result['analysisRevision'] == 0 and result['analysisGeneration'] == fixture.snapshot.generation
        assert result['provenance']['productionQualified'] is False
        assert result['pointAnalysis']['analysisId'] == MATCH_ID
        review = owner.get(base_url+'/review-source', params={'generation': fixture.snapshot.generation})
        assert review.status_code == 200, review.text
        source = review.json()
        assert source['analysis_id'] == MATCH_ID and source['initial_state']['fields']['match_id'] == MATCH_ID
        assert source['is_current'] and source['total_inputs'] == 3
        assert source['initial_state']['fields']['points_p2'] == 0
        assert len(source['inputs']) == 3
        with ResultDocument(output, MATCH_ID) as document:
            snapshot = document.snapshot
            assert document.value['summary']['frames'] == 12
        replay_scratch = ART/'verified-reproduction'
        replay_scratch.mkdir()
        with reproduce_rule_timeline(snapshot.resolve_file('native_rule_replay.json'), replay_scratch,
                artifact_sha256=snapshot.file_sha256('native_rule_replay.json'),
                source_sha256=snapshot.file_sha256('source.mp4'), analysis_id=MATCH_ID,
                analysis_revision=0) as reproduction:
            assert reproduction.engine.state.points_p2 == 0
        assert owner.get(base_url+'/media/source').content == snapshot.resolve_file('source.mp4').read_bytes()
        assert owner.get(base_url+'/media/annotated').content == snapshot.resolve_file('annotated_web.mp4').read_bytes()
        assert owner.get(base_url+'/corrections').json() == []
        for client, code in ((guest, 401), (other, 404)):
            for url in (base_url+'/result', base_url+'/review-source', base_url+'/media/source', base_url+'/corrections'):
                assert client.get(url).status_code == code, url
        assert CorrectionStore(jobs).claim() is None
    example = {key: value for key, value in fixture.request.items() if key not in {'analysis_id', 'reviewer_id'}}
    (ART/'request_example.json').write_text(json.dumps(example, indent=2)+'\n', encoding='utf-8')
    (ART/'initial_review_source.json').write_text(json.dumps(source, indent=2)+'\n', encoding='utf-8')
    credentials = {**OWNER, 'owner_id': owner_id,
        'scope': 'DISPOSABLE LOCAL SYNTHETIC TEST ACCOUNT ONLY; NOT A USER SECRET OR DEPLOYMENT CREDENTIAL'}
    (ART/'synthetic_credentials.json').write_text(json.dumps(credentials, indent=2)+'\n', encoding='utf-8')
    after = pins()
    assert before == after and not attempts and not {'torch', 'torchvision', 'ultralytics'} & set(sys.modules)
    receipt = {'status': 'PREPARED_VERIFIED_SYNTHETIC_FIXTURE_SERVER_NOT_STARTED',
        'created_at': datetime.now(timezone.utc).isoformat(), 'analysis_id': MATCH_ID,
        'label': LABEL, 'base_generation': fixture.snapshot.generation, 'base_revision': 0,
        'storage_root': str(STORE), 'config': str(CONFIG), 'frontend_root': str(ROOT/'frontend/dist'),
        'port_checked_available': PORT, 'host': '127.0.0.1', 'owner_id': owner_id,
        'analysis_url': f'http://127.0.0.1:{PORT}/analysis/{MATCH_ID}',
        'historical_url': f'http://127.0.0.1:{PORT}/analysis/{MATCH_ID}?generation={fixture.snapshot.generation}',
        'credentials_file': 'synthetic_credentials.json', 'request_example': 'request_example.json',
        'factory_transformations': factories, 'actual_source_encoding': encoding,
        'frames': 12, 'width': 1280, 'height': 960, 'fps': 30, 'duration_seconds': .4,
        'source_sha256': snapshot.file_sha256('source.mp4'),
        'generation_file_hashes': {name: dict(record) for name, record in snapshot._files.items()},
        'native_rule_reproduction_verified': True, 'owner_result_review_media_verified': True,
        'guest_foreign_owner_rejected': True, 'source_before': before, 'source_after': after,
        'source_stable': before == after, 'resources_before_imports': resources,
        'inference_import_attempts': attempts, 'inference_loaded': sorted({'torch', 'torchvision', 'ultralytics'} & set(sys.modules)),
        'scope': 'SYNTHETIC OBSERVATIONS AND GREY TEST VIDEO; REAL NATIVE RULES/STORAGE/AUTH/HTTP; NO DETECTOR OR INDEPENDENT ACCURACY CLAIM',
        'actual_host_ram_gpu_probes': 0, 'actual_model_inference': False,
        'adapted_from': 'artifacts/validation/yolo26_execution_2026-10-22/correction_browser_fixture_v1/generate.py',
        'adapted_from_sha256': sha(ROOT/'artifacts/validation/yolo26_execution_2026-10-22/correction_browser_fixture_v1/generate.py'),
        'server_started': False, 'worker_started': False, 'seconds': time.perf_counter()-started}
    (ART/'fixture_manifest.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in ('status', 'analysis_id', 'base_generation', 'analysis_url',
        'source_stable', 'inference_import_attempts', 'seconds')}))
except BaseException as error:
    (ART/'generation_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
finally:
    builtins.__import__ = original_import
