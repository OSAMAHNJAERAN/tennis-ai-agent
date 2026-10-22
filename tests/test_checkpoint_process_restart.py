"""Two actual Python processes prove completed-stage finalization recovery only.

No actual video, model inference, physical accuracy or mid-inference resume is
exercised. The child replaces those boundaries with marked synthetic fixtures.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from src.api.analysis_resume import REQUIRED_PIPELINE_FILES, OPTIONAL_PIPELINE_FILES
from src.api.jobs import JobStore
from src.api.result_document import ResultDocument
from src.api.result_storage import DiskSeries


SCOPE = 'SYNTHETIC_PROCESS_CRASH_STORAGE_ONLY_NO_MEDIA_OR_INFERENCE'
CHILD = Path(__file__).parent / 'helpers/checkpoint_worker_child.py'


def test_os_process_loss_after_checkpoint_resumes_same_job_without_repeating_pipeline(tmp_path):
    root = tmp_path / 'storage'
    config = tmp_path / 'synthetic_config.json'
    config.write_text(json.dumps({'model_system': {'family': 'yolo26'}, 'scope': SCOPE}),
                      encoding='utf-8')
    original_bytes = (SCOPE + '\nORIGINAL_NOT_PLAYABLE_MEDIA').encode('ascii')
    store = JobStore(root)
    analysis_id = store.new_id()
    store.create(analysis_id, 'synthetic-process-crash.mp4', '.mp4', len(original_bytes),
                 config, owner_id='process-restart-owner')
    original = store.upload_dir(analysis_id) / 'original.mp4'
    original.parent.mkdir(parents=True)
    original.write_bytes(original_bytes)
    config_bytes = config.read_bytes()
    output = store.output_dir(analysis_id)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')

    def run_child(mode):
        # subprocess.run kills and waits for this owned child if timeout fires.
        return subprocess.run([sys.executable, str(CHILD.resolve()), mode, str(root),
                               str(config), analysis_id], capture_output=True, text=True,
                              timeout=30, env=env, cwd=Path(__file__).resolve().parents[1],
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    first = run_child('crash')
    assert first.returncode == 73, (first.returncode, first.stdout, first.stderr)
    interrupted = JobStore(root).get(analysis_id)
    assert interrupted['id'] == analysis_id and interrupted['owner_id'] == 'process-restart-owner'
    assert interrupted['status'] == 'PROCESSING' and interrupted['attempt'] == 1
    assert interrupted['completed_at'] is None
    assert not (output / '.result/current.store').exists()
    assert not (output / 'result.json').exists()
    assert not (output / 'annotated_web.mp4').exists()
    marker = output / '.pipeline_checkpoint.json'
    marker_bytes = marker.read_bytes()
    manifest = json.loads(marker_bytes)
    assert manifest['stage'] == 'PHASE6_COMPLETED'
    assert manifest['identity']['analysis_id'] == analysis_id
    assert manifest['identity']['owner_id'] == 'process-restart-owner'
    assert manifest['identity']['original_sha256'] == hashlib.sha256(original_bytes).hexdigest()
    assert set(manifest['requiredFiles']) == set(REQUIRED_PIPELINE_FILES)
    before = {name: (output / name).read_bytes() if (output / name).exists() else None
              for name in REQUIRED_PIPELINE_FILES + OPTIONAL_PIPELINE_FILES}
    assert all(before[name] is not None for name in REQUIRED_PIPELINE_FILES)
    # The actual timing parser consumed two valid synthetic FFmpeg stats rows.
    timing = json.loads(before['media_metadata.json'])
    assert timing['timeMapping']['frames'] == 2
    assert timing['timeMapping']['status'] == 'CAPTURED_FFMPEG_FRAME_MAP'
    assert len((output / 'source_frame_mapping.jsonl').read_text(encoding='utf-8').splitlines()) == 2

    second = run_child('resume')
    assert second.returncode == 0, (second.returncode, second.stdout, second.stderr)
    restarted = JobStore(root)
    completed = restarted.get(analysis_id)
    assert completed['status'] == 'COMPLETED' and completed['attempt'] == 2
    assert completed['completed_at'] and completed['error'] is None
    assert completed['id'] == analysis_id and completed['owner_id'] == 'process-restart-owner'
    assert marker.read_bytes() == marker_bytes
    assert {name: (output / name).read_bytes() if (output / name).exists() else None
            for name in before} == before
    assert original.read_bytes() == original_bytes and config.read_bytes() == config_bytes
    calls = [json.loads(line) for line in (root / 'pipeline_calls.log').read_text(encoding='utf-8').splitlines()]
    assert len(calls) == 1 and calls[0]['analysis_id'] == analysis_id and calls[0]['scope'] == SCOPE
    observations = [json.loads(line) for line in (root / 'child_observations.jsonl').read_text(encoding='utf-8').splitlines()]
    pids = {row['pid'] for row in observations}
    assert len(pids) == 2 and os.getpid() not in pids
    claims = [row for row in observations if row['event'] == 'claimed']
    assert [row['attempt'] for row in claims] == [1, 2]
    assert all(row['analysis_id'] == analysis_id and row['scope'] == SCOPE for row in observations)
    assert [row['event'] for row in observations].count('synthetic_normalization') == 1
    assert [row['event'] for row in observations].count('synthetic_pipeline_returned') == 1
    assert [row['event'] for row in observations].count('synthetic_result_builder') == 1
    publication = [row for row in observations if row['event'] == 'published_database_opened_before_completion']
    assert len(publication) == 1 and publication[0]['attempt'] == 2
    guarded = [row for row in observations if 'forbidden_loaded' in row]
    assert len(guarded) == 3
    assert all(row['forbidden_loaded'] == [] and row['blocked_imports'] == [] for row in guarded)
    with restarted.connect() as connection:
        analyses = connection.execute('SELECT id,owner_id FROM analyses').fetchall()
        matches = connection.execute('SELECT id,analysis_id,owner_id FROM matches').fetchall()
    assert [tuple(row) for row in analyses] == [(analysis_id, 'process-restart-owner')]
    assert [tuple(row) for row in matches] == [(analysis_id, analysis_id, 'process-restart-owner')]
    with ResultDocument(output, analysis_id) as document:
        assert document.storage is not None and document.value['accuracyQualified'] is False
        frames = document.value['syntheticFrames']
        assert isinstance(frames, DiskSeries)
        assert [row['frame_index'] for row in frames] == [0, 1]
        assert [row['id'] for row in frames] == [f'{analysis_id}:frame:0', f'{analysis_id}:frame:1']
        assert all(row['analysisId'] == analysis_id and row['scope'] == SCOPE for row in frames)
        expected_result = json.loads(''.join(document.json_chunks()))
    assert json.loads((output / 'result.json').read_text(encoding='utf-8')) == expected_result
    assert len(list((output / '.result').glob('*.sqlite3'))) == 1
    assert restarted.claim() is None
    evidence = {'scope': SCOPE, 'analysis_id': analysis_id, 'process_ids': sorted(pids),
                'child_exit_codes': [first.returncode, second.returncode],
                'attempts': [1, 2], 'pipeline_calls': len(calls),
                'marker_sha256': hashlib.sha256(marker_bytes).hexdigest(),
                'required_artifacts': len(REQUIRED_PIPELINE_FILES),
                'sealed_artifact_bytes_unchanged': True, 'original_and_config_unchanged': True,
                'result_database_opened': True, 'logical_frame_rows': 2,
                'forbidden_imports_loaded': [], 'accuracy_qualified': False}
    (tmp_path / 'process_restart_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
