"""Bind mid-tile execution and text-only source work, without runtime probes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
BASE = ART.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def binding(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest}


def main():
    if set(path.name for path in ART.iterdir()) != {'build_receipt.py'}:
        raise RuntimeError('Preserve existing aggregate and use a new directory')
    final_path = BASE / 'native_inference_midtile_checks_v1/receipt.json'
    final = read(final_path)
    assert binding(final_path)['sha256'] == 'c31a78e7085e999eb151c4098f469c1429f61a01593b30eb766ffd29d8b2861b'
    assert final['status'] == 'PASS' and final['exit_code'] == 0
    assert final['counts'] == {'tests': 485, 'errors': 0, 'failures': 0, 'skipped': 0}
    assert len(final['source_before']) == 193 and final['source_python_files'] == 168
    assert final['source_stable'] and final['source_before'] == final['source_after']
    assert final['inference_import_attempts'] == final['forbidden_inference_loaded'] == []
    assert all(final[key] for key in ('guard_controls_valid', 'workflow_imports_valid',
                                    'recovery_imports_valid', 'child_imports_valid', 'process_recovery_valid'))
    assert all(binding(ROOT / name)['sha256'] == digest for name, digest in final['source_before'].items())
    before_path = BASE / 'native_inference_integration_checks_v3/receipt.json'
    before = read(before_path)['source_before']
    current = final['source_before']
    changed = {name: {'before': before[name], 'after': digest} for name, digest in current.items()
               if name in before and before[name] != digest}
    assert set(changed) == {'tests/helpers/native_pipeline_resume_child.py', 'tests/test_native_pipeline_process_resume.py'}
    assert {name: digest for name, digest in current.items() if name.startswith('src/')} == {
        name: digest for name, digest in before.items() if name.startswith('src/')}
    paths = [final_path, final_path.with_name('junit.xml'), final_path.with_name('started.json'),
             final_path.with_name('run_checks.py'), before_path,
             BASE / 'native_inference_resume_closure_v1/receipt.json', Path(__file__).resolve()]
    paths.extend((BASE / 'native_inference_midtile_review_v1').glob('*.json'))
    process_evidence = sorted(final_path.parent.rglob('native_worker_process_recovery_evidence.json'))
    assert len(process_evidence) == 2
    cases = {value['recovery_case']: value for value in final['actual_worker_process_recovery']}
    assert set(cases) == {'after_two_complete_frames', 'during_first_tile_of_frame_2'}
    for evidence in process_evidence:
        value = read(evidence)
        assert cases[value['recovery_case']] == value
        paths.extend(path for path in evidence.parent.rglob('*') if path.is_file())
    documentation = ('docs/plans/NATIVE_INFERENCE_RESUME.md', 'docs/plans/YOLO26_EXECUTION_PROGRESS.md',
        'docs/PRODUCT_WORKFLOW.md', 'docs/plans/ANALYTICS_CAPACITY_SEAM_2026-10-16.md',
        'docs/data/INDEPENDENT_SOURCE_ACQUISITION.md',
        'docs/research/TENNIS_SPEED_REFERENCE_FOLLOWUP_2026-10-22.md',
        'docs/research/INDEPENDENT_TENNIS_MATCH_SOURCES_2026-10-22.md')
    paths.extend(ROOT / name for name in documentation)
    for name in ('independent_speed_source_followup_v1', 'independent_speed_source_followup_v2',
                 'independent_match_source_followup_v1', 'independent_match_uploader_v1'):
        paths.extend(path for path in (BASE / name).rglob('*') if path.is_file())
    records = [binding(path) for path in sorted(set(paths))]
    assert records == [binding(ROOT / value['path']) for value in records]
    receipt = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'MID_TILE_SOFTWARE_RECOVERY_VERIFIED_MASTER_INCOMPLETE',
        'counts': final['counts'], 'unchanged_pins': len(current),
        'production_source_unchanged_since_484_check_run': True,
        'two_process_test_file_delta': changed, 'current_final_source_pins': current,
        'actual_process_recovery': list(cases.values()), 'retained_evidence_and_documents': records,
        'limits': {'actual_neural_runtime_imports_loads_forwards': 0,
            'selected_vision_and_asset_identity': 'EXPLICIT_SYNTHETIC_BOUNDARIES', 'synthetic_frames_per_case': 8,
            'actual_selected_model_mid_tile_recovery_qualified': False,
            'independent_accuracy_speed_long_match_or_live_qualified': False,
            'windows_power_loss_qualified': False, 'new_source_media_or_target_downloads': False,
            'training_or_ram_gpu_admission_probes': False, 'master_goal_complete': False,
            'waiting_for_human_ram_completion_notification': True}}
    with (ART / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': binding(ART / 'receipt.json'), 'evidence_bindings': len(records),
                      'counts': final['counts'], 'status': receipt['status']}, indent=2))


if __name__ == '__main__':
    main()
