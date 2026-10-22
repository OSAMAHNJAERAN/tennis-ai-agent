"""Bind retained execution/review evidence without inference or resource probes."""
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
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
            'sha256': digest}


def main():
    if set(path.name for path in ART.iterdir()) != {'build_receipt.py'}:
        raise RuntimeError('Retain existing closure outputs and use a new directory')
    final_path = BASE / 'native_inference_integration_checks_v3/receipt.json'
    final = read(final_path)
    if binding(final_path)['sha256'] != 'a007ad6413cb22fc84c661da8d6e0a4fa7244511c1f76429117bd96f9cfa08bb':
        raise RuntimeError('The reviewed final receipt differs')
    assert final['status'] == 'PASS' and final['exit_code'] == 0
    assert final['counts'] == {'tests': 484, 'errors': 0, 'failures': 0, 'skipped': 0}
    assert final['source_stable'] and final['source_before'] == final['source_after']
    assert len(final['source_before']) == 193 and final['source_python_files'] == 168
    assert final['inference_import_attempts'] == final['forbidden_inference_loaded'] == []
    assert all(final[key] for key in ('guard_controls_valid', 'workflow_imports_valid',
                                    'recovery_imports_valid', 'child_imports_valid', 'process_recovery_valid'))
    current = {name: binding(ROOT / name)['sha256'] for name in final['source_before']}
    assert current == final['source_before']

    storage_path = BASE / 'native_inference_storage_checks_v3/receipt.json'
    storage = read(storage_path)
    assert storage['status'] == 'PASS' and storage['counts']['tests'] == 270
    old = storage['source_before']
    added = {name: digest for name, digest in current.items() if name.startswith('src/') and name not in old}
    changed = {name: {'before': old[name], 'after': digest} for name, digest in current.items()
               if name.startswith('src/') and name in old and digest != old[name]}
    removed = [name for name in old if name.startswith('src/') and name not in current]
    assert set(added) == {'src/utils/native_inference_session.py'}
    assert set(changed) == {'src/api/worker.py', 'src/pipeline/phase6_pipeline.py'} and not removed

    paths = [final_path, final_path.with_name('junit.xml'), final_path.with_name('started.json'),
             final_path.with_name('run_checks.py'), storage_path,
             BASE / 'native_inference_integration_checks_v1/receipt.json',
             BASE / 'native_inference_integration_checks_v2/receipt.json',
             BASE / 'player_crop_transition_checks_v4/receipt.json', Path(__file__).resolve()]
    paths.extend(sorted((BASE / 'native_inference_storage_review_v1').glob('*.json')))
    paths.extend(sorted((BASE / 'native_inference_integration_review_v1').glob('*.json')))
    paths.extend(ROOT / name for name in ('docs/plans/NATIVE_INFERENCE_RESUME.md',
        'docs/plans/YOLO26_EXECUTION_PROGRESS.md', 'docs/PRODUCT_WORKFLOW.md',
        'docs/plans/ANALYTICS_CAPACITY_SEAM_2026-10-16.md'))
    process_directory = final_path.parent / 'temp/pytest/test_hard_process_loss_resumes0'
    assert read(process_directory / 'native_worker_process_recovery_evidence.json') == final['actual_worker_process_recovery'][0]
    paths.extend(path for path in sorted(process_directory.rglob('*')) if path.is_file())
    records = [binding(path) for path in sorted(set(paths))]
    assert records == [binding(ROOT / value['path']) for value in records]
    receipt = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'SOFTWARE_NATIVE_BALL_RESUME_VERIFIED_MASTER_INCOMPLETE',
        'final_counts': final['counts'], 'unchanged_final_pins': len(current),
        'production_delta_since_storage_snapshot': {'added': added, 'changed': changed, 'removed': removed},
        'actual_process_recovery': final['actual_worker_process_recovery'],
        'all_final_source_test_config_runner_pins': current, 'retained_evidence_and_documents': records,
        'preserved_attempts': {'integration_v1': '480 passes, two Windows UTF-8 failures and oversized test-ID setup/teardown errors',
            'integration_v2': '483 passes and a probe incorrectly counting ffprobe as normalization',
            'integration_v3': '484 passes; test-only encoding/IDs/command classification repaired'},
        'limits': {'selected_model_results_and_asset_identity': 'EXPLICIT_SYNTHETIC_BOUNDARIES',
            'video_frames': 8, 'actual_neural_runtime_imports_loads_forwards': 0,
            'host_ram_gpu_admission_probes_or_training': False,
            'actual_selected_model_crop_or_mid_tile_process_recovery_qualified': False,
            'independent_cv_speed_accuracy_qualified': False, 'real_long_match_capacity_qualified': False,
            'live_operation_qualified': False, 'windows_power_loss_qualified': False,
            'master_goal_complete': False, 'waiting_for_human_ram_completion_notification': True}}
    with (ART / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': binding(ART / 'receipt.json'), 'evidence_bindings': len(records),
                      'source_pins': len(current), 'status': receipt['status']}, indent=2))


if __name__ == '__main__':
    main()
