"""Bind the implemented crop transition and sequential decoder software scope."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DAY = HERE.parent


def binding(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    output = HERE / 'receipt.json'
    if output.exists():
        raise FileExistsError('Preserve the completed closure receipt')
    checks_path = DAY / 'player_crop_transition_checks_v4/receipt.json'
    checks = json.loads(checks_path.read_text())
    assert checks['status'] == 'PASS' and checks['counts'] == {
        'tests': 132, 'errors': 0, 'failures': 0, 'skipped': 0}
    assert checks['source_stable'] and not checks['inference_import_attempts']
    assert not checks['forbidden_inference_loaded']
    for name, expected in checks['source_after'].items():
        assert binding(ROOT / name)['sha256'] == expected, name
    baseline = DAY / 'correction_json_wire_checks_v1/sources'
    changed = []
    delta = []
    for path in sorted((ROOT / 'src').rglob('*.py')):
        name = path.relative_to(ROOT).as_posix()
        old = baseline / name
        if old.read_bytes() != path.read_bytes():
            changed.append(name)
            delta.extend(difflib.unified_diff(old.read_text().splitlines(keepends=True),
                path.read_text().splitlines(keepends=True), fromfile='before/' + name, tofile='after/' + name))
    assert changed == ['src/detection/yolo26_adapters.py', 'src/pipeline/phase6_pipeline.py',
                       'src/utils/video_frame_sequence.py'], changed
    patch = HERE / 'post_correction_source_delta.patch'
    with patch.open('x', encoding='utf-8') as stream:
        stream.write(''.join(delta))
    ultra = Path(r'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\Lib\site-packages\ultralytics')
    runtime_sources = []
    for relative in ('engine/model.py', 'engine/predictor.py', 'trackers/track.py'):
        path = ultra / relative
        raw = path.read_bytes()
        snapshot = HERE / 'installed_primary_source' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open('xb') as stream:
            stream.write(raw)
        runtime_sources.append({'installed_path': str(path), 'snapshot': binding(snapshot),
            'sha256': hashlib.sha256(raw).hexdigest()})
    receipt = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS_SOFTWARE_SCOPE_ONLY',
        'scope': 'SHARED_SELECTED_PLAYER_TRACK_TO_CROP_TRANSITION_AND_NATIVE_BALL_SEQUENTIAL_DECODING',
        'checks': binding(checks_path), 'junit': binding(checks_path.parent / 'junit.xml'),
        'counts': checks['counts'], 'elapsed_seconds': checks['elapsed_seconds'],
        'source_pin_count': len(checks['source_after']), 'changed_source_since_193_checks': changed,
        'source_delta': binding(patch), 'installed_ultralytics_primary_source_snapshots': runtime_sources,
        'implementation': {
            'selected_player_model_reused': True, 'additional_selected_model_instances': 0,
            'tracking_callbacks_predictor_and_feature_hook_disposed_after_complete_player_pass': True,
            'crop_device_explicitly_bound_to_admitted_actual_device': True,
            'checkpoint_crop_overrides_snapshotted_before_tracking': True,
            'failed_crop_prevents_metrics_and_result_publication': True,
            'actual_forward_counts_retained_in_metrics_and_return': True,
            'native_ball_pass_uses_fresh_sequential_decoder': True,
            'sequential_suffix_discards_decoded_prefix_without_yielding_it': True,
            'original_failure_preserved_when_decoder_cleanup_also_fails': True},
        'actual_media_check': {'scope': '17_FRAME_SYNTHETIC_64x48_25FPS_MP4V_VIDEO',
            'full_decode_and_suffix_pixels_equal_after_random_access': True,
            'real_match_or_pts_qualification': False},
        'retained_attempts': {
            'v1': 'Failed before collection: -I hides the existing user-site pygments dependency; no installation or environment mutation. Retained unchanged.',
            'v2': '108 passing checks before sequential decoder changes.',
            'v3': '126 passing checks before independent cleanup/boolean findings; superseded by v4.',
            'counts_overlap_do_not_sum': True},
        'limits': {'torch_ultralytics_imports_or_attempts': 0, 'hardware_resource_probes': 0,
            'actual_selected_model_forwards': 0, 'training': False,
            'partial_ball_inference_frontier_persisted': False, 'partial_player_tracking_restored': False,
            'within_inference_worker_resume_implemented_or_tested': False,
            'long_match_live_or_independent_accuracy_qualification': False},
        'waiting_for': 'User completion notification before next RAM/GPU admission; no probe performed in this scope',
        'builder': binding(__file__)}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({'receipt': binding(output), 'counts': receipt['counts'],
                      'source_pin_count': receipt['source_pin_count']}))


if __name__ == '__main__':
    main()
