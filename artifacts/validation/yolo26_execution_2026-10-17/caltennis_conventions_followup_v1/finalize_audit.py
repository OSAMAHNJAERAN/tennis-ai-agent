"""Finalize metadata/text audit receipts without reading any media or arrays."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
batches = []
requests = []
for number in range(1, 5):
    path = BASE/f'primary_batch{number}'/'receipt.json'
    receipt = json.loads(path.read_text(encoding='utf-8'))
    batches.append({'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(path)})
    requests.extend(receipt['requests'])
inputs = ('docs/research/CALTENNIS_CAMERA_CONVENTIONS.md',
    'data/external/caltennis_metadata_v1/README.md',
    'data/external/caltennis_metadata_v1/acquisition_receipt.json',
    'data/external/caltennis_metadata_v1/publisher_metadata.json',
    'data/external/caltennis_reference_assets_v1/acquisition_receipt.json',
    'data/yolo26/caltennis_reference_reservation_v2.json',
    'artifacts/validation/yolo26_execution_2026-10-09/caltennis_prior_use_correction_receipt.json')
bindings = {name: sha(ROOT/name) for name in inputs}
assert bindings['data/yolo26/caltennis_reference_reservation_v2.json'] == '23af7b2875abe45bcdcd3829e3d13bc79ae80245d70d7837996ff9cfafed0a95'
note = ROOT/'docs/research/CALTENNIS_CAMERA_CONVENTIONS.md'
text = note.read_text(encoding='utf-8')
original = text.split('\n## Follow-up audit: 17 October 2026\n')[0].rstrip()+'\n'
assert hashlib.sha256(original.encode('utf-8')).hexdigest() == '565994bca9e324cd46267ca563fb61cdfe607a7ee3efde9b052a61c90f73116f'
good = [record for record in requests if record.get('status') == 200 and 'sha256' in record]
receipt = {'scope': 'BOUNDED_PRIMARY_CALIBRATION_AND_CLOCK_CONVENTIONS_FOLLOWUP_ONLY',
    'finished_utc': datetime.now(timezone.utc).isoformat(), 'client_date': '2026-10-17',
    'status': 'NEW_CONDITIONAL_CODE_LEAD; PUBLISHER_ARTIFACT_CONVENTIONS_STILL_UNRESOLVED',
    'note_path': str(note.relative_to(ROOT)).replace('\\', '/'),
    'note_original_prefix_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest(),
    'note_original_prefix_unchanged': True, 'note_final_sha256': sha(note),
    'local_metadata_source_bindings_sha256': bindings,
    'batch_receipts': batches, 'bounded_capture_request_count': len(requests),
    'successful_text_captures': len(good), 'failed_text_captures': len(requests)-len(good),
    'captured_text_bytes': sum(record['bytes'] for record in good), 'max_bytes_per_capture': 524288,
    'publisher_site_commit': '543c878446b68f150e967f801400fc12cbf8526d',
    'publisher_site_code_link_unassigned': True,
    'conditional_code_lead': {
        'repository': 'https://github.com/selinananana/caltech-tennis-benchmark',
        'commit': '4a24b98ead5a000a2597770e617bb31225f8f122',
        'tree_entries': 891, 'tree_truncated': False,
        'verified_publisher_or_export_linkage': False,
        'findings': [
            {'source': 'camera_pose_recovery/courtannotator.py#L24',
             'finding': 'The repository labels its court coordinates in meters: x runs along court length, y across width, and listed court points have z=0. No geographic axis signs or correspondence to the pinned HF calibrations is established.'},
            {'source': 'camera_pose_recovery/README.md#L42',
             'finding': 'The repository specifies a top-left court origin; this is not a verified origin assignment for the acquired calibration files.'},
            {'source': 'utils/video_alignment_utils.py#L96',
             'finding': 'The timestamp loader documents milliseconds and multiplies decoded PTS*time_base by 1000; the extraction script saves this relative NPY array. The exact writer of the acquired HF arrays remains unverified.'},
            {'source': 'prompthmr/visualization_scripts/visualize_time_calibrated_two_processed_world4ds_procrustes_aligned.py',
             'finding': 'Later alignment loads timestamp arrays into relative/global time fields and performs a separate offset optimization. This is not a released per-camera offset table.'},
            {'source': 'camera_pose_recovery/camera_calibration.py#L232',
             'finding': 'K is scaled for annotation-image dimensions relative to assumed 1920x1080. This does not define the anonymized published-video export transform.'}
        ]},
    'web_tool_only_observations': [
        {'url': 'https://huggingface.co/datasets/demalenk/caltennis/tree/main',
         'observation': 'The public tree displayed head 4af17dd4d398aad4116a35b5a74de4388658d356 and metadata/calibration path maintenance. Direct commit/API capture was unavailable; this displayed metadata observation did not change any local reservation.'},
        {'url': 'https://huggingface.co/datasets/demalenk/caltennis/blob/main/README.md',
         'observation': 'The public card was readable through the web tool and showed its existing README metadata. Direct pinned README capture reset the connection.'},
        {'url': 'https://arxiv.org/abs/2606.20542',
         'observation': 'The abstract page showed only v1, submitted 18 June 2026. Paper calibration/synchronization sections do not bind the acquired files to a court template, released synchronization offsets or image export transform.'}],
    'unresolved': ['PUBLISHED_CALIBRATION_WORLD_ORIGIN_AXIS_SIGNS_AND_UNITS_BINDING',
                   'PUBLISHED_TIMESTAMP_WRITER_UNITS_AND_APPLIED_INTERCAMERA_OFFSETS_BINDING',
                   '1920X1080_CALIBRATION_TO_1920X1088_PUBLISHED_IMAGE_TRANSFORM'],
    'reserved_revision': '7ab45fb0f159f9e421e32d8ebc9e52c40b256d86',
    'reserved_data_unchanged': True, 'frames_acquired': 0, 'frames_viewed': 0,
    'timestamp_arrays_read': 0, 'downloaded_source_executed': False,
    'model_inference_executed': False, 'reference_or_accuracy_qualified': False,
    'production_source_edits': False,
    'finalizer_sha256': sha(Path(__file__))}
destination = BASE/'followup_receipt.json'
with destination.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({key: receipt[key] for key in ('status', 'successful_text_captures', 'failed_text_captures',
    'captured_text_bytes', 'note_original_prefix_unchanged', 'note_final_sha256', 'reserved_data_unchanged')}))
print('receipt_sha256='+sha(destination))
