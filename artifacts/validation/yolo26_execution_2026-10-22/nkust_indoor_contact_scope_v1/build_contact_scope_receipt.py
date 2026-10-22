"""Bind the retained NKUST DEV pair review; never decode media or read CSV values."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = HERE / 'contact_scope_receipt.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def path_in_root(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    path.relative_to(ROOT)
    return path


def binding(path):
    path = path_in_root(path)
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': digest(path), 'bytes': path.stat().st_size}


def read_json(path):
    return json.loads(path_in_root(path).read_text(encoding='utf-8-sig'))


def verify_binding(record):
    current = binding(record['path'])
    if current['sha256'] != record['sha256']:
        raise ValueError('Retained evidence hash changed: ' + current['path'])
    return current


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the frozen aggregate; use a new revision.')
    manifest_path = 'data/external/nkust_landing_reference_indoor_development_v4/source_manifest.json'
    audit_path = 'artifacts/validation/yolo26_execution_2026-10-22/nkust_indoor_acquisition_v1/development_audit_receipt.json'
    native_path = 'artifacts/validation/yolo26_execution_2026-10-22/nkust_indoor_acquisition_v1/native_inspection_receipt.json'
    source = read_json(manifest_path)
    audit = read_json(audit_path)
    native = read_json(native_path)
    if source['partition'] != 'reserved_reference_development' or source['group_id'] != 'nkust-2023-03-02-indoor-session':
        raise ValueError('This receipt is restricted to the reserved indoor development session.')
    verify_binding(audit['source_manifest'])
    if path_in_root(audit['source_manifest']['path']) != path_in_root(manifest_path):
        raise ValueError('Development audit binds another source manifest.')

    inputs = [binding(manifest_path), binding(audit_path), binding(native_path)]
    for value in source['evidence'].values():
        inputs.append(verify_binding(value))
    acquisition_dir = path_in_root(audit_path).parent
    for name, expected in audit['receipt_bindings'].items():
        inputs.append(verify_binding({'path': acquisition_dir / name, 'sha256': expected}))

    local_evidence_paths = {
        'side_extraction': HERE / 'side/extraction_receipt.json',
        'side_visual_review': HERE / 'side/visual_review.json',
        'top_extraction': HERE / 'top/extraction_receipt.json',
        'top_visual_review': HERE / 'top/agent_visual_review.json',
        'primary_text_receipt': HERE / 'primary_text/receipt.json',
        'primary_text_findings': HERE / 'primary_text/FINDINGS.md',
        'primary_source_check': HERE / 'primary_text/source_check.json',
    }
    evidence = {key: binding(path) for key, path in local_evidence_paths.items()}
    inputs.extend(evidence.values())
    side = read_json(local_evidence_paths['side_visual_review'])
    side_extraction = read_json(local_evidence_paths['side_extraction'])
    top = read_json(local_evidence_paths['top_visual_review'])
    top_extraction = read_json(local_evidence_paths['top_extraction'])
    primary = read_json(local_evidence_paths['primary_text_receipt'])
    primary_check = read_json(local_evidence_paths['primary_source_check'])
    for value in primary['artifacts']:
        inputs.append(verify_binding({**value, 'path': HERE / 'primary_text' / value['path']}))
    if evidence['side_extraction']['sha256'] != side['extraction_receipt_sha256']:
        raise ValueError('Side review and extraction do not agree.')
    verify_binding(top['extraction_receipt'])

    views = []
    for media in source['media']:
        inputs.append(verify_binding(media))
        inputs.append(verify_binding(media['native_pts_evidence']))
        retained = next(value for value in native['files'] if value['filename'] == media['filename'])
        if retained['source_sha256'] != media['sha256']:
            raise ValueError('Native source hash differs from the acquisition manifest.')
        if (retained['native_frame_count'] if 'native_frame_count' in retained else retained['frame_count']) != 139:
            raise ValueError('Unexpected inspected sample frame count.')
        if retained['avg_frame_rate'] != '60000/1001' or retained['time_base'] != '1/60000':
            raise ValueError('Unexpected inspected sample native timing.')
        is_top = media['filename'] == 'IST01.mp4'
        if is_top:
            anchors = top['viewed_anchors']
            if top_extraction['source_sha256'] != media['sha256'] or top['source']['sha256'] != media['sha256']:
                raise ValueError('Top review source differs from the acquired sample.')
        else:
            anchors = [value for value in retained['exact_native_frame_anchors']
                       if value['native_frame_index'] in (104, 138)] + side_extraction['anchors']
            if side_extraction['source_sha256'] != media['sha256'] or side['source_sha256'] != media['sha256']:
                raise ValueError('Side review source differs from the acquired sample.')
        anchors = sorted(anchors, key=lambda value: value['native_frame_index'])
        if [value['native_frame_index'] for value in anchors] != [104, 110, 118, 126, 134, 138]:
            raise ValueError('Aggregate scope must stay at the six reviewed contact anchors per view.')
        verified_anchors = []
        for value in anchors:
            pin = verify_binding(value)
            inputs.append(pin)
            if value['source_time_base'] != '1/60000' or int(value['source_pts']) != value['native_frame_index'] * 1001:
                raise ValueError('Reviewed anchor does not match retained native PTS.')
            verified_anchors.append({**pin, 'native_frame_index': value['native_frame_index'],
                'source_pts': value['source_pts'], 'source_time_base': value['source_time_base'],
                'source_seconds_exact': str(Fraction(int(value['source_pts']), 60000)),
                'display_seconds_rounded': value['source_pts_time']})
        if retained['duration_seconds'] != '2.318983' or retained['last_pts'] != '138138':
            raise ValueError('Unexpected inspected sample ending.')
        views.append({'filename': media['filename'], 'source': binding(media['path']),
            'native_frames': 139, 'native_frame_rate': '60000/1001', 'source_time_base': '1/60000',
            'container_duration_seconds_exact': '2318983/1000000',
            'last_native_frame_seconds_exact': '23023/10000', 'reviewed_contact_anchors': verified_anchors})

    if top['limits']['human_review'] or top['visual_racket_contact_candidate']['exact_contact_frame_verified']:
        raise ValueError('Agent phase observations cannot become independent ground truth.')
    if side['measured_outgoing_ground_contact_established'] or side['readable_numbered_landing_marker_established']:
        raise ValueError('The retained side review requires a different scope conclusion.')
    if top['eligibility']['measured_outgoing_landing_demonstrably_visible_in_reviewed_anchors']:
        raise ValueError('The retained top review requires a different scope conclusion.')
    if primary_check['findings']['general_pre_landing_truncation_rule_found']:
        raise ValueError('Primary-text scope changed; review a new aggregate revision.')
    # CSV bytes/metadata are bound from the earlier source manifest, never parsed here.
    labels = source['landing_labels']
    start, end = Fraction(104104, 60000), Fraction(118118, 60000)
    container_end = Fraction('2.318983')
    inputs.append(binding(Path(__file__)))
    pins = {value['path']: value for value in inputs}
    for value in pins.values():
        verify_binding(value)
    receipt = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'DIRECT_LANDING_REFERENCE_NOT_ESTABLISHED_FOR_THIS_PAIR',
        'scope': 'ONE_ACQUIRED_INDOOR_DEVELOPMENT_STRAIGHT_SHOT_VIEW_PAIR_SPARSE_CONTACT_ANCHORS',
        'group_id': source['group_id'], 'partition': source['partition'],
        'dataset_doi': source['dataset_doi'], 'publisher_license': source['publisher_license'],
        'prior_immutable_source_manifest': binding(manifest_path),
        'prior_immutable_development_audit': binding(audit_path),
        'current_review_evidence': evidence, 'source_views': views,
        'reviewer_status': {'side': side['reviewer_kind'], 'top': top['reviewer']['kind'],
            'aggregator': 'AI_AGENT_NOT_HUMAN', 'independent_human_ground_truth': False},
        'coarse_visual_racket_contact_phase': {
            'native_frame_bracket_per_stream': [104, 118], 'candidate_near_frame_per_stream': 110,
            'source_pts_bracket_per_stream': ['104104', '118118'], 'source_time_base': '1/60000',
            'start_seconds_exact': str(start), 'end_seconds_exact': str(end),
            'start_seconds_rounded': '1.735067', 'end_seconds_rounded': '1.968633',
            'status': 'AGENT_VISUAL_PHASE_ONLY_NOT_INDEPENDENT_CONTACT_GROUND_TRUTH',
            'exact_contact_frame_or_timestamp_accepted': False,
            'container_remaining_after_phase_seconds': {
                'minimum_exact': str(container_end - end), 'maximum_exact': str(container_end - start),
                'minimum_rounded': '0.350350', 'maximum_rounded': '0.583916'},
            'cross_camera_offset_or_synchronization_inferred': False},
        'sample_findings': {
            'measured_outgoing_ground_contact_established_in_sampled_frames': False,
            'readable_numbered_landing_marker_established_in_sampled_frames': False,
            'observed_scope': 'Six retained contact-region anchors per view; initial coarse reviews remain separately bound.',
            'interpretation': 'The sampled anchors show strike/follow-through, and the retained clips end shortly after the coarse visual strike phase. They do not establish the measured outgoing landing for a direct observed-contact comparison.',
            'every_139_frames_individually_reviewed': False,
            'all_clip_frames_or_entire_dataset_lack_landings_claimed': False,
            'universal_pre_landing_truncation_policy_claimed': False,
            'regulation_court_dimensions_or_global_axes_assumed': False},
        'primary_text_conclusion': {
            'sources': [{'url': value['url'], 'doi': value['doi']} for value in primary_check['sources']],
            'landing_prediction_after_contact_described': primary_check['findings']['landing_prediction_after_contact_explicit'],
            'exact_clip_start_end_specification_found': primary_check['findings']['exact_clip_start_end_definition_found'],
            'general_pre_landing_truncation_rule_found': False,
            'scope': 'Reviewed author/publisher text only; no universal segmentation rule inferred from this pair.'},
        'reference_status': {
            'csv_schema_and_hash_from_prior_manifest_only': {'path': labels['path'], 'sha256': labels['sha256'],
                'rows': labels['rows'], 'columns': labels['columns'], 'unit': labels['signed_line_distance_units']},
            'exact_csv_clip_target_join_verified': False, 'joined_targets': 0, 'evaluated_targets': 0,
            'accuracy': None, 'independent_contact_ground_truth_created': False,
            'unresolved': audit['unresolved']},
        'limits': {'new_downloads': 0, 'new_frame_extractions_in_aggregation': 0,
            'additional_images_viewed_in_aggregation': 0, 'csv_target_values_read_in_aggregation': False,
            'outdoor_test_media_or_targets_inspected': False, 'model_imports_inference_or_training': False,
            'hardware_resource_probes': False, 'prior_immutable_receipts_modified': False,
            'new_reference_qualified': False, 'qualification_evidence': False},
        'verified_input_bindings': sorted(pins.values(), key=lambda value: value['path']),
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    saved = read_json(OUTPUT)
    if saved['status'] != receipt['status'] or len(saved['source_views']) != 2:
        raise ValueError('Aggregate round-trip validation failed.')
    for value in saved['verified_input_bindings']:
        verify_binding(value)
    print(json.dumps({'path': str(OUTPUT), 'sha256': digest(OUTPUT),
        'verified_input_count': len(pins), 'status': saved['status']}))


if __name__ == '__main__':
    main()
