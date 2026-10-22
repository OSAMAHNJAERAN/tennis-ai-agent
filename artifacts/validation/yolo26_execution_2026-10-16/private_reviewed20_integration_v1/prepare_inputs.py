"""Prepare new private engineering inputs; never edit a production bundle."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path.cwd()
BASE = Path(__file__).resolve().parent
PREFIX = 'configs/evaluation/yolo26.private-reviewed20-match172-2026-10-16'
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    old_prefix = ROOT / 'configs/evaluation/yolo26.private-pilot-match172'
    config = yaml.safe_load(Path(str(old_prefix) + '.yaml').read_text())
    manifest = json.loads(Path(str(old_prefix) + '.artifacts.json').read_text())
    sources = json.loads(Path(str(old_prefix) + '.sources.json').read_text())
    export_path = ROOT / 'data/yolo26/ball_reviewed_development_v1/source_manifest.json'
    export = json.loads(export_path.read_text())
    assert sha(export_path) == 'b3fa03c171c797db9df52038fe1b984cd6aef6aafb942e6885207430832a25ab'
    exposure = [value for value in export['sources'] if value['source_id'] == 'match172']
    assert [value['sample_id'] for value in exposure] == [
        'positive_0000', 'positive_0007', 'tile_review_beijing_frame15_scoreboard']
    assert all(value['derived_split'] == 'train' for value in exposure)
    provenance = next(value for value in export['derived_source_provenance'] if value['source_id'] == 'match172')
    assert provenance['derived_source_split'] == 'train'
    source = sources['sources'][0]
    source.update(match_id=None, session_id=None, actual_match_identity_verified=False,
        actual_session_identity_verified=False, development_split='train',
        current_conservative_context='beijing', current_component_id=provenance['derived_component_id'],
        training_exposed=True, final_test_used=False,
        current_source_registry={'path': str(export_path.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(export_path)},
        current_ball_training_samples=[{'sample_id': value['sample_id'], 'frame_index': value['source_frame_index'],
                                       'split': value['derived_split']} for value in exposure],
        grouping_status='TRAIN_EXPOSED_BEIJING_CONTEXT_NOT_INDEPENDENT_MATCH_OR_SESSION',
        qualification_evidence=False,
        historical_private_engineering_evaluations='Previously inspected and repeatedly evaluated; never a final test',
        separate_old_development_quarantine='match182 is St Petersburg; match172 is Beijing training, not that quarantine')
    assert sha(ROOT / source['source_path']) == source['sha256']
    source_path = ROOT / (PREFIX + '.sources.json')
    write(source_path, sources)
    manifest.update(bundle_id='private-yolo26-reviewed20-native640-tiles-match172-2026-10-16',
                    status='UNQUALIFIED_PRIVATE_REVIEWED20_DEVELOPMENT',
                    evaluation_sources={'path': source_path.relative_to(ROOT).as_posix(), 'sha256': sha(source_path)})
    for role, expected in [('ball', '90a141489dc7198789043466d7ab3c0245b86cb0135122f7b78d62115a8a9fcc'),
                           ('court', 'e47db3784adb39b82be6df7fca73ed0206f67f4f5d1bb029f0f862a34a4feefa')]:
        directory = ROOT / f'outputs/yolo26_{role}_reviewed_development_run1'
        weights, metadata_path = directory / 'training/weights/best.pt', directory / 'run_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        assert metadata['role'] == role and metadata['status'] == 'TRAINED_UNQUALIFIED'
        assert metadata['data_audit']['final_test_used'] is False and sha(weights) == expected
        assert any(Path(record['path']).resolve() == weights.resolve() and record['sha256'] == expected
                   for record in metadata['checkpoints'])
        manifest['models'][role].update(path=weights.relative_to(ROOT).as_posix(), sha256=expected,
            training_provenance={'path': metadata_path.relative_to(ROOT).as_posix(), 'sha256': sha(metadata_path)})
    manifest_path = ROOT / (PREFIX + '.artifacts.json')
    write(manifest_path, manifest)
    config['model_system'].update(artifact_manifest=manifest_path.relative_to(ROOT).as_posix(),
        device='0', status='UNQUALIFIED_PRIVATE_REVIEWED20_DEVELOPMENT')
    config['ball_detection'].update(model=manifest['models']['ball']['path'], imgsz=640)
    config['ball_detection']['tiles'].update(enabled=True, size_px=640)
    config['court_detection']['model_path'] = manifest['models']['court']['path']
    assert config['ball_detection']['low_conf'] == .05 and config['ball_detection']['high_conf'] == .25
    assert config['event_detection']['authoritative_enabled'] is False
    assert config['camera_intervals']['reviews'] == [] and config['match_identity']['anchors'] == []
    config_path = ROOT / (PREFIX + '.yaml')
    with config_path.open('x', encoding='utf-8') as stream:
        stream.write('# Private development only. Beijing TRAIN-exposed match172; never independent qualification.\n')
        yaml.safe_dump(config, stream, sort_keys=False)
    from src.detection.yolo26_bundle import validate_evaluation_config
    from scripts.evaluate.run_yolo26_candidate import evaluation_source, runtime_source_binding, evaluation_input_binding
    validated = validate_evaluation_config(config)
    selected, video = evaluation_source(validated, 'match172')
    inputs = evaluation_input_binding(config_path, config, validated, video)
    plan = dict(status='PREPARED_PRIVATE_UNQUALIFIED_NOT_EXECUTED', prepared_utc=datetime.now(timezone.utc).isoformat(),
        config=config_path.relative_to(ROOT).as_posix(), config_sha256=sha(config_path),
        manifest_sha256=sha(manifest_path), source_manifest_sha256=sha(source_path),
        argv=[sys.executable, '-u', '-m', 'scripts.evaluate.run_yolo26_candidate', '--config',
              config_path.relative_to(ROOT).as_posix(), '--source-id', 'match172', '--output',
              'outputs/private_yolo26_reviewed20_candidate_preflight_2026-10-16', '--frames', '30', '--device', '0'],
        operating_settings={'ball': {'imgsz': 640, 'low_conf': .05, 'high_conf': .25,
            'overlap': .2, 'dedup_iou': .5, 'dedup_center_px': 2., 'native_tiles_per_frame': 8,
            'full_frame_additional_per_frame': 1, 'full_frame_is_downscaled': True,
            'planned_total_ball_image_inputs': 270},
            'court': {'imgsz': 1280, 'confidence': .4, 'keypoint_confidence': .5,
                      'rect': 'PINNED_ULTRALYTICS_DEFAULT_TRUE_UNCHANGED'},
            'players': {'imgsz': 960, 'confidence': .4, 'pose_confidence': .35, 'reid': False}},
        unchanged_resource_floor={'post_cuda_available_ram_bytes': 1536*1024**2,
            'free_vram_bytes': 2*1024**3, 'free_disk_bytes': 1024**3},
        source_train_exposure=selected['current_ball_training_samples'],
        runtime_sources=runtime_source_binding(), runtime_inputs=inputs,
        qualification_evidence=False, final_test_used=False, production_promoted=False,
        public_redistribution_allowed=False, full_match_capacity_proven=False, live_operation_proven=False,
        training_performed=False, torch_imported='torch' in sys.modules, ultralytics_imported='ultralytics' in sys.modules)
    write(BASE / 'prepared_input_receipt.json', plan)
    print(json.dumps({key: plan[key] for key in ('status', 'config', 'config_sha256', 'manifest_sha256',
                                               'source_manifest_sha256', 'torch_imported', 'ultralytics_imported')}))


if __name__ == '__main__':
    main()
