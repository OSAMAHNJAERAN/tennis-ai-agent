"""Bounded, developer-only official YOLO11n person-mask diagnostic.

Preparation only downloads one verified official asset and pinned source files.
Inference is a separate command and never creates production participant labels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickletools
import platform
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
# Revision 01 contains a partial PNG from the recorded full-disk failure.
OUTPUT = ROOT / 'artifacts/validation/product_execution/person_mask_diagnostic02'
ASSET_FOLDER = ROOT / 'artifacts/models/person/segmentation_diagnostic01'
ASSET_API = 'https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0'
CODE_REF_API = 'https://api.github.com/repos/ultralytics/ultralytics/git/ref/tags/v8.4.36'
SOURCE_FILES = ('LICENSE', 'ultralytics/nn/tasks.py', 'ultralytics/nn/modules/block.py',
                'ultralytics/nn/modules/conv.py', 'ultralytics/nn/modules/head.py',
                'ultralytics/models/yolo/segment/predict.py',
                'ultralytics/cfg/models/11/yolo11-seg.yaml',
                'ultralytics/utils/ops.py', 'ultralytics/utils/torch_utils.py')
PROTOCOL_NOTE = ROOT / 'docs/experiments/PRODUCT_PERSON_MASK_DIAGNOSTIC_2026-10-01.md'
SOURCE_SHA = '79b556aa969ca4ee28eb1ddcfe31d4f2a83dd8bd31443f9fb6e4898323c51a21'


def bbox_iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = ((first[2] - first[0]) * (first[3] - first[1]) +
            (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / (area - intersection) if area > intersection else 0.0


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': 'tennis-mask-diagnostic'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare() -> None:
    if (ASSET_FOLDER / 'manifest.json').exists():
        raise FileExistsError('Retain the existing asset manifest; do not replace provenance')
    release = json.loads(read_url(ASSET_API))
    asset = next(item for item in release['assets'] if item['name'] == 'yolo11n-seg.pt')
    if not asset.get('digest', '').startswith('sha256:') or asset['size'] > 10_000_000:
        raise ValueError('Missing publisher digest or unexpected small-model size')
    expected = asset['digest'].split(':', 1)[1]
    reference = json.loads(read_url(CODE_REF_API))
    resolved = reference['object']
    if resolved['type'] == 'tag':
        if not resolved['url'].startswith('https://api.github.com/repos/ultralytics/ultralytics/git/tags/'):
            raise ValueError('Unexpected annotated tag source')
        resolved = json.loads(read_url(resolved['url']))['object']
    if resolved['type'] != 'commit':
        raise ValueError('Tag does not resolve directly to a commit')
    revision = resolved['sha']
    ASSET_FOLDER.mkdir(parents=True, exist_ok=True)
    checkpoint = ASSET_FOLDER / asset['name']
    if checkpoint.exists():
        if checkpoint.stat().st_size != asset['size'] or digest(checkpoint) != expected:
            raise ValueError('Existing checkpoint does not match official asset')
    else:
        temporary = checkpoint.with_suffix('.pt.partial')
        with temporary.open('xb') as handle:
            handle.write(read_url(asset['browser_download_url']))
        if temporary.stat().st_size != asset['size'] or digest(temporary) != expected:
            raise ValueError('Downloaded checkpoint does not match official size/digest')
        temporary.rename(checkpoint)
    distribution = importlib.metadata.distribution('ultralytics')
    if distribution.version != '8.4.36':
        raise ValueError('This bounded diagnostic requires the reviewed installed version')
    source_dir = ASSET_FOLDER / 'official_source'
    files = {}
    for name in SOURCE_FILES:
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/ultralytics/ultralytics/{revision}/{name}'
        target.write_bytes(read_url(url))
        installed = Path(distribution.locate_file(name))
        files[name] = {'url': url, 'sha256': digest(target),
                       'installed_path': str(installed) if installed.exists() else None,
                       'installed_sha256': digest(installed) if installed.exists() else None,
                       'installed_matches_official': digest(installed) == digest(target) if installed.exists() else None}
        if name != 'LICENSE' and files[name]['installed_matches_official'] is not True:
            raise ValueError(f'Installed source differs from pinned official source: {name}')
    with zipfile.ZipFile(checkpoint) as archive:
        pickle_name = next(name for name in archive.namelist() if name.endswith('/data.pkl'))
        data = archive.read(pickle_name)
    globals_found = sorted({arg.replace(' ', '.') for opcode, arg, _ in pickletools.genops(data)
                            if opcode.name == 'GLOBAL'})
    if any(opcode.name == 'STACK_GLOBAL' for opcode, _, _ in pickletools.genops(data)):
        raise ValueError('Dynamic global references require a separate static review')
    manifest = {'complete': True, 'qualification_evidence': False,
                'scope': 'ONE_OFFICIAL_SMALL_ASSET_DEVELOPER_ONLY_NO_MODEL_LOADING',
                'python': platform.python_version(), 'ultralytics': distribution.version,
                'metadata_url': ASSET_API, 'source_ref_url': CODE_REF_API,
                'release': {key: release[key] for key in ('id', 'tag_name', 'published_at', 'html_url')},
                'asset': {key: asset[key] for key in ('id', 'name', 'size', 'digest', 'browser_download_url', 'created_at', 'updated_at')},
                'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': expected,
                'source_revision': revision, 'source_files': files,
                'static_pickle_globals': globals_found, 'checkpoint_executed': False,
                'script_sha256': digest(Path(__file__))}
    save_json(ASSET_FOLDER / 'manifest.json', manifest)
    print(json.dumps({'asset': manifest['asset'], 'source_revision': revision,
                      'static_pickle_globals': globals_found, 'source_matches': True}, indent=2))


def prepare_onnx() -> None:
    target_manifest = ASSET_FOLDER / 'onnx_manifest.json'
    if target_manifest.exists():
        raise FileExistsError('Retain existing ONNX provenance')
    metadata_url = 'https://api.github.com/repos/ultralytics/assets/releases/tags/v8.3.0'
    release = json.loads(read_url(metadata_url))
    asset = next(item for item in release['assets'] if item['name'] == 'yolo11n-seg.onnx')
    expected = '0bc32bc92e985b881141ef9bd2216e2a746f70519d0d24da9fc85decc4428cf4'
    if asset['digest'] != 'sha256:' + expected or asset['size'] != 11_763_717:
        raise ValueError('Official ONNX export changed')
    checkpoint = ASSET_FOLDER / asset['name']
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix('.onnx.partial')
        with temporary.open('xb') as handle:
            handle.write(read_url(asset['browser_download_url']))
        if temporary.stat().st_size != asset['size'] or digest(temporary) != expected:
            raise ValueError('ONNX export fails official size/digest')
        temporary.rename(checkpoint)
    if checkpoint.stat().st_size != asset['size'] or digest(checkpoint) != expected:
        raise ValueError('Existing ONNX export differs')
    original = json.loads((ASSET_FOLDER / 'manifest.json').read_text())
    distribution = importlib.metadata.distribution('ultralytics')
    extras = {}
    for name in ('ultralytics/nn/autobackend.py', 'ultralytics/data/augment.py',
                 'ultralytics/engine/predictor.py', 'ultralytics/engine/model.py',
                 'ultralytics/utils/checks.py', 'ultralytics/utils/nms.py'):
        target = ASSET_FOLDER / 'official_source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/ultralytics/ultralytics/{original['source_revision']}/{name}"
        target.write_bytes(read_url(url))
        installed = Path(distribution.locate_file(name))
        if digest(installed) != digest(target):
            raise ValueError('Source parity failure: ' + name)
        extras[name] = {'url': url, 'sha256': digest(target), 'installed_path': str(installed)}
    dependencies = {}
    for name in ('onnxruntime', 'onnxruntime-gpu', 'onnx', 'psutil', 'coloredlogs', 'flatbuffers',
                 'numpy', 'packaging', 'protobuf', 'sympy', 'torch', 'ultralytics'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    manifest = {'complete': True, 'qualification_evidence': False, 'metadata_url': metadata_url,
                'asset': {key: asset[key] for key in ('id', 'name', 'size', 'digest', 'browser_download_url', 'created_at', 'updated_at')},
                'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': expected,
                'source_revision': original['source_revision'], 'additional_source_files': extras,
                'rejected_pickle_asset_executed': False, 'checkpoint_executed': False,
                'installed_dependency_versions': dependencies, 'script_sha256': digest(Path(__file__))}
    save_json(target_manifest, manifest)
    print(json.dumps({'asset': manifest['asset'], 'dependencies': dependencies}, indent=2))


def freeze() -> None:
    import cv2
    if OUTPUT.exists():
        raise FileExistsError('Use a new revision; existing diagnostic artifacts are immutable')
    source = ROOT / 'artifacts/demos/sources/dimitrov_tiafoe_2024_rally.mkv'
    manifest_path = ROOT / 'artifacts/validation/product_execution/active_person_original_review/manifest.json'
    labels_path = manifest_path.with_name('labels.json')
    manifest = json.loads(manifest_path.read_text())
    labels = json.loads(labels_path.read_text())
    asset_manifest = ASSET_FOLDER / 'onnx_manifest.json'
    asset = json.loads(asset_manifest.read_text())
    if digest(source) != SOURCE_SHA or manifest['sourceSha256'] != SOURCE_SHA or labels['sourceSha256'] != SOURCE_SHA:
        raise ValueError('Source identity changed')
    frames = []
    for image in manifest['images']:
        path = ROOT / image['path']
        actual = digest(path)
        if actual != image['sha256']:
            raise ValueError('Original active-person image differs from manifest: ' + str(path))
        frame = cv2.imread(str(path))
        if frame is None or frame.shape != (1080, 1920, 3):
            raise ValueError('Unexpected exact-original image shape')
        frames.append({'frame_index': image['frame_index'], 'source_frame_index': image['frame_index'] + 390,
                       'timestamp_seconds': image['frame_index'] / 60, 'path': image['path'],
                       'sha256': actual, 'review_labels_available': True})
    OUTPUT.mkdir(parents=True)
    originals = OUTPUT / 'inputs'
    originals.mkdir()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError('Cannot decode original licensed development clip')
    try:
        for index in range(313):
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Sequential decode failed before bounded adjacent sequence')
            if index in (309, 310, 311, 312):
                path = originals / f'original_frame_{index:04d}.png'
                if not cv2.imwrite(str(path), frame):
                    raise OSError('Original PNG write failed')
                frames.append({'frame_index': index, 'source_frame_index': index + 390,
                               'timestamp_seconds': index / 60, 'path': str(path.relative_to(ROOT)),
                               'sha256': digest(path), 'review_labels_available': False})
    finally:
        capture.release()
    protocol = {'schema_version': 1, 'qualification_evidence': False,
                'scientific_split': 'DIAGNOSTIC_PRODUCT_DEMO',
                'purpose': 'RAW_PERSON_MASK_OVERLAY_FEASIBILITY_NO_PRODUCTION_IDENTITIES',
                'source': str(source.relative_to(ROOT)), 'source_sha256': SOURCE_SHA,
                'source_fps': 60, 'width': 1920, 'height': 1080,
                'labels': str(labels_path.relative_to(ROOT)), 'labels_sha256': digest(labels_path),
                'label_scope': labels['labelScope'], 'negative_scope': labels['negativeScope'],
                'asset_manifest_sha256': digest(asset_manifest), 'checkpoint_sha256': asset['checkpoint_sha256'],
                'rejected_pickle_checkpoint_executed': False,
                'configuration': {'method': 'OFFICIAL_YOLO11N_SEG_ONNX', 'backend': 'OPENCV_DNN_CPU',
                                  'input_shape': [1, 3, 640, 640], 'confidence': .25,
                                  'nms_iou': .7, 'max_det': 100, 'class_id': 0,
                                  'mask_logit_threshold': 0.0, 'overlay_alpha': .22,
                                  'opencv_threads': 2, 'numpy_blas_threads': 2,
                                  'preprocessing': 'BGR_TO_RGB_FLOAT32_DIV255_CENTER_LETTERBOX_114',
                                  'postprocessing': 'PERSON_MAX_CLASS_NMS_NATIVE_LOGIT_MASK_UNPAD_BILINEAR_ORIGINAL_BOX_CROP',
                                  'interpolation': 'OPENCV_INTER_LINEAR_NOT_CLAIMED_NUMERICALLY_IDENTICAL_TO_TORCH',
                                  'identity_or_tracking_used_for_selection': False},
                'coverage_bbox_iou_threshold': .5,
                'temporal_scope': 'FOUR_ADJACENT_FRAMES_ONLY_UNLABELED_MUTUAL_BOX_MATCH_PROXY',
                'mask_pixel_ground_truth_available': False,
                'required_review': ['both_visible_bodies', 'limbs_and_feet', 'false_background_masks',
                                    'crew_spectator_masks', 'absent_near_body_at_frame300', 'adjacent_mask_area_and_shape_proxy'],
                'protocol_note_sha256': digest(PROTOCOL_NOTE), 'script_sha256': digest(Path(__file__)),
                'frames': sorted(frames, key=lambda item: item['frame_index'])}
    save_json(OUTPUT / 'protocol.json', protocol)
    print(json.dumps({'frozen_frames': len(frames), 'protocol_sha256': digest(OUTPUT / 'protocol.json'),
                      'original_hashes_verified': True, 'model_loaded': False}, indent=2))


def infer() -> None:
    # This path never imports torch, ultralytics, an ONNX converter, or pickle.
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    import cv2
    import numpy as np
    import psutil
    protocol_path = OUTPUT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    report_path = OUTPUT / 'report.json'
    if report_path.exists() or (OUTPUT / 'failure.json').exists():
        raise FileExistsError('Retain diagnostic results; do not rerun into the same revision')
    if digest(Path(__file__)) != protocol['script_sha256'] or digest(PROTOCOL_NOTE) != protocol['protocol_note_sha256']:
        raise ValueError('Frozen code/protocol changed')
    source = ROOT / protocol['source']
    if digest(source) != protocol['source_sha256']:
        raise ValueError('Original source changed')
    asset_path = ASSET_FOLDER / 'onnx_manifest.json'
    asset = json.loads(asset_path.read_text())
    checkpoint = ROOT / asset['checkpoint']
    if digest(asset_path) != protocol['asset_manifest_sha256'] or digest(checkpoint) != protocol['checkpoint_sha256']:
        raise ValueError('Official ONNX asset changed')
    for item in protocol['frames']:
        if digest(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen original frame changed')
    labels_path = ROOT / protocol['labels']
    if digest(labels_path) != protocol['labels_sha256']:
        raise ValueError('Diagnostic labels changed')
    labels = {item['frame_index']: item for item in json.loads(labels_path.read_text())['frames']}
    config = protocol['configuration']
    cv2.setNumThreads(config['opencv_threads'])
    process = psutil.Process()
    memory_before = process.memory_info().rss
    load_started = time.perf_counter()
    network = cv2.dnn.readNetFromONNX(str(checkpoint))
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    load_seconds = time.perf_counter() - load_started
    output_names = network.getUnconnectedOutLayersNames()
    if len(output_names) != 2:
        raise ValueError('Expected one raw detection output and one mask prototype output')
    def predict(frame):
        height, width = frame.shape[:2]
        gain = min(640 / height, 640 / width)
        resized_width, resized_height = round(width * gain), round(height * gain)
        pad_x, pad_y = (640 - resized_width) / 2, (640 - resized_height) / 2
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, round(pad_y - .1), round(pad_y + .1),
                                   round(pad_x - .1), round(pad_x + .1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        network.setInput(tensor)
        inference_started = time.perf_counter()
        outputs = network.forward(output_names)
        inference_seconds = time.perf_counter() - inference_started
        detection = next(value for value in outputs if value.ndim == 3)
        prototypes = next(value for value in outputs if value.ndim == 4)[0]
        if detection.shape != (1, 116, 8400) or prototypes.shape != (32, 160, 160):
            raise ValueError('Unexpected official YOLO11n segmentation output shapes')
        rows = detection[0].T
        class_scores = rows[:, 4:84]
        chosen_class = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        candidates = rows[(chosen_class == 0) & (scores >= config['confidence'])]
        confidence = scores[(chosen_class == 0) & (scores >= config['confidence'])]
        xyxy = candidates[:, :4].copy()
        xyxy[:, :2] -= candidates[:, 2:4] / 2
        xyxy[:, 2:4] += xyxy[:, :2]
        order = np.argsort(-confidence, kind='stable')
        selected = []
        while len(order) and len(selected) < config['max_det']:
            current = int(order[0])
            selected.append(current)
            order = np.array([index for index in order[1:]
                              if bbox_iou(xyxy[current], xyxy[index]) <= config['nms_iou']], dtype=np.int64)
        boxes = xyxy[selected].copy()
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / gain, 0, width)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / gain, 0, height)
        mask_logits = (candidates[selected, 84:] @ prototypes.reshape(32, -1)).reshape(-1, 160, 160)
        mask_gain = min(160 / height, 160 / width)
        mask_pad_x = (160 - round(width * mask_gain)) / 2
        mask_pad_y = (160 - round(height * mask_gain)) / 2
        left, right = round(mask_pad_x - .1), 160 - round(mask_pad_x + .1)
        top, bottom = round(mask_pad_y - .1), 160 - round(mask_pad_y + .1)
        masks, kept_boxes, kept_scores = [], [], []
        for logit, box, score in zip(mask_logits, boxes, confidence[selected], strict=True):
            original_logit = cv2.resize(logit[top:bottom, left:right], (width, height), interpolation=cv2.INTER_LINEAR)
            x1, y1, x2, y2 = np.rint(box).astype(int)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = (original_logit[y1:y2, x1:x2] > 0).astype(np.uint8)
            if mask.any():
                masks.append(mask)
                kept_boxes.append(box.tolist())
                kept_scores.append(float(score))
        return kept_boxes, kept_scores, np.array(masks, dtype=np.uint8).reshape(-1, height, width), inference_seconds
    report = {'complete': False, 'qualification_evidence': False,
              'purpose': protocol['purpose'], 'protocol_sha256': digest(protocol_path),
              'script_sha256': digest(Path(__file__)), 'checkpoint_sha256': digest(checkpoint),
              'checkpoint_format': 'PUBLISHER_ONNX_NO_PICKLE_EXECUTION',
              'rejected_pickle_checkpoint_executed': False, 'installed_packages_modified': False,
              'torch_imported': False, 'gpu_used': False, 'configuration': config,
              'runtime': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__,
                          'psutil': psutil.__version__, 'processor': platform.processor(),
                          'model_load_seconds': load_seconds, 'memory_before_load_bytes': memory_before},
              'frames': [], 'temporal_pairs': [], 'mask_pixel_accuracy_measured': False}
    save_json(report_path, report)
    first = cv2.imread(str(ROOT / protocol['frames'][0]['path']))
    warm_started = time.perf_counter()
    predict(first)
    report['runtime']['one_warmup_seconds'] = time.perf_counter() - warm_started
    transient = {}
    for item in protocol['frames']:
        index = item['frame_index']
        frame = cv2.imread(str(ROOT / item['path']))
        started = time.perf_counter()
        boxes, scores, masks, inference_seconds = predict(frame)
        total_seconds = time.perf_counter() - started
        overlay = frame.copy()
        records = []
        for proposal, (box, score, mask) in enumerate(zip(boxes, scores, masks, strict=True)):
            yy, xx = np.where(mask > 0)
            area = int(mask.sum())
            fill = area / max(1, (box[2] - box[0]) * (box[3] - box[1]))
            records.append({'proposal': proposal, 'class_id': 0, 'confidence': score, 'bbox': box,
                            'mask_area_pixels': area, 'box_fill_ratio': fill,
                            'lowest_mask_pixel_y': int(yy.max()), 'track_id': None})
            foreground = mask.astype(bool)
            overlay[foreground] = np.rint((1 - config['overlay_alpha']) * overlay[foreground] +
                                          config['overlay_alpha'] * np.array([235, 180, 80])).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (235, 180, 80), 1)
            cv2.putText(overlay, f'p{proposal}', (int(box[0]), max(12, int(box[1]) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f'DEVELOPER DIAGNOSTIC | all person masks | frame {index} | no participant selection',
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        mask_path = OUTPUT / f'masks_{index:04d}.npz'
        np.savez_compressed(mask_path, masks=masks, boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                            confidence=np.asarray(scores, dtype=np.float32))
        overlay_path = OUTPUT / f'overlay_{index:04d}.png'
        cv2.imwrite(str(overlay_path), overlay)
        row = {'frame_index': index, 'source_frame_index': item['source_frame_index'],
               'source_image_sha256': item['sha256'], 'person_masks': records,
               'inference_seconds': inference_seconds, 'preprocess_infer_mask_seconds': total_seconds,
               'rss_bytes': process.memory_info().rss, 'peak_rss_bytes': process.memory_info().peak_wset,
               'available_ram_bytes': psutil.virtual_memory().available,
               'masks': str(mask_path.relative_to(ROOT)), 'masks_sha256': digest(mask_path),
               'overlay': str(overlay_path.relative_to(ROOT)), 'overlay_sha256': digest(overlay_path)}
        if index in labels:
            truth = labels[index]
            row['label_assisted_body_coverage'] = []
            row['explicit_nonparticipant_matches'] = []
            for kind, key in (('active', 'label_assisted_body_coverage'), ('negatives', 'explicit_nonparticipant_matches')):
                for label in truth[kind]:
                    options = [(bbox_iou(label['bbox'], box), proposal) for proposal, box in enumerate(boxes)]
                    score, proposal = max(options, default=(0.0, None))
                    row[key].append({'identity_or_kind': label.get('identity', label.get('kind')),
                                     'bbox_iou': float(score), 'mask_proposal': proposal if score >= .5 else None,
                                     'scope': 'ORACLE_DIAGNOSTIC_MATCH_NOT_DEPLOYABLE_SELECTION'})
            row['ignored_active'] = truth.get('ignoredActive', [])
            row['absent_active'] = truth.get('absentActive', [])
        report['frames'].append(row)
        if index in (309, 310, 311, 312):
            transient[index] = (boxes, masks)
        save_json(report_path, report)
        print(json.dumps({'frame': index, 'person_masks': len(records), 'seconds': round(total_seconds, 3),
                          'peak_rss_mb': round(row['peak_rss_bytes'] / 1e6, 1)}), flush=True)
    for previous, current in ((309, 310), (310, 311), (311, 312)):
        before_boxes, before_masks = transient[previous]
        after_boxes, after_masks = transient[current]
        for before, box in enumerate(before_boxes):
            options = [(bbox_iou(box, next_box), after) for after, next_box in enumerate(after_boxes)]
            score, after = max(options, default=(0.0, None))
            if score < .5:
                continue
            reverse = max(((bbox_iou(after_boxes[after], old_box), i) for i, old_box in enumerate(before_boxes)),
                          default=(0.0, None))
            if reverse[1] != before:
                continue
            first_mask, second_mask = before_masks[before] > 0, after_masks[after] > 0
            intersection = int(np.logical_and(first_mask, second_mask).sum())
            union = int(np.logical_or(first_mask, second_mask).sum())
            report['temporal_pairs'].append({'from_frame': previous, 'to_frame': current,
                                             'from_proposal': before, 'to_proposal': after,
                                             'bbox_iou': float(score), 'unwarped_mask_iou': intersection / union,
                                             'area_ratio': int(second_mask.sum()) / int(first_mask.sum()),
                                             'scope': 'UNLABELED_MUTUAL_BOX_ASSOCIATION_PROXY_NOT_ID_OR_FLICKER_ACCURACY'})
    times = [item['preprocess_infer_mask_seconds'] for item in report['frames']]
    report['runtime'].update({'measured_frames': len(times), 'pipeline_seconds_total': sum(times),
                              'mean_pipeline_seconds_per_frame': float(np.mean(times)),
                              'median_pipeline_seconds_per_frame': float(np.median(times)),
                              'p95_pipeline_seconds_per_frame': float(np.percentile(times, 95)),
                              'pipeline_fps': len(times) / sum(times),
                              'mean_inference_seconds_per_frame': float(np.mean([item['inference_seconds'] for item in report['frames']])),
                              'peak_process_rss_bytes': process.memory_info().peak_wset,
                              'memory_measurement_scope': 'PROCESS_PEAK_WORKING_SET_INCLUDING_PNG_AND_MASK_ARTIFACT_CREATION',
                              'disk_write_time_excluded_from_pipeline_fps': True})
    coverage = [match for frame in report['frames'] for match in frame.get('label_assisted_body_coverage', [])]
    negatives = [match for frame in report['frames'] for match in frame.get('explicit_nonparticipant_matches', [])]
    report['summary'] = {'labeled_visible_bodies': len(coverage),
                         'body_bbox_matches_with_nonempty_masks': sum(item['mask_proposal'] is not None for item in coverage),
                         'explicit_nonparticipant_boxes': len(negatives),
                         'nonparticipant_boxes_with_mask_bbox_match': sum(item['mask_proposal'] is not None for item in negatives),
                         'unmatched_proposals_are_not_detector_false_positives': True,
                         'participant_mask_precision_measured': False, 'foot_localization_error_measured': False,
                         'mask_flicker_accuracy_measured': False, 'identity_switches_measured': False}
    report['complete'] = True
    save_json(report_path, report)
    print(json.dumps({'complete': True, 'runtime': report['runtime'], 'summary': report['summary']}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--inspect-pickle', action='store_true')
    parser.add_argument('--prepare-onnx', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--storage-status', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.storage_status:
        import shutil
        print(json.dumps({'disk_usage': shutil.disk_usage(ROOT)._asdict(),
                          'partial_diagnostic_files': [{'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size}
                                                       for path in OUTPUT.rglob('*') if path.is_file()]}, indent=2))
    elif args.prepare_onnx:
        prepare_onnx()
    elif args.freeze:
        freeze()
    elif args.infer:
        try:
            infer()
        except Exception as error:
            failure_path = OUTPUT / 'failure.json'
            if OUTPUT.exists() and not failure_path.exists():
                save_json(failure_path, {'complete': False, 'qualification_evidence': False,
                                         'error_type': type(error).__name__, 'error': str(error),
                                         'script_sha256': digest(Path(__file__)),
                                         'pickle_checkpoint_executed': False,
                                         'scope': 'PRESERVED_BOUNDED_DIAGNOSTIC_FAILURE_NO_RETRY'})
            raise
    elif args.inspect_pickle:
        manifest = json.loads((ASSET_FOLDER / 'manifest.json').read_text())
        checkpoint = ROOT / manifest['checkpoint']
        if digest(checkpoint) != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        with zipfile.ZipFile(checkpoint) as archive:
            data = archive.read(next(name for name in archive.namelist() if name.endswith('/data.pkl')))
        operations = list(pickletools.genops(data))
        for index, (opcode, argument, offset) in enumerate(operations):
            if opcode.name == 'GLOBAL' and 'getattr' in argument:
                print('GETATTR_STATIC_CONTEXT')
                for op, arg, pos in operations[max(0, index - 12):index + 24]:
                    print(pos, op.name, repr(arg))
    else:
        parser.error('Inference awaits the frozen protocol and static source/global review')


if __name__ == '__main__':
    main()
