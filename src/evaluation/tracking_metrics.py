"""Strict frame/identity adapter for the official TrackEval metric classes."""

import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
from scipy.optimize import linear_sum_assignment


class _NumpyCompatibility:
    """Restore removed dtype aliases only inside private metric modules."""
    def __getattr__(self, name):
        if name == 'int':
            return int
        if name == 'float':
            return float
        return getattr(np, name)


def load_trackeval_metrics(source):
    source = Path(source).resolve()
    package = source / 'trackeval'
    if not (source / 'LICENSE').is_file() or not (package / 'metrics/hota.py').is_file():
        raise FileNotFoundError('Reviewed TrackEval source is required')
    suffix = hashlib.sha256(str(source).encode()).hexdigest()[:12]
    name = f'_tennis_trackeval_{suffix}'
    for package_name, path in ((name, package), (name + '.metrics', package / 'metrics')):
        if package_name not in sys.modules:
            module = types.ModuleType(package_name)
            module.__path__ = [str(path)]
            sys.modules[package_name] = module
    classes = {}
    for file, class_name in [('hota', 'HOTA'), ('clear', 'CLEAR'), ('identity', 'Identity')]:
        module_name = name + '.metrics.' + file
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, package / f'metrics/{file}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            module.np = _NumpyCompatibility()
        classes[class_name] = getattr(sys.modules[module_name], class_name)
    files = ['LICENSE', 'trackeval/_timing.py', 'trackeval/utils.py', 'trackeval/metrics/_base_metric.py',
             'trackeval/metrics/hota.py', 'trackeval/metrics/clear.py', 'trackeval/metrics/identity.py']
    provenance = {'source': str(source), 'upstream': 'https://github.com/JonathonLuiten/TrackEval',
                  'source_hashes': {file: hashlib.sha256((source / file).read_bytes()).hexdigest() for file in files},
                  'compatibility': 'Private module numpy proxy: removed np.int/np.float aliases map to int/float; source and global numpy unchanged'}
    return classes, provenance


def box_iou_matrix(first, second):
    first = np.asarray(first, dtype=float).reshape(-1, 4)
    second = np.asarray(second, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(first[:, None, 2:], second[None, :, 2:]) -
                              np.maximum(first[:, None, :2], second[None, :, :2])).prod(axis=2)
    union = (first[:, 2:] - first[:, :2]).prod(axis=1)[:, None] + \
            (second[:, 2:] - second[:, :2]).prod(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def prepare_sequence(ground_truth, predictions):
    """Frames contain dicts with integer id and continuous xyxy box; no GT filling."""
    if len(ground_truth) != len(predictions) or not ground_truth:
        raise ValueError('Require equal, nonempty complete frame sequences')
    frame_sets, identities = [], []
    for frames in (ground_truth, predictions):
        parsed, all_ids = [], set()
        for frame in frames:
            ids, boxes = [], []
            for item in frame:
                identity = item['id']
                box = np.asarray(item['box'], dtype=float)
                if not isinstance(identity, int) or isinstance(identity, bool) or identity < 0:
                    raise ValueError('Track identity must be a nonnegative integer')
                if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                    raise ValueError('Invalid finite xyxy bounding box')
                ids.append(identity)
                boxes.append(box)
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate identity within frame')
            all_ids.update(ids)
            parsed.append((ids, np.asarray(boxes, dtype=float).reshape(-1, 4)))
        mapping = {identity: index for index, identity in enumerate(sorted(all_ids))}
        frame_sets.append([(np.array([mapping[i] for i in ids], dtype=int), boxes) for ids, boxes in parsed])
        identities.append(mapping)
    gt, pred = frame_sets
    return {'num_timesteps': len(ground_truth), 'num_gt_ids': len(identities[0]),
            'num_tracker_ids': len(identities[1]), 'num_gt_dets': sum(len(ids) for ids, _ in gt),
            'num_tracker_dets': sum(len(ids) for ids, _ in pred),
            'gt_ids': [ids for ids, _ in gt], 'tracker_ids': [ids for ids, _ in pred],
            'similarity_scores': [box_iou_matrix(a, b) for (_, a), (_, b) in zip(gt, pred, strict=True)]}


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    return value


def detection_gap_diagnostics(ground_truth, predictions):
    """IoU0.5 localization gaps on explicitly annotated GT frames, not occlusion labels."""
    coverage = {}
    for frame_index, (gt, pred) in enumerate(zip(ground_truth, predictions, strict=True)):
        overlaps = box_iou_matrix([item['box'] for item in gt], [item['box'] for item in pred])
        valid = overlaps >= .5
        matched = set()
        if overlaps.size:
            # Prioritize number of valid localizations, then total overlap.
            score = valid * (min(overlaps.shape) + 1 + overlaps)
            rows, cols = linear_sum_assignment(-score)
            matched = {int(row) for row, col in zip(rows, cols) if valid[row, col]}
        for index, item in enumerate(gt):
            coverage.setdefault(item['id'], []).append((frame_index, index in matched))
    gaps = []
    for identity, observations in coverage.items():
        previous_seen, previous_frame, gap_start, gap_end, starts_after_match = False, None, None, None, False
        for frame, matched in observations:
            if previous_frame is not None and frame != previous_frame + 1:
                if gap_start is not None:
                    gaps.append({'gt_id': identity, 'start_frame': gap_start, 'end_frame': gap_end,
                                 'frames': gap_end - gap_start + 1, 'recovered_within_annotated_segment': False})
                previous_seen, gap_start = False, None
            if matched:
                if gap_start is not None:
                    gaps.append({'gt_id': identity, 'start_frame': gap_start, 'end_frame': gap_end,
                                 'frames': gap_end - gap_start + 1, 'recovered_within_annotated_segment': starts_after_match})
                    gap_start = None
                previous_seen = True
            else:
                if gap_start is None:
                    gap_start, starts_after_match = frame, previous_seen
                gap_end = frame
            previous_frame = frame
        if gap_start is not None:
            gaps.append({'gt_id': identity, 'start_frame': gap_start, 'end_frame': gap_end,
                         'frames': gap_end - gap_start + 1, 'recovered_within_annotated_segment': False})
    return {'scope': 'Maximum-cardinality frame IoU0.5 matching; gaps require explicit GT presence; not occlusion recovery or identity continuity',
            'missed_annotated_frames': sum(gap['frames'] for gap in gaps),
            'longest_gap_frames': max((gap['frames'] for gap in gaps), default=0),
            'recovered_gaps': sum(gap['recovered_within_annotated_segment'] for gap in gaps), 'gaps': gaps}


def evaluate_sequence(ground_truth, predictions, source):
    data = prepare_sequence(ground_truth, predictions)
    classes, provenance = load_trackeval_metrics(source)
    results = {}
    for name, metric_class in classes.items():
        metric = metric_class() if name == 'HOTA' else metric_class({'THRESHOLD': .5, 'PRINT_CONFIG': False})
        results[name] = serializable(metric.eval_sequence(data))
    summary = {name: float(np.mean(results['HOTA'][name])) for name in ('HOTA', 'DetA', 'AssA', 'LocA')}
    summary.update({name: results['CLEAR'][name] for name in ('MOTA', 'MOTP', 'IDSW', 'Frag', 'CLR_TP', 'CLR_FP', 'CLR_FN')})
    summary.update({name: results['Identity'][name] for name in ('IDF1', 'IDP', 'IDR')})
    return {'summary': summary, 'metrics': results, 'provenance': provenance,
            'num_frames': data['num_timesteps'], 'gt_detections': data['num_gt_dets'],
            'tracker_detections': data['num_tracker_dets'],
            'detection_gaps': detection_gap_diagnostics(ground_truth, predictions),
            'upstream_caveat': 'Pinned CLEAR implementation can undercount Frag when an entire tracker frame is empty; raw metric retained. Separate detection gap diagnostics count those missing intervals.',
            'threshold_scope': 'CLEAR/Identity IoU0.5; HOTA mean over IoU0.05:0.05:0.95',
            'identity_scope': 'Persistent IDs within this sequence; class/ignore handling belongs to the audited dataset adapter'}
