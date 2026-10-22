"""Private, bounded three-model engineering evaluation, never web publication.

Run as ``python -m scripts.evaluate.run_yolo26_candidate --help``.
Preflight is the default; ``--execute`` explicitly runs at most 30 native frames.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
from functools import wraps
from html import escape
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import yaml

from src.detection.yolo26_bundle import REPO_ROOT, sha256_file, validate_evaluation_config


def _run(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=120,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(f'Private media preparation failed: {result.stderr[-3000:]}')
    return result.stdout


def _finite_json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON token in {path.name}: {value}')
    value = json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid)
    def inspect(node):
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError(f'Nonfinite number in {path.name}')
        if isinstance(node, dict):
            for item in node.values():
                inspect(item)
        elif isinstance(node, list):
            for item in node:
                inspect(item)
    inspect(value)
    return value


def evaluation_source(manifest, source_id, root=REPO_ROOT):
    ref = manifest.get('evaluation_sources')
    if not isinstance(ref, dict) or not isinstance(ref.get('path'), str):
        raise ValueError('Private evaluation needs an explicit pinned source manifest')
    path = (Path(root) / ref['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or sha256_file(path) != ref.get('sha256'):
        raise ValueError('Private source manifest path or SHA256 mismatch')
    sources = _finite_json(path).get('sources', [])
    selected = [item for item in sources if item.get('source_id') == source_id]
    if len(selected) != 1:
        raise ValueError('Select one declared private development source_id')
    source = selected[0]
    if (source.get('training_allowed') is not True or source.get('public_redistribution_allowed') is not False
            or not str(source.get('use_scope', '')).startswith('PRIVATE_LOCAL_')
            or source.get('development_split') not in {'train', 'val', 'development'}
            or source.get('qualification_evidence') is not False):
        raise ValueError('Source must be authorized private development footage, outside final qualification')
    video = (Path(root) / source['source_path']).resolve()
    if not video.is_relative_to(Path(root).resolve()) or not video.is_file() or sha256_file(video) != source.get('sha256'):
        raise ValueError('Private source video path or SHA256 mismatch')
    return source, video


class ResourceGuardError(ValueError):
    def __init__(self, message, resources):
        super().__init__(message)
        self.resources = resources


def resource_guard(output, device):
    import psutil
    resources = {'free_ram_bytes': psutil.virtual_memory().available,
                 'free_disk_bytes': shutil.disk_usage(output).free, 'free_vram_bytes': None,
                 'device': device}
    if device != 'cpu':
        import torch
        if not torch.cuda.is_available():
            raise ResourceGuardError('Requested GPU is unavailable; no device fallback', resources)
        resources['free_vram_bytes'] = torch.cuda.mem_get_info(int(device))[0]
        # Include runtime initialization in the available-memory guard.
        resources['free_ram_bytes'] = psutil.virtual_memory().available
    if resources['free_ram_bytes'] < 1536 * 1024 ** 2 or resources['free_disk_bytes'] < 1024 ** 3:
        raise ResourceGuardError('Private inference requires 1.5 GiB available RAM and 1 GiB free disk', resources)
    if device != 'cpu':
        if resources['free_vram_bytes'] < 2 * 1024 ** 3:
            raise ResourceGuardError('Private inference requires 2 GiB free VRAM; other processes are preserved', resources)
    return resources


def resource_snapshot(device):
    import psutil
    process = psutil.Process()
    memory = process.memory_info()
    record = {'available_ram_bytes': psutil.virtual_memory().available,
              'process_rss_bytes': memory.rss,
              'process_peak_working_set_bytes': getattr(memory, 'peak_wset', None)}
    if device != 'cpu':
        import torch
        record.update(free_vram_bytes=torch.cuda.mem_get_info(int(device))[0],
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(int(device)),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(int(device)))
    return record


def runtime_source_binding(root=REPO_ROOT):
    """Bind every repository runtime Python source without importing it."""
    root = Path(root).resolve()
    files = sorted((root / 'src').rglob('*.py'))
    if not files:
        raise ValueError('No runtime sources available for source binding')
    result = {}
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError('Runtime source binding escaped the repository')
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def evaluation_input_binding(config_path, config, manifest, video, root=REPO_ROOT):
    """Hash exact script, config, source and three checkpoints plus transitives."""
    root = Path(root).resolve()
    paths = [Path(__file__), Path(config_path), Path(video)]
    def add(value):
        path = Path(value)
        paths.append(path if path.is_absolute() else root / path)
    add(config['model_system']['artifact_manifest'])
    add(config['player_detection']['tracker'])
    add(manifest['evaluation_sources']['path'])
    for record in manifest['models'].values():
        add(record['path'])
        for key in ('training_provenance', 'origin_provenance'):
            if isinstance(record.get(key), dict):
                add(record[key]['path'])
    result = {}
    for path in sorted({path.resolve() for path in paths}):
        if not path.is_relative_to(root):
            raise ValueError('Private evaluation input escaped the repository')
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def binding_changes(before, after):
    return {'changed': sorted(key for key in before.keys() & after.keys() if before[key] != after[key]),
            'added': sorted(after.keys() - before.keys()), 'removed': sorted(before.keys() - after.keys())}


class OwnedResourceSampler:
    """Bounded aggregates for this process and the selected device; no history list."""

    def __init__(self, device, interval_seconds=.25):
        self.device, self.interval_seconds = device, interval_seconds
        self.summary = {'sample_interval_seconds': interval_seconds, 'samples': 0, 'fields': {},
                        'sampling_errors_count': 0, 'sampling_errors': [],
                        'scope': 'OWNED_PROCESS_RSS_AND_SYSTEM_AVAILABLE_RAM_SELECTED_DEVICE_FREE_VRAM',
                        'observed_peaks_are_sampling_lower_bounds': True}
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        try:
            values = resource_snapshot(self.device)
            self.summary['samples'] += 1
            for key, value in values.items():
                if value is None:
                    continue
                record = self.summary['fields'].setdefault(key, {'minimum': value, 'maximum': value, 'last': value})
                record.update(minimum=min(record['minimum'], value), maximum=max(record['maximum'], value), last=value)
        except Exception as error:
            self.summary['sampling_errors_count'] += 1
            if len(self.summary['sampling_errors']) < 8:
                self.summary['sampling_errors'].append(f'{type(error).__name__}: {error}')

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, name='private-owned-resource-sampler', daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_error):
        self._stop.set()
        self._thread.join()
        self._sample()
        return False


def native_prefix(video, output, maximum):
    """Trim before normalization and retain FFmpeg decoder-to-output PTS tokens."""
    from src.api.media_timing import frame_mapping
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise ValueError('Private evaluation requires FFmpeg and FFprobe')
    probe = json.loads(_run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                             '-show_streams', '-show_format', '-of', 'json', str(video)]))
    stream = probe['streams'][0]
    rate = Fraction(stream.get('avg_frame_rate') or stream.get('r_frame_rate'))
    if not .1 <= float(rate) <= 120 or not 0 < int(stream['width']) <= 3840 or not 0 < int(stream['height']) <= 2160:
        raise ValueError('Private source has unsupported frame rate or dimensions')
    stats, prefix = output / 'normalization_frame_stats.txt', output / 'source.mp4'
    _run([ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-copyts', '-i', str(video), '-map', '0:v:0', '-an',
          '-vf', f'trim=end_frame={maximum},fps={rate},setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2',
          '-frames:v', str(maximum), '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
          '-stats_enc_pre', str(stats), '-stats_enc_pre_fmt', '{n} {ni} {ptsi} {tbi} {pts} {tb}', str(prefix)])
    mapping = frame_mapping(stats, output / 'source_frame_mapping.jsonl', rate,
                            source_start_time=stream.get('start_time') or 0)
    if not 1 <= mapping['frames'] <= maximum or mapping['uniqueMappedSourceFrames'] != mapping['frames']:
        raise ValueError('Private prefix must contain distinct native frames without duplication')
    metadata = {'original': {'width': int(stream['width']), 'height': int(stream['height']),
                            'fps': float(rate), 'timeBase': stream.get('time_base')},
                'normalization': {'method': 'BOUNDED_NATIVE_PREFIX_THEN_FFMPEG_FPS_FILTER',
                                  'fps': float(rate), 'maximumNativeFrames': maximum},
                'timeMapping': mapping, 'full_source_decode_qualified': False}
    (output / 'media_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    return prefix, metadata


def decoded_count(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot decode private artifact {path.name}')
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None:
                raise ValueError(f'Empty decoded frame in {path.name}')
            count += 1
        return count
    finally:
        capture.release()


def instrument_models(pipeline):
    stats = {}
    for role, owner, method in [('players', pipeline.player_detector, 'detect_and_track'),
                                 ('court', pipeline.court_detector, 'predict'),
                                 ('ball', pipeline.ball_detector, 'extract_candidates')]:
        original = getattr(owner, method)
        record = stats[role] = {'calls': 0, 'elapsed_seconds': 0.}
        def wrapper(original=original, record=record):
            @wraps(original)
            def measured(*args, **kwargs):
                start = time.perf_counter()
                result = original(*args, **kwargs)
                record['calls'] += 1
                record['elapsed_seconds'] += time.perf_counter() - start
                return result
            return measured
        setattr(owner, method, wrapper())
    return stats


def report_html(receipt):
    rows = ''.join(f'<tr><td>{escape(role)}</td><td>{escape(record["sha256"])}</td>'
                   f'<td>{receipt["model_inference"][role]["calls"]}</td></tr>'
                   for role, record in receipt['models'].items())
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Private tennis pipeline evaluation</title>
<style>body{{font:16px system-ui;max-width:980px;margin:40px auto;padding:20px;color:#182b26}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:8px}}td:nth-child(2){{font:12px monospace}}video{{max-width:100%}}</style>
<h1>Private tennis pipeline evaluation</h1><p>Unqualified development footage and pilot weights. Public redistribution is not authorized.</p>
<p>{receipt['frames']} source-aligned frames decoded and rendered. This is a bounded compatibility check; it does not qualify complete matches or live latency.</p>
<video controls src="annotated.mp4"></video><p>Events, score, physical speed, player distance and measured ground positions remain unavailable. Camera activity and A/B identity remain unknown without independent review.</p>
<table><tr><th>Role</th><th>Pinned checkpoint SHA256</th><th>Inference calls</th></tr>{rows}</table>
<p>Court landmarks: finite observed slots remain image evidence; invisible slots are null. Model confidence and detection coverage are not accuracy measurements.</p>
<p><a href="evaluation_receipt.json">Run receipt</a> · <a href="camera_intervals.json">Camera intervals</a> · <a href="match_identity.json">Identity audit</a> · <a href="source_frame_mapping.jsonl">Source timing</a></p></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--device', default='cpu', choices=['cpu', '0'])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 30:
        parser.error('This engineering entrypoint is bounded to 1–30 native frames')
    output = args.output.resolve()
    if not output.is_relative_to(REPO_ROOT) or output.exists():
        raise ValueError('Choose a new private run directory inside the repository')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    manifest = validate_evaluation_config(config)
    source, video = evaluation_source(manifest, args.source_id)
    output.mkdir(parents=True)
    receipt = {'schemaVersion': 1, 'status': 'PREFLIGHT_ONLY_UNQUALIFIED', 'use_scope': 'PRIVATE_OFFLINE_EVALUATION',
               'public_redistribution_allowed': False, 'production_qualified': False, 'accuracy_qualified': False,
               'full_match_capacity_qualified': False, 'live_latency_qualified': False,
               'started_utc': datetime.now(timezone.utc).isoformat(), 'source': source,
               'config_sha256': sha256_file(args.config), 'script_sha256': sha256_file(__file__),
               'models': manifest['models'], 'ultralytics': importlib.metadata.version('ultralytics'),
               'source_manifest': manifest['evaluation_sources'],
               'runtime_sources_before': runtime_source_binding(),
               'runtime_inputs_before': evaluation_input_binding(args.config, config, manifest, video),
               'artifact_manifest': {'path': config['model_system']['artifact_manifest'],
                                     'sha256': sha256_file(REPO_ROOT / config['model_system']['artifact_manifest'])}}
    receipt_path = output / 'evaluation_receipt.json'
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    def final_bindings():
        receipt['runtime_sources_after'] = runtime_source_binding()
        receipt['runtime_inputs_after'] = evaluation_input_binding(args.config, config, manifest, video)
        receipt['runtime_source_changes'] = binding_changes(receipt['runtime_sources_before'], receipt['runtime_sources_after'])
        receipt['runtime_input_changes'] = binding_changes(receipt['runtime_inputs_before'], receipt['runtime_inputs_after'])
        receipt['runtime_source_freeze_verified'] = not any(
            receipt[key][kind] for key in ('runtime_source_changes', 'runtime_input_changes')
            for kind in ('changed', 'added', 'removed'))
    try:
        receipt['resources'] = resource_guard(output, args.device)
    except ResourceGuardError as error:
        receipt.update(status='RESOURCE_GUARD_BLOCKED_PRIVATE_UNQUALIFIED', resources=error.resources,
                       failure_type=type(error).__name__, failure_reason=str(error))
        try:
            final_bindings()
        except Exception as binding_error:
            receipt.update(runtime_source_freeze_verified=False, binding_verification_error=str(binding_error))
        save()
        raise
    save()
    if not args.execute:
        print(json.dumps({'status': receipt['status'], 'output': str(output)}))
        return receipt
    receipt['status'] = 'RUNNING_PRIVATE_UNQUALIFIED'
    save()
    try:
        sampler = OwnedResourceSampler(args.device)
        receipt['sampled_resources'] = sampler.summary
        with sampler, (output / 'processing.log').open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            receipt['resource_snapshots'] = {'before_media': resource_snapshot(args.device)}
            stages = receipt['stage_elapsed_seconds'] = {}
            start = time.perf_counter()
            prefix, media = native_prefix(video, output, args.frames)
            stages['prefix_decode_and_encode'] = time.perf_counter() - start
            config['model_system']['device'] = args.device
            config['match_configuration']['match_id'] = output.name
            # Activity/identity reviews are explicit inputs, never inferred from development labels.
            config_path = output / 'inference_config.yaml'
            config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
            start = time.perf_counter()
            from src.pipeline.phase6_pipeline import Phase6Pipeline
            stages['pipeline_module_import'] = time.perf_counter() - start
            start = time.perf_counter()
            pipeline = Phase6Pipeline(str(config_path), evaluation_only=True)
            stages['three_model_construction'] = time.perf_counter() - start
            receipt['resource_snapshots']['after_models'] = resource_snapshot(args.device)
            stats = instrument_models(pipeline)
            start = time.perf_counter()
            pipeline.run(str(prefix), str(output))
            stages['pipeline_inference_tracking_render_and_export'] = time.perf_counter() - start
            receipt['resource_snapshots']['after_pipeline'] = resource_snapshot(args.device)
            expected = media['timeMapping']['frames']
            start = time.perf_counter()
            counts = {path.name: decoded_count(path) for path in (prefix, output / 'annotated.mp4')}
            stages['complete_bounded_artifact_decode'] = time.perf_counter() - start
            if set(counts.values()) != {expected} or any(item['calls'] < expected for item in stats.values()):
                raise ValueError('Three-model inference or full bounded artifact decode did not align')
            artifacts = {}
            for path in sorted(output.glob('*.json')):
                _finite_json(path)
                if path != receipt_path:
                    artifacts[path.name] = sha256_file(path)
            receipt.update(frames=expected, decoded_artifact_frames=counts,
                           encoding={'prefix': {'codec': 'h264', 'frame_count_fully_decoded': counts[prefix.name]},
                                     'annotated': {'frame_count_fully_decoded': counts['annotated.mp4'],
                                                   'sha256': sha256_file(output / 'annotated.mp4')},
                                     'isolated_annotated_encoding_time_available': False},
                           model_inference=stats, artifact_sha256=artifacts,
                           completed_utc=datetime.now(timezone.utc).isoformat())
            (output / 'report.html').write_text(report_html(receipt), encoding='utf-8')
            final_bindings()
            if not receipt['runtime_source_freeze_verified']:
                raise ValueError('Runtime sources or pinned evaluation inputs changed during execution')
        # The resource sampler is stopped and its final sample is included.
        receipt['status'] = 'COMPLETED_PRIVATE_UNQUALIFIED'
        save()
    except BaseException as error:
        receipt.update(status='FAILED_PRIVATE_UNQUALIFIED', failure_type=type(error).__name__, failure_reason=str(error))
        if 'runtime_sources_after' not in receipt:
            try:
                final_bindings()
            except Exception as binding_error:
                receipt.update(runtime_source_freeze_verified=False, binding_verification_error=str(binding_error))
        save()
        raise
    print(json.dumps({'status': receipt['status'], 'frames': receipt['frames'], 'output': str(output)}))
    return receipt


if __name__ == '__main__':
    main()
