"""Short software smoke on licensed original pixels; no product-state publication."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/validation/product_installation/pipeline_smoke_30_frames'
CLEAN_PYTHON = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=120).stdout


def probe(path):
    ffprobe = os.getenv('TENNIS_FFPROBE') or shutil.which('ffprobe')
    return json.loads(command([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                              '-show_frames', '-show_streams', '-of', 'json', str(path)]))


def strict_json(path):
    def reject_constant(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)


def prepare_excerpt():
    import cv2
    import numpy as np
    source = next(row for row in strict_json(ROOT / 'data/demos/sources.json')['sources']
                  if row['id'] == 'dimitrov_tiafoe_2024')
    source_path = ROOT / source['inputPath']
    if digest(source_path) != source['inputSha256']:
        raise ValueError('Retained licensed source hash changed')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    excerpt = OUTPUT / 'original_frames_0000_0029.mkv'
    if not excerpt.exists():
        ffmpeg = os.getenv('TENNIS_FFMPEG') or shutil.which('ffmpeg')
        command([ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-i', str(source_path),
                 '-map', '0:v:0', '-an', '-vf', 'trim=start_frame=0:end_frame=30,setpts=PTS-STARTPTS',
                 '-fps_mode', 'passthrough', '-c:v', 'libx264', '-preset', 'fast', '-qp', '0',
                 '-pix_fmt', 'yuv420p', str(excerpt)])
    original, clip = cv2.VideoCapture(str(source_path)), cv2.VideoCapture(str(excerpt))
    equality = []
    try:
        for index in range(30):
            ok_source, source_frame = original.read()
            ok_clip, clip_frame = clip.read()
            equality.append(bool(ok_source and ok_clip and np.array_equal(source_frame, clip_frame)))
        extra_frame = clip.read()[0]
    finally:
        original.release()
        clip.release()
    source_probe, clip_probe = probe(source_path), probe(excerpt)
    source_pts = [float(frame['best_effort_timestamp_time']) for frame in source_probe['frames'][:30]]
    clip_pts = [float(frame['best_effort_timestamp_time']) for frame in clip_probe['frames']]
    relative_error = max(abs((a - source_pts[0]) - (b - clip_pts[0])) for a, b in zip(source_pts, clip_pts))
    stream = clip_probe['streams'][0]
    passed = (all(equality) and not extra_frame and len(clip_pts) == 30 and relative_error == 0
              and int(stream['width']) == 1920 and int(stream['height']) == 1080
              and stream['avg_frame_rate'] == '60/1')
    report = {
        'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': passed,
        'scope': 'Licensed short software-smoke input; no accuracy or full-match qualification.',
        'sourcePath': source['inputPath'], 'sourceSha256': source['inputSha256'],
        'originalSourcePath': source['sourcePath'], 'originalSourceSha256': source['sourceSha256'],
        'sourcePage': source['sourcePage'], 'author': source['author'],
        'license': source['license'], 'licenseUrl': source['licenseUrl'],
        'derivativeLicense': source['derivativeLicense'],
        'excerptPath': excerpt.relative_to(ROOT).as_posix(), 'excerptSha256': digest(excerpt),
        'sourceFrames': [0, 30], 'originalSourceFrames': [390, 420],
        'modifications': 'First 30 decoded frames of the retained lossless rally excerpt; lossless video-only H.264, source dimensions and native 60 FPS preserved; audio omitted.',
        'decodedFramesIdentical': all(equality), 'framesCompared': len(equality),
        'frames': len(clip_pts), 'width': stream['width'], 'height': stream['height'],
        'averageFrameRate': stream['avg_frame_rate'], 'timeBase': stream['time_base'],
        'maximumRelativeTimestampDifferenceSeconds': relative_error,
        'sourcePtsSeconds': source_pts, 'excerptPtsSeconds': clip_pts,
        'qualificationEvidence': False, 'inferenceRun': False,
    }
    (OUTPUT / 'input_excerpt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not passed:
        raise ValueError('Original pixel/cadence excerpt verification failed')
    return excerpt, report


def run_smoke(excerpt, input_report):
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_OFFLINE'] = 'true'
    (OUTPUT / 'ultralytics_settings').mkdir(exist_ok=True)
    (OUTPUT / 'matplotlib_settings').mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(OUTPUT / 'ultralytics_settings')
    os.environ['MPLCONFIGDIR'] = str(OUTPUT / 'matplotlib_settings')
    attempts = []
    def blocked_network(sock, address):
        attempts.append(str(address))
        raise RuntimeError('Offline pipeline smoke attempted a network connection')
    socket.socket.connect = blocked_network
    socket.socket.connect_ex = blocked_network
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    selected = 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml'
    report = {
        'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': False,
        'scope': '30-frame, half-second software smoke only; no full-match, accuracy or public-deployment qualification.',
        'interpreter': sys.executable, 'isolatedEnvironment': sys.prefix != sys.base_prefix,
        'configuration': selected, 'configurationSha256': digest(ROOT / selected),
        'ultralyticsOfflineSetting': 'YOLO_OFFLINE=true',
        'outputDirectory': (OUTPUT / 'analysis_offline_retry').relative_to(ROOT).as_posix(),
        'inputEvidence': (OUTPUT / 'input_excerpt.json').relative_to(ROOT).as_posix(),
        'inputSha256': input_report['excerptSha256'], 'inferenceRun': False,
        'networkAttempts': attempts, 'productionQualified': False,
        'publicDeploymentPermissionsVerified': False, 'productStatePublished': False,
        'diskUsageBeforeBytes': shutil.disk_usage(ROOT)._asdict(),
    }
    started = time.perf_counter()
    try:
        import torch
        import psutil
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        from src.pipeline.phase6_pipeline import Phase6Pipeline
        model_load_started = time.perf_counter()
        pipeline = Phase6Pipeline(selected)
        report['modelLoadElapsedSeconds'] = time.perf_counter() - model_load_started
        assets = {'court': pipeline.court_detector.model,
                  'personAndRacket': pipeline.player_detector.model.model,
                  'ball': pipeline.ball_detector.model,
                  'pose': pipeline.motion_tracker.model.model}
        report['loadedAssets'] = {name: {'parameters': sum(parameter.numel() for parameter in model.parameters()),
                                      'devices': sorted({parameter.device.type for parameter in model.parameters()})}
                                  for name, model in assets.items()}
        report['cudaAvailable'] = torch.cuda.is_available()
        report['modelLoadCompleted'] = True
        report['inferenceRun'] = True
        result = pipeline.run(str(excerpt), str(OUTPUT / 'analysis_offline_retry'))
        report['pipelineResult'] = result
        report['modelDevicesAfterExecution'] = {name: sorted({parameter.device.type for parameter in model.parameters()})
                                                for name, model in assets.items()}
        analysis = OUTPUT / 'analysis_offline_retry'
        json_paths = sorted(analysis.glob('*.json'))
        for path in json_paths:
            strict_json(path)
        metrics = strict_json(analysis / 'metrics.json')
        detections = strict_json(analysis / 'detections.json')
        video = probe(analysis / 'annotated.mp4')
        report['exportedJsonFiles'] = [path.name for path in json_paths]
        report['exportedJsonCount'] = len(json_paths)
        report['metrics'] = metrics
        report['detectionFrames'] = len(detections['frames'])
        report['courtCalibration'] = strict_json(analysis / 'court_geometry.json')
        report['limitations'] = ['This excerpt has rejected court calibration and zero selected player-role coverage.',
                                 'Pose/racket models loaded and their pipeline stages completed; meaningful player pose/racket observations require accepted player roles.',
                                 'No independent labels or accuracy metrics were produced.']
        report['annotatedDecodedFrames'] = len(video['frames'])
        report['annotatedAverageFrameRate'] = video['streams'][0]['avg_frame_rate']
        report['outputArtifacts'] = {path.name: {'bytes': path.stat().st_size, 'sha256': digest(path)}
                                    for path in sorted(analysis.iterdir()) if path.is_file()}
        report['passed'] = (result['status'] == 'success' and report['isolatedEnvironment']
                            and metrics['total_frames'] == 30 and len(detections['frames']) == 30
                            and len(video['frames']) == 30 and not attempts
                            and metrics['authoritative_events_enabled'] is False
                            and metrics['production_qualified'] is False)
    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        report['elapsedSeconds'] = time.perf_counter() - started
        report['diskUsageAfterBytes'] = shutil.disk_usage(ROOT)._asdict()
        if 'psutil' in sys.modules:
            memory = sys.modules['psutil'].Process().memory_info()
            report['processMemoryBytes'] = memory._asdict()
            report['processPeakWorkingSetBytes'] = getattr(memory, 'peak_wset', None)
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
            cuda = sys.modules['torch'].cuda
            report['cudaAllocatorPeakBytes'] = {'allocated': cuda.max_memory_allocated(),
                                              'reserved': cuda.max_memory_reserved(),
                                              'scope': 'PyTorch process allocator; excludes driver/other-process memory.'}
        (OUTPUT.parent / 'real_pipeline_smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({key: report.get(key) for key in ['passed', 'interpreter', 'inferenceRun', 'networkAttempts', 'elapsedSeconds', 'detectionFrames', 'annotatedDecodedFrames', 'error']}))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', help='Verify a licensed lossless excerpt without loading models.')
    args = parser.parse_args()
    # Permit a trusted base-Python launcher while all imports and inference use
    # the already installed isolated product environment.
    if Path(sys.executable).resolve() != CLEAN_PYTHON.resolve():
        result = subprocess.run([str(CLEAN_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT)
        raise SystemExit(result.returncode)
    excerpt, input_report = prepare_excerpt()
    if args.prepare_only:
        print(json.dumps({'passed': True, 'inferenceRun': False, 'frames': 30, 'originalPixelsAndCadenceVerified': True}))
        return
    raise SystemExit(run_smoke(excerpt, input_report))


if __name__ == '__main__':
    main()
