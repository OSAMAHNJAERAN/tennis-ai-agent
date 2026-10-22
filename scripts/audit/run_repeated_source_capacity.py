"""Monitor one isolated real-worker stress job built from licensed repeated footage."""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
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
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
SELECTED = ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml'
sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def command(args, timeout=900):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        print(result.stderr, flush=True)
        raise subprocess.CalledProcessError(result.returncode, args, result.stdout, result.stderr)
    return result.stdout


def run_child(directory):
    import psutil
    begin = time.perf_counter()
    source = next(value for value in json.loads((ROOT / 'data/demos/sources.json').read_text())['sources']
                  if value['id'] == 'dimitrov_tiafoe_2024')
    original = ROOT / source['inputPath']
    if digest(original) != source['inputSha256']:
        raise ValueError('Retained source hash changed')
    report = {'startedAt': stamp(), 'passed': False, 'inferenceRun': False, 'productStatePublished': False,
              'scope': 'Five-minute repeated-source engineering stress, not representative footage or accuracy/full-match qualification.',
              'interpreter': sys.executable, 'configuration': str(SELECTED.relative_to(ROOT)),
              'configurationSha256': digest(SELECTED), 'source': source, 'repetitions': 30,
              'sourceFramesPerRepetition': 600, 'expectedFrames': 18000, 'expectedFps': '60/1',
              'repetitionBoundaryFrames': list(range(600, 18000, 600)), 'networkAttempts': [],
              'productionQualified': False, 'publicDeploymentPermissionsVerified': False}
    job = None
    try:
        ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
        prepared = directory / 'repeated_licensed_input.mp4'
        preparation = time.perf_counter()
        print(json.dumps({'at': stamp(), 'stage': 'Preparing 30 licensed repetitions at native 60 FPS'}), flush=True)
        command([ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-stream_loop', '29', '-i', str(original),
                 '-map', '0:v:0', '-an', '-frames:v', '18000',
                 '-vf', 'settb=expr=1/60,setpts=N', '-fps_mode', 'passthrough',
                 '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                 '-movflags', '+faststart', str(prepared)])
        probe = json.loads(command([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                    '-show_streams', '-show_format', '-of', 'json', str(prepared)]))
        video = probe['streams'][0]
        if not (int(video['nb_read_frames']) == 18000 and Fraction(video['avg_frame_rate']) == 60
                and video['width'] == 1920 and video['height'] == 1080 and prepared.stat().st_size <= 1024 ** 3):
            raise ValueError('Prepared workload did not meet the frozen frames/FPS/dimensions/upload ceiling')
        report['preparationElapsedSeconds'] = time.perf_counter() - preparation
        report['preparedInput'] = {'bytes': prepared.stat().st_size, 'sha256': digest(prepared), 'probe': probe,
                                  'modifications': 'Thirty consecutive repetitions of the licensed source frames 390–989; video-only H.264 CRF 18 compression and source dimensions preserved. A synthetic uniform 60 FPS clock assigns timestamp N/60; original quantized source PTS are not claimed. Artificial discontinuities occur every 600 frames.'}
        from src.api.jobs import JobStore, Settings
        from src.api.worker import process_job
        from src.api.reporting import telemetry_csv_chunks
        storage = directory / 'isolated_product'
        store = JobStore(storage)
        settings = Settings(storage, SELECTED)
        job = store.create(store.new_id(), 'repeated_licensed_stress.mp4', '.mp4', prepared.stat().st_size, SELECTED)
        upload = store.upload_dir(job['id'])
        upload.mkdir(parents=True, exist_ok=False)
        destination = upload / 'original.mp4'
        if not destination.resolve().is_relative_to(directory):
            raise ValueError('Input move target escaped owned storage')
        prepared.replace(destination)
        output = store.output_dir(job['id'])
        output.mkdir(parents=True, exist_ok=False)
        credit = {key: source[key] for key in ('id', 'title', 'author', 'sourcePage', 'license', 'licenseUrl',
                                             'derivativeLicense', 'sourceSha256')}
        credit.update({'title': source['title'] + ' · repeated systems stress',
                       'inputSha256': report['preparedInput']['sha256'],
                       'modifications': report['preparedInput']['modifications'] + ' Analysis adds evidence overlays.'})
        save(output / 'media_source.json', credit)
        report['analysisId'] = job['id']
        save(directory / 'accepted_job.json', {'analysisId': job['id'], 'storageRoot': str(storage),
                                               'startedAt': stamp(), 'productStatePublished': False})
        job = store.claim()
        os.environ.update({'YOLO_AUTOINSTALL': 'false', 'YOLO_OFFLINE': 'true',
                           'YOLO_CONFIG_DIR': str(directory / 'ultralytics_settings'),
                           'MPLCONFIGDIR': str(directory / 'matplotlib_settings')})
        for name in ('ultralytics_settings', 'matplotlib_settings'):
            (directory / name).mkdir(exist_ok=True)

        def deny_network(_socket, address):
            report['networkAttempts'].append(str(address))
            raise RuntimeError('Offline stress job attempted a network connection')
        socket.socket.connect = deny_network
        socket.socket.connect_ex = deny_network
        import torch
        report['cudaAvailable'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        worker_begin = time.perf_counter()
        report['inferenceRun'] = True
        print(json.dumps({'at': stamp(), 'stage': 'Executing actual isolated process_job', 'analysisId': job['id']}), flush=True)
        process_job(store, job, settings)
        report['workerElapsedSeconds'] = time.perf_counter() - worker_begin
        report['job'] = store.get(job['id'])
        if report['job']['status'] != 'COMPLETED':
            raise RuntimeError('Actual worker did not publish a completed result: ' + str(report['job'].get('error')))
        result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
        if result['summary']['frames'] != 18000:
            raise ValueError('Published frame count differs')
        report['summary'] = result['summary']
        report['pipelineMetrics'] = json.loads((output / 'metrics.json').read_text())
        report['mapping'] = json.loads((output / 'media_metadata.json').read_text())['timeMapping']
        csv_path = directory / 'published_telemetry.csv'
        with csv_path.open('x', encoding='utf-8', newline='') as stream:
            for chunk in telemetry_csv_chunks(result):
                stream.write(chunk)
        count, last = 0, None
        with csv_path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['analysis_id'] != job['id'] or int(row['frame']) != count:
                    raise ValueError('CSV analysis identity/frame order mismatch')
                count += 1
                last = row
        report['csvValidation'] = {'rows': count, 'lastFrame': int(last['frame']), 'bytes': csv_path.stat().st_size,
                                   'sha256': digest(csv_path)}
        report['outputArtifacts'] = {path.name: {'bytes': path.stat().st_size, 'sha256': digest(path)}
                                     for path in sorted(output.iterdir()) if path.is_file()}
        report['sourceUnchanged'] = digest(original) == source['inputSha256']
        report['limitations'] = ['Repeated ten-second input cannot represent a full match or independent sessions.',
                                 'Artificial repetition boundaries are not physical events.',
                                 'Rejected court calibration may withhold selected-player pose/racket/ground observations.',
                                 'This run cannot qualify accuracy, physical ball speed, reconstructed 3D or public deployment rights.',
                                 'Pipeline timing excludes final export/publication; workerElapsedSeconds includes preparation through full decode checks and result publication.']
        report['passed'] = count == 18000 and report['mapping'].get('frames') == 18000 and report['sourceUnchanged'] and not report['networkAttempts']
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        if isinstance(error, subprocess.CalledProcessError):
            report['commandStderr'] = error.stderr
        raise
    finally:
        report['completedAt'] = stamp()
        report['totalChildElapsedSeconds'] = time.perf_counter() - begin
        report['processMemoryBytes'] = psutil.Process().memory_info()._asdict()
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            report['cudaAllocatorPeakBytes'] = {'allocated': torch.cuda.max_memory_allocated(),
                                               'reserved': torch.cuda.max_memory_reserved(),
                                               'scope': 'PyTorch process allocator; excludes driver and other processes.'}
        save(directory / 'worker_receipt.json', report)
        print(json.dumps({key: report.get(key) for key in ('passed', 'analysisId', 'workerElapsedSeconds', 'error')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    directory = (ROOT / args.directory).resolve()
    allowed = (ROOT / 'artifacts/validation/product_execution').resolve()
    if not directory.is_relative_to(allowed) or directory == allowed:
        raise ValueError('Use a fresh isolated workspace capacity directory')
    if args.child:
        run_child(directory)
        return
    import psutil
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {'declaredAt': stamp(), 'scriptSha256': digest(Path(__file__).resolve()),
                'configurationSha256': digest(SELECTED), 'scope': 'Isolated repeated-source engineering stress only.',
                'sourceFiles': {relative: digest(ROOT / relative) for relative in (
                    'src/api/worker.py', 'src/api/results.py', 'src/api/media_timing.py',
                    'src/api/reporting.py', 'src/pipeline/phase6_pipeline.py',
                    'src/utils/json_array_artifact.py', 'src/utils/video_frame_sequence.py',
                    'data/demos/sources.json')},
                'frames': 18000, 'durationSeconds': 300, 'nativeFps': 60, 'width': 1920, 'height': 1080,
                'timestampPolicy': 'SYNTHETIC_UNIFORM_60_FPS_N_OVER_60; NOT_ORIGINAL_SOURCE_PTS',
                'repetitions': 30, 'sampleIntervalSeconds': 2, 'maximumWallSeconds': 10800,
                'maximumSampledOwnedTreeRssBytes': 4_000_000_000,
                'minimumAvailableRamBytes': 400_000_000, 'lowRamConsecutiveSamplesToStop': 5,
                'hostRamBeforeBytes': psutil.virtual_memory()._asdict(), 'diskBeforeBytes': shutil.disk_usage(ROOT)._asdict(),
                'productStatePublished': False, 'qualificationEvidence': False,
                'retention': 'All owned stress inputs, intermediate videos, model evidence, logs and receipts retained; no production cleanup.'}
    save(directory / 'protocol.json', protocol)
    begin = time.perf_counter()
    report = {'startedAt': stamp(), 'passed': False, 'samples': 0, 'peakSampledOwnedTreeRssBytes': 0,
              'peakSampledOwnedTreePrivateBytes': 0, 'minimumSampledAvailableRamBytes': psutil.virtual_memory().available,
              'scope': 'Sampled owned process tree; shared RSS pages may be counted twice and short peaks may be missed.'}
    with (directory / 'child_stdout.txt').open('x', encoding='utf-8') as log, (directory / 'resource_samples.jsonl').open('x', encoding='utf-8') as samples:
        process = subprocess.Popen([str(CLEAN), str(Path(__file__).resolve()), '--directory', str(directory), '--child'],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        save(directory / 'running_process.json', {'pid': process.pid, 'startedAt': stamp(), 'command': str(Path(__file__).resolve())})
        low_ram = 0
        last_stage = None
        isolated_store = None
        try:
            while process.poll() is None:
                owned = []
                try:
                    root_process = psutil.Process(process.pid)
                    for item in [root_process, *root_process.children(recursive=True)]:
                        try:
                            memory = item.memory_info()
                            owned.append({'pid': item.pid, 'rssBytes': memory.rss, 'privateBytes': getattr(memory, 'private', memory.vms)})
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                available = psutil.virtual_memory().available
                sample = {'at': stamp(), 'elapsedSeconds': time.perf_counter() - begin, 'processes': owned,
                          'availableRamBytes': available, 'freeDiskBytes': shutil.disk_usage(ROOT).free,
                          'ownedTreeRssBytes': sum(row['rssBytes'] for row in owned),
                          'ownedTreePrivateBytes': sum(row['privateBytes'] for row in owned)}
                accepted = directory / 'accepted_job.json'
                if accepted.is_file():
                    from src.api.jobs import JobStore
                    identity = json.loads(accepted.read_text())['analysisId']
                    if isolated_store is None:
                        isolated_store = JobStore(directory / 'isolated_product')
                    job = isolated_store.get(identity)
                    sample.update({'analysisId': identity, 'status': job['status'], 'stage': job['stage'], 'step': job['step']})
                samples.write(json.dumps(sample) + '\n')
                samples.flush()
                report['samples'] += 1
                report['peakSampledOwnedTreeRssBytes'] = max(report['peakSampledOwnedTreeRssBytes'], sample['ownedTreeRssBytes'])
                report['peakSampledOwnedTreePrivateBytes'] = max(report['peakSampledOwnedTreePrivateBytes'], sample['ownedTreePrivateBytes'])
                report['minimumSampledAvailableRamBytes'] = min(report['minimumSampledAvailableRamBytes'], available)
                stage = sample.get('stage', 'Preparing engineering input')
                if stage != last_stage:
                    print(json.dumps({key: sample.get(key) for key in ('at', 'elapsedSeconds', 'analysisId', 'status', 'stage', 'ownedTreeRssBytes', 'availableRamBytes')}), flush=True)
                    last_stage = stage
                low_ram = low_ram + 1 if available < protocol['minimumAvailableRamBytes'] else 0
                if (sample['ownedTreeRssBytes'] > protocol['maximumSampledOwnedTreeRssBytes']
                        or low_ram >= protocol['lowRamConsecutiveSamplesToStop']
                        or sample['elapsedSeconds'] > protocol['maximumWallSeconds']):
                    report['safeguardStop'] = 'Predeclared owned workload resource threshold reached'
                    raise RuntimeError(report['safeguardStop'])
                time.sleep(2)
            report['returncode'] = process.returncode
            report['passed'] = process.returncode == 0 and json.loads((directory / 'worker_receipt.json').read_text())['passed']
        except BaseException:
            if process.poll() is None:
                try:
                    owner = psutil.Process(process.pid)
                    for descendant in reversed(owner.children(recursive=True)):
                        try:
                            descendant.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    process.terminate()
                    process.wait(timeout=30)
                except psutil.NoSuchProcess:
                    pass
            if isolated_store is not None:
                interrupted = isolated_store.get(identity)
                if interrupted['status'] in {'UPLOADED', 'PROCESSING'}:
                    isolated_store.finish(identity, 'Isolated engineering stress was interrupted by its monitoring safeguard; partial artifacts retained.')
            raise
        finally:
            report['completedAt'] = stamp()
            report['totalMonitoredWallSeconds'] = time.perf_counter() - begin
            report['diskAfterBytes'] = shutil.disk_usage(ROOT)._asdict()
            report['ownedFilesBytes'] = sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())
            save(directory / 'resource_receipt.json', report)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
