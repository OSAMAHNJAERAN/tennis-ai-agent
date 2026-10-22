"""Paired isolated serialization measurements; synthetic records are not video qualification."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
TEMPLATE = ROOT / 'outputs/product/analyses/match_02d494b88e99400d8cf7243528302f9b'
LEGACY = ROOT / 'artifacts/validation/product_execution/reporting_before_streamed_csv_capacity_2026-10-05.py'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def child(args):
    import psutil
    from src.utils.json_array_artifact import write_json_array_artifact
    from src.api.reporting import telemetry_csv_chunks
    output = args.directory / f'{args.kind}_{args.mode}_{args.frames}'
    before = psutil.Process().memory_info()._asdict()
    started = time.perf_counter()
    if args.kind == 'camera':
        template = json.loads((TEMPLATE / 'camera_registration.json').read_text(encoding='utf-8'))
        seed = template.pop('frames')

        def records():
            for index in range(args.frames):
                record = copy.deepcopy(seed[index % len(seed)])
                record['frame_index'] = index
                yield record

        destination = output.with_suffix('.json')
        if args.mode == 'old':
            frames = list(records())
            with destination.open('x', encoding='utf-8') as stream:
                json.dump({**template, 'frames': frames}, stream, indent=2, allow_nan=False)
        else:
            write_json_array_artifact(destination, template, 'frames', records(), allow_nan=False)
    else:
        template = json.loads((TEMPLATE / 'result.json').read_text(encoding='utf-8'))
        seed = template['trajectories']['ballTrajectory']
        points = []
        for index in range(args.frames):
            point = copy.deepcopy(seed[index % len(seed)])
            point['frame_index'], point['timestamp_seconds'] = index, index / 60
            points.append(point)
        # Only the fields consumed by CSV are retained in both paired runs.
        result = {'summary': {'id': 'match_synthetic_capacity'}, 'mediaSource': template.get('mediaSource'),
                  'trajectories': {'ballTrajectory': points,
                                   'player1CourtPositions': [None] * args.frames, 'player1Speeds': [None] * args.frames,
                                   'player2CourtPositions': [None] * args.frames, 'player2Speeds': [None] * args.frames}}
        destination = output.with_suffix('.csv')
        if args.mode == 'old':
            spec = importlib.util.spec_from_file_location('dated_csv_baseline', LEGACY)
            legacy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy)
            text = legacy.telemetry_csv(result)
            # The former Response also retained the full encoded byte copy.
            encoded = text.encode('utf-8')
            with destination.open('xb') as stream:
                stream.write(encoded)
        else:
            with destination.open('xb') as stream:
                for chunk in telemetry_csv_chunks(result):
                    stream.write(chunk.encode('utf-8'))
    elapsed = time.perf_counter() - started
    after = psutil.Process().memory_info()._asdict()
    receipt = {'kind': args.kind, 'mode': args.mode, 'frames': args.frames,
               'elapsedSeconds': elapsed, 'memoryBeforeBytes': before, 'memoryAfterBytes': after,
               'peakWorkingSetBytes': after.get('peak_wset'), 'peakPrivateCommitBytes': after.get('peak_pagefile'),
               'outputBytes': destination.stat().st_size, 'outputSha256': digest(destination),
               'outputFile': destination.name}
    with output.with_suffix('.receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt), flush=True)


def frame_digest(path):
    """Parse the final frame array incrementally for both JSON layouts."""
    decoder = json.JSONDecoder()
    hashed, count = hashlib.sha256(), 0
    with path.open(encoding='utf-8') as stream:
        buffer = ''
        marker = '"frames": ['
        while marker not in buffer:
            block = stream.read(65536)
            if not block:
                raise ValueError('Missing frame array')
            buffer += block
        head, buffer = buffer.split(marker, 1)
        metadata = json.loads(head + '"frames": []}')
        metadata.pop('frames')
        hashed.update(json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        ended = False
        while not ended:
            buffer = buffer.lstrip(' \r\n\t,')
            if buffer.startswith(']'):
                ended = True
                continue
            try:
                record, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                block = stream.read(65536)
                if not block:
                    raise
                buffer += block
                continue
            hashed.update(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
            hashed.update(b'\n')
            count += 1
            buffer = buffer[offset:]
        remainder = (buffer[1:] + stream.read()).strip()
        if remainder != '}':
            raise ValueError('Unexpected trailing JSON')
    return {'frames': count, 'canonicalSha256': hashed.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--kind', choices=['camera', 'csv'])
    parser.add_argument('--mode', choices=['old', 'new'])
    parser.add_argument('--frames', type=int)
    args = parser.parse_args()
    args.directory = (ROOT / args.directory).resolve()
    allowed = (ROOT / 'artifacts/validation/product_execution').resolve()
    if not args.directory.is_relative_to(allowed) or args.directory == allowed:
        raise ValueError('Use a fresh isolated capacity directory')
    if args.child:
        child(args)
        return
    import psutil
    args.directory.mkdir(parents=True, exist_ok=False)
    report = {'startedAt': datetime.now(timezone.utc).isoformat(), 'scope': 'Synthetic export-only systems stress; no video inference, accuracy or full-match qualification.',
              'frames': [600, 18000, 432000], 'fpsForScaleOnly': 60,
              'limitations': ['CSV pairs include existing O(N) input arrays in both measurements.',
                              'Camera pairs isolate the removed export-only audit accumulation, without model allocations.',
                              'One process per run; Windows peak working set/commit include interpreter imports; host background activity is uncontrolled.',
                              'These paths do not bound all pipeline/result/API/frontend arrays.'],
              'sourceFiles': {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__).resolve(), LEGACY,
                  ROOT / 'src/utils/json_array_artifact.py', ROOT / 'src/api/reporting.py', TEMPLATE / 'camera_registration.json', TEMPLATE / 'result.json']},
              'runs': [], 'comparisons': [], 'passed': False}
    try:
        for kind in ('camera', 'csv'):
            for frames in report['frames']:
                for mode in ('old', 'new'):
                    stdout = args.directory / f'{kind}_{mode}_{frames}.stdout.txt'
                    with stdout.open('x', encoding='utf-8') as log:
                        process = subprocess.Popen([str(CLEAN), str(Path(__file__).resolve()), '--directory', str(args.directory),
                                                    '--child', '--kind', kind, '--mode', mode, '--frames', str(frames)],
                                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                        samples, began = [], time.perf_counter()
                        while process.poll() is None:
                            try:
                                memory = psutil.Process(process.pid).memory_info()
                                available = psutil.virtual_memory().available
                                samples.append({'seconds': time.perf_counter() - began, 'rssBytes': memory.rss,
                                                'privateBytes': getattr(memory, 'private', None), 'availableRamBytes': available})
                                if memory.rss > 1_750_000_000 or available < 400_000_000 or time.perf_counter() - began > 900:
                                    process.terminate()
                                    process.wait(timeout=30)
                                    raise RuntimeError('Owned export workload exceeded predeclared resource safeguard')
                            except psutil.NoSuchProcess:
                                pass
                            time.sleep(.25)
                        if process.returncode:
                            raise RuntimeError(f'Child failed: {kind}/{mode}/{frames}; see {stdout.name}')
                    receipt = json.loads((args.directory / f'{kind}_{mode}_{frames}.receipt.json').read_text())
                    receipt['parentSamples'] = samples
                    report['runs'].append(receipt)
                    print(json.dumps({key: receipt[key] for key in ['kind', 'mode', 'frames', 'elapsedSeconds', 'peakWorkingSetBytes', 'outputBytes']}), flush=True)
                pair = report['runs'][-2:]
                if kind == 'camera':
                    values = [frame_digest(args.directory / row['outputFile']) for row in pair]
                    identical = values[0] == values[1] and values[0]['frames'] == frames
                else:
                    values = [{'bytes': row['outputBytes'], 'sha256': row['outputSha256']} for row in pair]
                    identical = values[0] == values[1]
                report['comparisons'].append({'kind': kind, 'frames': frames, 'identicalValues': identical, 'verification': values})
                if not identical:
                    raise ValueError('Paired outputs differ')
        report['passed'] = True
    finally:
        report['completedAt'] = datetime.now(timezone.utc).isoformat()
        with (args.directory / 'report.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'passed': report['passed'], 'directory': str(args.directory)}), flush=True)


if __name__ == '__main__':
    main()
