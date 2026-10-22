"""Exclusive durable worker: normalize video, call Phase6Pipeline, publish output."""

from contextlib import redirect_stdout, redirect_stderr
from fractions import Fraction
import json
import gc
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

from filelock import FileLock, Timeout
import yaml

from src.api.jobs import JobStore, REPO_ROOT, Settings
from src.api.results import build_result
from src.api.result_storage import ResultStorage
from src.api.result_document import publish_result
from src.api.media_timing import frame_mapping


class AnalysisError(Exception):
    """A user-actionable error whose text can be returned by the API."""


def binary(name):
    path = os.getenv('TENNIS_' + name.upper()) or shutil.which(name)
    if not path:
        raise AnalysisError(f'{name} is missing on the analysis server. Install FFmpeg and restart the server.')
    return path


def execute(args, timeout=7200):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        print(result.stderr[-6000:])
        raise AnalysisError('The video could not be decoded or encoded. Check disk space, export a playable MP4 and try again.')
    return result.stdout


def check_models(config_path):
    if not config_path.is_file():
        raise AnalysisError('The inference configuration is missing on the server.')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    paths = [config['court_detection']['model_path'], config['player_detection']['model'], config['ball_detection']['model']]
    if config.get('pose_estimation', {}).get('enabled', True):
        paths.append(config.get('pose_estimation', {}).get('model_path', 'yolo11n-pose.pt'))
    if config['ball_detection'].get('backend') == 'wasb':
        paths.append(config['ball_detection']['source_directory'])
        if config['ball_detection'].get('ensemble_checkpoint'):
            paths.append(config['ball_detection']['ensemble_checkpoint'])
    if any(not (REPO_ROOT / value).exists() for value in paths):
        raise AnalysisError('Required model weights or inference source files are missing. Restore the configured assets on the server and retry.')
    return config


class StageLog:
    """Consume the pipeline's existing stage messages, without changing inference."""
    def __init__(self, stream, store, analysis_id):
        self.stream, self.store, self.analysis_id = stream, store, analysis_id
        self.pending = ''

    def write(self, text):
        self.stream.write(text)
        self.stream.flush()
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            match = re.search(r'\[Step (\d+)/11\]\s*(.*)', line)
            if match:
                self.store.stage(self.analysis_id, match[2].strip('.'), int(match[1]))
        if len(self.pending) > 8192:
            self.pending = self.pending[-4096:]
        return len(text)

    def flush(self):
        self.stream.flush()


def process_job(store, job, settings):
    output = store.output_dir(job['id'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'processing.log').open('a', encoding='utf-8') as log:
        stream = StageLog(log, store, job['id'])
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                config = check_models(Path(job['config_path']))
                original = store.upload_dir(job['id']) / ('original' + job['extension'])
                probe = json.loads(execute([binary('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
                                           '-show_streams', '-show_format', '-of', 'json', str(original)], timeout=60))
                video = probe['streams'][0]
                frame_rate = Fraction(video.get('avg_frame_rate') or video.get('r_frame_rate'))
                fps = float(frame_rate)
                duration = float(video.get('duration') or probe['format']['duration'])
                if not (.1 <= fps <= 120 and 0 < duration <= settings.max_duration_seconds):
                    raise AnalysisError(f'Use a video with a valid frame rate (up to 120 FPS) and duration up to {settings.max_duration_seconds // 60} minutes.')
                if not (0 < int(video['width']) <= 3840 and 0 < int(video['height']) <= 2160):
                    raise AnalysisError('Use a video no larger than 3840 × 2160 pixels.')
                media_metadata = {
                    'original': {'width': int(video['width']), 'height': int(video['height']),
                                 'fps': fps, 'durationSeconds': duration, 'timeBase': video.get('time_base'),
                                 'averageFrameRate': video.get('avg_frame_rate'),
                                 'nominalFrameRate': video.get('r_frame_rate'),
                                 'startTimeSeconds': video.get('start_time'), 'codec': video.get('codec_name')},
                    'normalization': {'method': 'FFMPEG_FPS_FILTER_CONSTANT_FRAME_RATE',
                                      'fps': fps, 'scaleX': 1, 'scaleY': 1,
                                      'paddingRightPx': int(video['width']) % 2,
                                      'paddingBottomPx': int(video['height']) % 2,
                                      'audio': 'OMITTED_FROM_ANALYSIS_PLAYBACK_ORIGINAL_RETAINED'},
                    'timeMapping': {'frameMapping': None, 'status': 'EXACT_SOURCE_FRAME_MAPPING_NOT_RECORDED',
                                    'semantics': 'Inference timestamps use normalized presentation time. '
                                                 'Duplicated/dropped source-frame mapping requires further qualification.'}}
                (output / 'media_metadata.json').write_text(json.dumps(media_metadata, allow_nan=False), encoding='utf-8')
                store.stage(job['id'], 'Preparing video for synchronized playback')
                source = output / 'source.mp4'
                frame_stats = output / 'normalization_frame_stats.txt'
                execute([binary('ffmpeg'), '-nostdin', '-y', '-v', 'error', '-xerror', '-copyts', '-i', str(original),
                         '-map', '0:v:0', '-an', '-vf', f'fps={frame_rate},setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2',
                         '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                         '-stats_enc_pre', str(frame_stats), '-stats_enc_pre_fmt', '{n} {ni} {ptsi} {tbi} {pts} {tb}',
                         '-movflags', '+faststart', str(source)])
                declared_frames = video.get('nb_frames')
                declared_frames = int(declared_frames) if isinstance(declared_frames, str) and declared_frames.isdigit() else None
                try:
                    media_metadata['timeMapping'] = frame_mapping(
                        frame_stats, output / 'source_frame_mapping.jsonl', frame_rate,
                        declared_frames, video.get('start_time') or 0)
                except (ValueError, OSError) as exc:
                    print(f'Source timing provenance unavailable: {exc}')
                    media_metadata['timeMapping'] = {'status': 'FRAME_MAPPING_UNAVAILABLE', 'frameMapping': None,
                                                    'validMeasurementTransitionIntervals': [],
                                                    'semantics': 'Original timing could not be matched; physical movement timing is withheld.'}
                media_metadata['normalization'].update({'fpsRational': str(frame_rate), 'timestampPolicy': 'COPY_SOURCE_PTS_THEN_REBASE_NORMALIZED_TO_ZERO'})
                (output / 'media_metadata.json').write_text(json.dumps(media_metadata, allow_nan=False), encoding='utf-8')
                store.stage(job['id'], 'Loading the existing tennis models')
                # Use a job-specific copy solely for provenance and match identity.
                config.setdefault('match_configuration', {})['match_id'] = job['id']
                job_config = output / 'inference_config.yaml'
                job_config.write_text(yaml.safe_dump(config), encoding='utf-8')
                from src.pipeline.phase6_pipeline import Phase6Pipeline
                Phase6Pipeline(str(job_config)).run(str(source), str(output))
                store.stage(job['id'], 'Preparing annotated playback and validating results', 11)
                execute([binary('ffmpeg'), '-nostdin', '-y', '-v', 'error', '-xerror', '-i', str(output / 'annotated.mp4'),
                         '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                         '-movflags', '+faststart', str(output / 'annotated_web.mp4')])
                generations = output / '.result'
                generations.mkdir(exist_ok=True)
                with ResultStorage(generations) as storage:
                    result = build_result(output, job, _storage=storage)
                    mapped_count = media_metadata['timeMapping'].get('frames')
                    if mapped_count is not None and mapped_count != result['summary']['frames']:
                        raise AnalysisError('Source timing and normalized analysis frames did not align. The incomplete result was not published.')
                    # Full decode verifies both browser videos and frame alignment.
                    for path in (source, output / 'annotated_web.mp4'):
                        decoded = json.loads(execute([binary('ffprobe'), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                                     '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)]))
                        if int(decoded['streams'][0]['nb_read_frames']) != result['summary']['frames']:
                            raise AnalysisError('Video and analysis frames did not align. The incomplete result was not published.')
                    publish_result(storage, result, output)
                store.finish(job['id'])
            except Exception as exc:
                traceback.print_exc()
                if isinstance(exc, AnalysisError):
                    message = str(exc)
                elif isinstance(exc, MemoryError) or 'out of memory' in str(exc).lower():
                    message = 'Insufficient processing memory. Try a shorter or lower-resolution video.'
                elif isinstance(exc, (ValueError, KeyError, IndexError, ZeroDivisionError)):
                    message = 'Video or analysis data was invalid. Export a playable MP4 and retry; see the server processing log for details.'
                elif isinstance(exc, subprocess.TimeoutExpired):
                    message = 'Video preparation exceeded its time limit. Try a shorter video.'
                else:
                    message = 'The analysis pipeline failed. Check the server processing log and configured model assets, then retry.'
                store.finish(job['id'], message)
            finally:
                gc.collect()
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()


def main():
    settings = Settings.from_env()
    store = JobStore(settings.root)
    os.chdir(REPO_ROOT)
    try:
        with FileLock(str(settings.root / 'worker.lock'), timeout=0):
            store.recover_interrupted()
            while True:
                job = store.claim()
                if job:
                    process_job(store, job, settings)
                else:
                    time.sleep(1)
    except Timeout:
        # Another process still holds the lock and owns the queue.
        return


if __name__ == '__main__':
    main()
