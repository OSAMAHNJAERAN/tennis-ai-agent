"""Exclusive durable worker: normalize video, call Phase6Pipeline, publish output."""

from contextlib import redirect_stdout, redirect_stderr
from fractions import Fraction
import json
import gc
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

from filelock import FileLock, Timeout
import yaml

from src.api.jobs import JobStore, REPO_ROOT, Settings, JobCancelled
from src.api.results import build_result
from src.api.result_storage import ResultStorage
from src.api.result_document import publish_result
from src.api.media_timing import frame_mapping
from src.api.analysis_resume import resume_identity, REQUIRED_PIPELINE_FILES, OPTIONAL_PIPELINE_FILES


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
    if config.get('model_system', {}).get('family') == 'yolo26':
        from src.detection.yolo26_bundle import validate_yolo26_config
        try:
            validate_yolo26_config(config, check_assets=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f'YOLO26 artifact validation failed: {exc}')
            raise AnalysisError('Analysis is unavailable because the selected model assets or required verification are incomplete. The server administrator must restore and verify the configured assets before retrying.') from exc
        return config
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
    def __init__(self, stream, store, analysis_id, *, storage_check=None):
        self.stream, self.store, self.analysis_id = stream, store, analysis_id
        self.storage_check = storage_check
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
                if self.storage_check is not None:
                    self.storage_check()
        if len(self.pending) > 8192:
            self.pending = self.pending[-4096:]
        return len(text)

    def flush(self):
        self.stream.flush()


def _checkpoint_identity(job, config, original):
    return resume_identity(job, config, original, root=REPO_ROOT,
                           binaries={name: binary(name) for name in ('ffmpeg', 'ffprobe')})


def _load_completed_checkpoint(output, identity):
    from src.api.pipeline_checkpoint import load_pipeline_checkpoint
    try:
        return load_pipeline_checkpoint(output, identity, required_files=REQUIRED_PIPELINE_FILES,
                                        optional_files=OPTIONAL_PIPELINE_FILES)
    except (ValueError, OSError) as error:
        raise AnalysisError('Saved processing files no longer match this analysis. Retry from the retained original video to create a new analysis.') from error


def _partial_inference_enabled(config):
    """Partial frames have a contract only for the selected native ball pass."""
    return (config.get('model_system', {}).get('family') == 'yolo26'
            and config.get('ball_detection', {}).get('backend') == 'yolo26')


def _load_partial_inference(output, identity, config):
    if not _partial_inference_enabled(config):
        return None
    from src.utils.native_inference_session import validate_partial_inference
    try:
        partial = validate_partial_inference(output, identity, config['ball_detection'])
        if partial is None and (output / '.native_inference_prefix.json').exists():
            raise ValueError('A retained player prefix has no valid normalized-source seal')
        return partial
    except (ValueError, OSError, sqlite3.DatabaseError) as error:
        raise AnalysisError('Saved inference files no longer match this analysis. Retry from the retained original video to create a new analysis.') from error


def recoverable_completed_job(store, job):
    """Admit a sealed stage or player-prefix/raw-ball frontier under the lock.

    Partial validation may perform SQLite hot-journal rollback. It never runs
    models, normalizes media or advances the accepted inference frontier.
    """
    output = store.output_dir(job['id'])
    if not any((output / name).exists() for name in
               ('.pipeline_checkpoint.json', '.native_inference_prefix.json')):
        return False
    try:
        config = check_models(Path(job['config_path']))
        if config.get('model_system', {}).get('family') != 'yolo26':
            return False
        config.setdefault('match_configuration', {})['match_id'] = job['id']
        original = store.upload_dir(job['id']) / ('original' + job['extension'])
        identity = _checkpoint_identity(job, config, original)
        if (output / '.pipeline_checkpoint.json').exists():
            # A corrupt completed marker never falls back to an older prefix.
            return _load_completed_checkpoint(output, identity) is not None
        return _load_partial_inference(output, identity, config) is not None
    except Exception as error:
        with (output / 'processing.log').open('a', encoding='utf-8') as stream:
            stream.write(f'Saved inference checkpoint cannot resume: {type(error).__name__}: {error}\n')
            traceback.print_exception(error, file=stream)
        return False


def _saved_mapped_frames(output):
    # Timing metadata can contain many transition intervals. Parse through the
    # existing bounded reader and detach only the scalar needed by finalization.
    with ResultStorage(output) as storage:
        metadata = storage.read_json(output / 'media_metadata.json')
        count = metadata['timeMapping'].get('frames')
        if count is not None and (type(count) is not int or count <= 0):
            raise ValueError('Saved normalized frame count is invalid')
        return count


def process_job(store, job, settings):
    from fastapi import HTTPException
    from src.api.resource_quotas import ResourceQuotas
    quotas = ResourceQuotas(store, settings)

    def check_storage():
        try:
            with store.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                quotas.check_storage(conn, job['owner_id'])
        except HTTPException as error:
            raise AnalysisError(error.detail) from error

    output = store.output_dir(job['id'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'processing.log').open('a', encoding='utf-8') as log:
        stream = StageLog(log, store, job['id'], storage_check=check_storage)
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                store.stage(job['id'], 'Validating video')
                check_storage()
                config = check_models(Path(job['config_path']))
                original = store.upload_dir(job['id']) / ('original' + job['extension'])
                config.setdefault('match_configuration', {})['match_id'] = job['id']
                checkpoint_enabled = config.get('model_system', {}).get('family') == 'yolo26'
                identity = _checkpoint_identity(job, config, original) if checkpoint_enabled else None
                checkpoint = _load_completed_checkpoint(output, identity) if checkpoint_enabled else None
                native_source_seal_required = False
                if _partial_inference_enabled(config):
                    if checkpoint is None:
                        # Every fresh/partial native execution must retain its
                        # normalized-input seal through the publication commit.
                        native_source_seal_required = True
                    else:
                        from src.utils.native_inference_session import SOURCE_SEAL_NAME
                        try:
                            (output / SOURCE_SEAL_NAME).lstat()
                        except FileNotFoundError:
                            # Older completed-only checkpoints predate this
                            # seal; their existing complete-stage contract stays.
                            pass
                        else:
                            # lstat also sees a broken symlink, which must fail
                            # verification rather than masquerade as old state.
                            native_source_seal_required = True
                source = output / 'source.mp4'
                if checkpoint is not None:
                    store.stage(job['id'], 'Resuming saved analysis finalization', 11)
                    mapped_count = _saved_mapped_frames(output)
                else:
                    partial = _load_partial_inference(output, identity, config) if checkpoint_enabled else None
                    if partial is not None:
                        store.stage(job['id'], 'Resuming saved ball inference')
                        mapped_count = partial['mapped_count']
                        from src.pipeline.phase6_pipeline import Phase6Pipeline
                        Phase6Pipeline(str(output / 'inference_config.yaml'),
                            inference_identity=partial['inference_identity']).run(str(source), str(output))
                    elif _partial_inference_enabled(config):
                        mapped_count = _prepare_inference(store, job, settings, config, original, output,
                                                          inference_base_identity=identity)
                    else:
                        mapped_count = _prepare_inference(store, job, settings, config, original, output)
                    if checkpoint_enabled:
                        if _checkpoint_identity(job, config, original) != identity:
                            raise AnalysisError('Processing inputs changed. The incomplete result was not published; retry with the retained original video.')
                        if _partial_inference_enabled(config):
                            from src.utils.native_inference_session import load_normalized_source
                            if load_normalized_source(output, identity) is None:
                                raise AnalysisError('Saved normalized source is unavailable. The incomplete result was not published; retry from the retained original video.')
                        from src.api.pipeline_checkpoint import write_pipeline_checkpoint
                        write_pipeline_checkpoint(output, identity, required_files=REQUIRED_PIPELINE_FILES,
                                                  optional_files=OPTIONAL_PIPELINE_FILES)
                store.stage(job['id'], 'Preparing annotated playback and validating results', 11)
                check_storage()
                execute([binary('ffmpeg'), '-nostdin', '-y', '-v', 'error', '-xerror', '-i', str(output / 'annotated.mp4'),
                         '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                         '-movflags', '+faststart', str(output / 'annotated_web.mp4')])
                generations = output / '.result'
                generations.mkdir(exist_ok=True)
                with ResultStorage(generations) as storage:
                    result = build_result(output, job, _storage=storage)
                    if mapped_count is not None and mapped_count != result['summary']['frames']:
                        raise AnalysisError('Source timing and normalized analysis frames did not align. The incomplete result was not published.')
                    # Full decode verifies both browser videos and frame alignment.
                    for path in (source, output / 'annotated_web.mp4'):
                        decoded = json.loads(execute([binary('ffprobe'), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                                     '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)]))
                        if int(decoded['streams'][0]['nb_read_frames']) != result['summary']['frames']:
                            raise AnalysisError('Video and analysis frames did not align. The incomplete result was not published.')
                    if checkpoint_enabled:
                        def final_publication_check():
                            store.stage(job['id'], 'Publishing saved analysis', 11)
                            check_storage()
                            if (_checkpoint_identity(job, config, original) != identity
                                    or _load_completed_checkpoint(output, identity) is None):
                                raise AnalysisError('Saved analysis inputs changed. The incomplete result was not published.')
                            if native_source_seal_required:
                                from src.utils.native_inference_session import load_normalized_source
                                if load_normalized_source(output, identity) is None:
                                    raise AnalysisError('Saved normalized source changed. The incomplete result was not published.')
                        final_publication_check()
                        files = {name: output / name for name in REQUIRED_PIPELINE_FILES}
                        files.update({name: output / name for name in OPTIONAL_PIPELINE_FILES
                                      if (output / name).exists()})
                        files['annotated_web.mp4'] = output / 'annotated_web.mp4'
                        files['original' + job['extension']] = original
                        credit = output.parent / (job['id'] + '.annotated_web_credit.json')
                        if credit.exists():
                            files['annotated_web_credit.json'] = credit
                        publish_result(storage, result, output, generation_files=files,
                            analysis_revision=config.get('analysis_revision', 0),
                            before_commit=final_publication_check)
                    else:
                        check_storage()
                        publish_result(storage, result, output)
                store.finish(job['id'])
            except Exception as exc:
                traceback.print_exc()
                if isinstance(exc, JobCancelled):
                    message = 'Analysis cancelled by its owner.'
                elif isinstance(exc, AnalysisError):
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


def _prepare_inference(store, job, settings, config, original, output, *, inference_base_identity=None):
    """Normalize and run the complete pipeline before any checkpoint is sealed."""
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
    if _partial_inference_enabled(config):
        from src.utils.native_inference_session import seal_normalized_source, inference_identity
        current_identity = _checkpoint_identity(job, config, original)
        if inference_base_identity is not None and current_identity != inference_base_identity:
            raise AnalysisError('Processing inputs changed before inference. Retry from the retained original video.')
        seal = seal_normalized_source(output, current_identity)
        Phase6Pipeline(str(job_config), inference_identity=inference_identity(seal)).run(str(source), str(output))
    else:
        Phase6Pipeline(str(job_config)).run(str(source), str(output))
    return media_metadata['timeMapping'].get('frames')


def process_next_work(store, corrections, settings, *, prefer_correction=False):
    """Process one item, alternating queues when both have pending work.

    The exclusive process lock remains the responsibility of ``main``. Keeping
    uploads and semantic corrections in one worker prevents concurrent model
    owners, while alternating avoids starving either queue.
    """
    from src.api.correction_worker import process_correction
    queues = ((corrections.claim, process_correction, False),
              (store.claim, process_job, True))
    if not prefer_correction:
        queues = queues[::-1]
    for claim, process, next_preference in queues:
        item = claim()
        if item is not None:
            process(store, item, settings)
            return True, next_preference
    return False, prefer_correction


def main():
    settings = Settings.from_env()
    store = JobStore(settings.root)
    os.chdir(REPO_ROOT)
    try:
        with FileLock(str(settings.root / 'worker.lock'), timeout=0):
            from src.api.corrections import CorrectionStore
            corrections = CorrectionStore(store)
            store.recover_interrupted(lambda job: recoverable_completed_job(store, job))
            corrections.recover_interrupted()
            prefer_correction = False
            while True:
                processed, prefer_correction = process_next_work(store, corrections, settings,
                    prefer_correction=prefer_correction)
                if not processed:
                    time.sleep(1)
    except Timeout:
        # Another process still holds the lock and owns the queue.
        return


if __name__ == '__main__':
    main()
