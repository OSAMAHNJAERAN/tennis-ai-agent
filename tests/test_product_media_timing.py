import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from src.api.media_timing import frame_mapping


def test_mapping_keeps_lossless_source_pts_and_excludes_duplicate_transitions(tmp_path):
    stats = tmp_path / 'map.txt'
    stats.write_text('0 0 5000 1/1000 0 1/10\n1 0 5000 1/1000 1 1/10\n2 1 5200 1/1000 2 1/10\n')
    path = tmp_path / 'mapping.jsonl'
    summary = frame_mapping(stats, path, 10, 3, 5)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['source_pts'] == '5000'
    assert rows[0]['source_timestamp_seconds'] == 5
    assert rows[0]['source_elapsed_seconds'] == 0
    assert rows[1]['source_frame_index'] == rows[0]['source_frame_index']
    assert summary['duplicatedOutputFrames'] == 1 and summary['droppedSourceFrames'] == 1
    assert summary['validMeasurementTransitionIntervals'] == []


@pytest.mark.parametrize('line', ['1 0 0 1/30 1 1/30', '0 0 0 1/30 3 1/30', 'broken'])
def test_incomplete_or_misaligned_mapping_cannot_validate_timing(tmp_path, line):
    stats = tmp_path / 'map.txt'
    stats.write_text(line)
    with pytest.raises(ValueError):
        frame_mapping(stats, tmp_path / 'mapping.jsonl', 30)


def test_unknown_source_timestamps_remain_null(tmp_path):
    stats = tmp_path / 'map.txt'
    stats.write_text('0 -1 9223372036854775807 0/1 0 1/30\n')
    path = tmp_path / 'mapping.jsonl'
    summary = frame_mapping(stats, path, 30)
    assert summary['status'] == 'PARTIAL_FFMPEG_FRAME_MAP'
    assert json.loads(path.read_text())['source_timestamp_seconds'] is None


@pytest.mark.parametrize('rate,duplicates,dropped', [(10, 5, 0), (2, 0, 3), (5, 0, 0)])
def test_actual_ffmpeg_mapping_preserves_offset_and_frame_changes(tmp_path, rate, duplicates, dropped):
    ffmpeg = os.getenv('TENNIS_FFMPEG') or shutil.which('ffmpeg')
    ffprobe = os.getenv('TENNIS_FFPROBE') or shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('FFmpeg/ffprobe integration tools unavailable')
    source, normalized = tmp_path / 'source.mkv', tmp_path / 'normalized.mp4'
    stats = tmp_path / 'stats.txt'
    subprocess.run([ffmpeg, '-nostdin', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=5:duration=1',
                    '-vf', 'setpts=PTS+5/TB', '-c:v', 'ffv1', str(source)], check=True, capture_output=True)
    subprocess.run([ffmpeg, '-nostdin', '-y', '-v', 'error', '-copyts', '-i', str(source), '-an',
                    '-vf', f'fps={rate},setpts=PTS-STARTPTS', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-stats_enc_pre', str(stats), '-stats_enc_pre_fmt', '{n} {ni} {ptsi} {tbi} {pts} {tb}',
                    str(normalized)], check=True, capture_output=True)
    path = tmp_path / 'mapping.jsonl'
    summary = frame_mapping(stats, path, rate, 5, 5)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    decoded = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'frame=pts_time', '-of', 'json', str(normalized)]))['frames']
    originals = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                           '-show_entries', 'frame=pts_time', '-of', 'json', str(source)]))['frames']
    assert len(decoded) == len(records) == summary['frames']
    for record, actual in zip(records, decoded):
        assert record['timestamp_seconds'] == pytest.approx(float(actual['pts_time']), abs=1e-6)
        assert record['source_timestamp_seconds'] == float(originals[record['source_frame_index']]['pts_time'])
    assert summary['duplicatedOutputFrames'] == duplicates
    assert summary['droppedSourceFrames'] == dropped
    assert summary['status'] == 'CAPTURED_FFMPEG_FRAME_MAP'
    assert summary['validMeasurementTransitionIntervals'] == ([{'startFrame': 1, 'endFrameExclusive': 5}] if rate == 5 else [])
