"""Retain actual DEV PTS and exact native-frame review anchors, no inference."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
DATA = ROOT/'data/external/nkust_landing_reference_indoor_development_v4'
OUT = ART/'native_review'
OUT.mkdir(exist_ok=False)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

ffprobe, ffmpeg = shutil.which('ffprobe'), shutil.which('ffmpeg')
assert ffprobe and ffmpeg
acquired = json.loads((ART/'media_acquisition_receipt.json').read_text(encoding='utf-8'))
assert acquired['status'] == 'ACQUIRED_PUBLISHER_HASH_VERIFIED_INDOOR_DEVELOPMENT_PAIR'
observations = []
for row in acquired['files']:
    source = ROOT/row['path']
    assert sha(source) == row['sha256'] == row['publisher_sha256']
    probe = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_streams', '-show_frames', '-show_format', '-of', 'json', str(source)],
        capture_output=True, check=True, timeout=40)
    target = OUT/(source.stem+'_native_probe.json')
    target.write_bytes(probe.stdout)
    value = json.loads(probe.stdout)
    stream, frames = value['streams'][0], value['frames']
    timestamps = [int(frame['best_effort_timestamp']) for frame in frames]
    assert timestamps and all(a < b for a, b in zip(timestamps, timestamps[1:]))
    selected = sorted(set([0, len(frames)//2, 3*len(frames)//4, len(frames)-1]))
    reviews = []
    for index in selected:
        image = OUT/(source.stem+f'_frame_{index:04}.png')
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(source),
            '-vf', f'select=eq(n\\,{index})', '-frames:v', '1', '-fps_mode', 'vfr', str(image)],
            capture_output=True, check=True, timeout=40)
        reviews.append({'native_frame_index': index, 'source_pts': str(timestamps[index]),
            'source_time_base': stream['time_base'], 'source_pts_time': frames[index].get('best_effort_timestamp_time'),
            'path': str(image.relative_to(ROOT)), 'sha256': sha(image)})
    observations.append({'filename': source.name, 'source_sha256': sha(source),
        'width': stream['width'], 'height': stream['height'], 'avg_frame_rate': stream['avg_frame_rate'],
        'r_frame_rate': stream['r_frame_rate'], 'time_base': stream['time_base'],
        'frame_count': len(frames), 'duration_seconds': value['format']['duration'],
        'first_pts': str(timestamps[0]), 'last_pts': str(timestamps[-1]),
        'probe_path': str(target.relative_to(ROOT)), 'probe_sha256': sha(target),
        'exact_native_frame_anchors': reviews})
labels = DATA/'Indoor Field - Straight Shot.csv'
csv_before = sha(labels)
assert csv_before == '6ea7649134718c692f31aacf55095e1797aec673b2905a7e1215aae21ba3b53d'
with labels.open(encoding='utf-8-sig', newline='') as handle:
    reader = csv.DictReader(handle)
    headers = reader.fieldnames
    ids = [row['Index'] for row in reader]
assert len(ids) == len(set(ids)) and sha(labels) == csv_before
receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'INDOOR RESERVED DEVELOPMENT ONLY; ACTUAL FFMPEG NATIVE PTS AND SPARSE FRAME ANCHORS',
    'files': observations, 'csv_sha256': csv_before, 'csv_headers': headers,
    'csv_row_count': len(ids), 'csv_identifier_examples': ids[:3]+ids[-3:],
    'csv_values_used_for_accuracy_or_tuning': False, 'exact_csv_clip_join_verified': False,
    'outdoor_test_media_or_target_values_accessed': False,
    'ffprobe_path': ffprobe, 'ffmpeg_path': ffmpeg, 'model_or_training_executed': False,
    'cross_camera_synchronization_verified': False, 'reviewer': 'PENDING_AGENT_VISUAL_REVIEW',
    'runner_sha256': sha(Path(__file__))}
(ART/'native_inspection_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': 'NATIVE_DEV_INSPECTION_RETAINED', 'files': [
    {key: row[key] for key in ('filename', 'frame_count', 'avg_frame_rate', 'time_base')} for row in observations],
    'csv_row_count': len(ids), 'csv_identifier_examples': receipt['csv_identifier_examples']}, indent=2))
