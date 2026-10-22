"""Retain native stream/PTS metadata for a separately reserved reference source."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    folder = ROOT / 'data/external/nkust_landing_reference_v4'
    media, labels = folder / 'OST01.mp4', folder / 'outdoor_straight.csv'
    reservation = ROOT / 'data/yolo26/nkust_reference_reservation.json'
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
                            '-show_frames', '-show_format', '-of', 'json', str(media)],
                           check=True, capture_output=True, timeout=60)
    native_path = folder / 'OST01_native_probe.json'
    native_path.write_bytes(probe.stdout)
    metadata = json.loads(probe.stdout)
    stream = metadata['streams'][0]
    frames = metadata['frames']
    with labels.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        int(row['Index'])
        float(row['To-Closest-Doubles-Sideline-Distance (m)'])
        float(row['To-Baseline-Distance (m)'])
    out = folder / 'acquisition_receipt.json'
    receipt = {
        'status': 'ACQUIRED_REFERENCE_UNQUALIFIED', 'acquired_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_url': 'https://data.mendeley.com/datasets/75m8vz7jr2/4',
        'doi': '10.17632/75m8vz7jr2.4', 'publisher_license': 'CC-BY-4.0',
        'attribution': 'Lai, Kalin Guanlun; Huang, Hsu-Chun; Lin, Shang-Yi; Lin, Wei-Ting; Lin, Kawuu Weicheng (2024), Tennis Shot Side-View and Top-View Data Set for Player Analysis in Tennis, Mendeley Data V4.',
        'acquisition_method': 'Official public browser Download file links; no login, challenge bypass or permission change',
        'reservation': {'path': str(reservation.relative_to(ROOT)), 'sha256': digest(reservation),
                        'group_id': 'nkust-2023-03-01-outdoor-session', 'partition': 'reserved_reference_test'},
        'media': {'path': str(media.relative_to(ROOT)), 'sha256': digest(media), 'bytes': media.stat().st_size,
                  'file_url': 'https://data.mendeley.com/public-files/datasets/75m8vz7jr2/files/76012c73-6ec3-44b9-bcd2-01e74b27296e/file_downloaded',
                  'width': stream['width'], 'height': stream['height'], 'average_frame_rate': stream['avg_frame_rate'],
                  'nominal_frame_rate': stream['r_frame_rate'], 'time_base': stream['time_base'],
                  'duration_seconds': metadata['format'].get('duration'), 'native_frame_count': len(frames)},
        'landing_reference': {'path': str(labels.relative_to(ROOT)), 'sha256': digest(labels),
                              'bytes': labels.stat().st_size, 'rows': len(rows), 'units': 'm',
                              'file_url': 'https://data.mendeley.com/public-files/datasets/75m8vz7jr2/files/3e99c0c9-b943-4ad2-8bfa-69d04c421dac/file_downloaded',
                              'source': 'Publisher measured landing distances; global-axis and clip-index correspondence pending independent verification'},
        'native_probe': {'path': str(native_path.relative_to(ROOT)), 'sha256': digest(native_path),
                         'all_frame_pts_retained': True, 'frames_rendered_or_model_predictions_run': False},
        'synchronized_speed_reference': None, 'independent_accuracy_computed': False,
        'limits': ['Practice forehand clip is not a singles match.',
                   'No intrinsic/extrinsic camera calibration or contact timing reference has been verified.',
                   'No radar or other same-definition physical-speed reference is supplied by this CSV.',
                   'One acquired clip does not establish coverage for the complete reserved session.'],
        'script_sha256': digest(Path(__file__)),
    }
    out.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'receipt': str(out), 'status': receipt['status'], 'media': receipt['media'],
                      'landing_rows': len(rows)}))


if __name__ == '__main__':
    main()
