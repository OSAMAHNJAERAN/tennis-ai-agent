"""Acquire January development calibration/timestamps, without source images."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    metadata_dir = ROOT / 'data/external/caltennis_metadata_v1'
    reservation_path = ROOT / 'data/yolo26/caltennis_reference_reservation_v1.json'
    reservation = json.loads(reservation_path.read_text(encoding='utf-8'))
    index = [json.loads(line) for line in (metadata_dir / 'metadata_mini.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = [row for row in index if row['session_id'] == '01_23_2026_17_00_court2']
    authorized = [entry for entry in reservation['groups'] if selected[0]['session_id'] in entry['sessions']]
    if len(selected) != 2 or len(authorized) != 1 or authorized[0]['assignment'] != 'reference_development':
        raise ValueError('Only the two explicitly reserved development cameras may be acquired')
    output = ROOT / 'data/external/caltennis_reference_assets_v1'
    output.mkdir(exist_ok=False)
    script = Path(__file__).read_bytes()
    (output / 'acquisition_script_snapshot.py').write_bytes(script)
    receipt = dict(schema_version=1, status='ACQUIRED_DEVELOPMENT_CALIBRATION_TIMESTAMPS_ONLY',
                   dataset=reservation['source'], revision=reservation['revision'], license=reservation['license'],
                   use_scope=reservation['use_scope'], created_utc=datetime.now(timezone.utc).isoformat(),
                   reservation_sha256=hashlib.sha256(reservation_path.read_bytes()).hexdigest(),
                   index_sha256=hashlib.sha256((metadata_dir / 'metadata_mini.jsonl').read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(script).hexdigest(), files=[], actual_video_acquired=0,
                   source_frames_rendered=0, qualification_evidence=False, ball_speed_reference=None)
    for camera, row in enumerate(selected):
        for kind in ('calibration', 'timestamps'):
            member = row[kind]
            if '..' in Path(member).parts or Path(member).is_absolute() or '\\' in member:
                raise ValueError('Unsafe publisher path')
            url = (f"https://huggingface.co/datasets/{reservation['source']}/resolve/{reservation['revision']}/"
                   + urllib.parse.quote(member, safe='/'))
            request = urllib.request.Request(url, headers={'User-Agent': 'TennisAcademicReferenceAudit/1.0'})
            with urllib.request.urlopen(request, timeout=40) as response:
                content = response.read(5 * 1024 ** 2 + 1)
            if len(content) > 5 * 1024 ** 2:
                raise ValueError('Reference asset exceeded the bounded transfer limit')
            target = output / f'camera_{camera}_{kind}{Path(member).suffix}'
            target.write_bytes(content)
            details = {}
            if kind == 'timestamps':
                times = np.load(target, allow_pickle=False)
                if times.ndim != 1 or times.dtype.kind not in 'iuf' or not np.isfinite(times).all():
                    raise ValueError('Publisher timestamps must be a finite numeric vector')
                details = dict(array_shape=list(times.shape), dtype=str(times.dtype),
                               first_value=float(times[0]) if len(times) else None,
                               last_value=float(times[-1]) if len(times) else None,
                               positive_intervals=int(np.count_nonzero(np.diff(times) > 0)),
                               duplicate_or_reversed_intervals=int(np.count_nonzero(np.diff(times) <= 0)),
                               units='UNVERIFIED_PUBLISHER_TIMESTAMP_ARRAY')
            else:
                calibration = json.loads(content)
                details = dict(publisher_keys=sorted(calibration),
                               world_axis_and_units='UNVERIFIED; NOT_ACCEPTED_AS_GROUND_CONTACT_ACCURACY')
            receipt['files'].append(dict(camera_id=row['video_id'], original_session_id=row['session_id'],
                                         original_video_path=row['video'], kind=kind, path=target.name,
                                         source_url=url, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                                         measured_metadata=details))
    receipt['limits'] = ['No source video or images acquired/inspected',
                         'Timestamp units, video-frame association and inter-camera synchronization not yet verified',
                         'Calibration alone is not an independent player-ground or ball-speed reference',
                         'Reserved test-session assets not opened']
    (output / 'acquisition_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
