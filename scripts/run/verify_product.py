"""Verify a persisted product run through HTTP; optionally upload a real video."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--analysis-id')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--require-contract-v2', action='store_true')
    parser.add_argument('--require-image-people', action='store_true', help='Require actual frame-aligned unverified person observations.')
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        assert client.get('/api/health').status_code == 200
        analysis_id = args.analysis_id
        if not analysis_id:
            if args.input is None:
                parser.error('Supply --analysis-id or --input')
            with args.input.open('rb') as stream:
                response = client.post('/api/analyses', params={'filename': args.input.name}, content=stream)
            assert response.status_code == 202, response.text
            analysis_id = response.json()['id']
        endpoint = '/api/analyses/' + analysis_id
        observed = []
        deadline = time.monotonic() + args.timeout
        while True:
            response = client.get(endpoint)
            response.raise_for_status()
            job = response.json()
            state = (job['status'], job['stage'])
            if not observed or tuple(observed[-1]) != state:
                observed.append(state)
                print(f'{analysis_id}: {state[0]} / {state[1]}', flush=True)
            if job['status'] in ('COMPLETED', 'FAILED'):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Job remains live; inspect {endpoint}. Do not submit a duplicate.')
            time.sleep(2)
        assert job['status'] == 'COMPLETED', job['error']
        response = client.get(endpoint + '/result')
        response.raise_for_status()
        result = response.json()
        summary = result['summary']
        assert summary['id'] == analysis_id and summary['fps'] > 0
        assert len(result['detections']['frames']) == summary['frames']
        assert len(result['trajectories']['ballTrajectory']) == summary['frames']
        assert len(result['trajectories']['player1CourtPositions']) == summary['frames']
        assert len(result['trajectories']['player2CourtPositions']) == summary['frames']
        json.dumps(result, allow_nan=False)
        people_checks = None
        if args.require_image_people:
            people = result.get('imagePeople')
            assert people is not None and people['accuracyValidated'] is False
            assert people['coordinateSystem'] == 'normalized_video_pixels'
            assert people['identityScope'] == 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID'
            assert len(people['frames']) == summary['frames']
            counts, role_counts, nullable_ids = [], {}, 0
            dimensions = result['detections']['metadata']
            for index, frame in enumerate(people['frames']):
                assert frame['frame_index'] == index
                assert abs(frame['timestamp_seconds'] - index / summary['fps']) <= .01
                counts.append(len(frame['people']))
                for person in frame['people']:
                    x1, y1, x2, y2 = person['bbox']
                    assert all(math.isfinite(v) for v in person['bbox'])
                    assert 0 <= x1 < x2 <= dimensions['width'] and 0 <= y1 < y2 <= dimensions['height']
                    assert 0 < person['confidence'] <= 1
                    identity = person['sourceTrackId']
                    assert identity is None or (type(identity) is int and identity >= 0)
                    nullable_ids += identity is None
                    role = person['courtRole']
                    assert role in (None, 1, 2)
                    role_counts[str(role)] = role_counts.get(str(role), 0) + 1
                    assert person['participantStatus'] == ('COURT_ROLE_CANDIDATE' if role is not None else 'UNKNOWN')
                    if not summary['capabilities']['court']:
                        assert role is None
            assert sum(counts) > 0 and summary['capabilities']['imagePeople'] is True
            people_checks = {'frames':len(counts),'observation_count':sum(counts),
                             'frames_with_observations':sum(count > 0 for count in counts),
                             'minimum_people_per_frame':min(counts),'maximum_people_per_frame':max(counts),
                             'court_role_counts':role_counts,'observations_without_tracker_id':nullable_ids,
                             'accuracy_validated':False,'participant_accuracy_measured':False}
        assert analysis_id in [run['id'] for run in client.get('/api/runs').json()]
        media_checks = {}
        for kind in ('source', 'annotated'):
            media = client.get(endpoint + '/media/' + kind, headers={'Range': 'bytes=0-1023'})
            assert media.status_code == 206 and len(media.content) == 1024
            assert media.headers['content-type'] == 'video/mp4'
            media_checks[kind] = media.headers['content-range']
        with client.stream('GET', endpoint + '/download/original') as original:
            original.raise_for_status()
            digest = hashlib.sha256()
            for chunk in original.iter_bytes():
                digest.update(chunk)
        original_sha = digest.hexdigest()
        if args.input:
            with args.input.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == original_sha
        report_response = client.get(endpoint + '/download/report')
        report_response.raise_for_status()
        assert report_response.json() == result
        export_checks = {}
        if args.require_contract_v2:
            assert result['contractVersion'] == '2.0'
            for section in ('poseTracking', 'racketTracking'):
                value = result[section]
                if value is not None:
                    assert len(value['frames']) == summary['frames']
                    assert value['coordinateSystem'] == 'normalized_video_pixels'
            html = client.get(endpoint + '/download/readable-report')
            html.raise_for_status()
            assert html.headers['content-type'].startswith('text/html') and analysis_id in html.text
            assert '<script' not in html.text.lower() and '<svg' in html.text
            telemetry = client.get(endpoint + '/download/csv')
            telemetry.raise_for_status()
            rows = list(csv.DictReader(io.StringIO(telemetry.text)))
            assert len(rows) == summary['frames'] and all(row['analysis_id'] == analysis_id for row in rows)
            export_checks = {'readable_report_bytes': len(html.content), 'csv_rows': len(rows),
                             'contract_version': result['contractVersion']}
        report = {'analysis_id': analysis_id, 'job': job, 'observed_states': observed,
                  'summary': summary, 'original_sha256': original_sha, 'media_ranges': media_checks,
                  'player_metrics': result['playerMetrics'], 'pipeline_metrics': result['pipelineMetrics'],
                  'warnings': result['warnings'], 'result_sha256': hashlib.sha256(response.content).hexdigest(),
                  'export_checks': export_checks,
                  'image_people_checks':people_checks,
                  'verification': 'HTTP_ARTIFACT_AND_CONTRACT_CHECK; NOT_MODEL_ACCURACY'}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Verified {summary["frames"]} frames; report: {args.report}', flush=True)


if __name__ == '__main__':
    main()
