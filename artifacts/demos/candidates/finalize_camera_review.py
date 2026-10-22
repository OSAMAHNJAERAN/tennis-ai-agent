"""Attach camera-review evidence only after revalidating each retained original."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
checked_at = datetime.now(timezone.utc).isoformat()

def verified(path):
    records = json.loads(path.read_text(encoding='utf-8'))
    for item in records['candidates']:
        original = Path(item['sourcePath']).resolve()
        original.relative_to(ROOT.resolve())
        if original.stat().st_size != item['sourceBytes']:
            raise ValueError('Original byte size changed')
        with original.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha1').hexdigest() != item['sourceSha1']:
                raise ValueError('Original publisher SHA-1 changed')
        with original.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sourceSha256']:
                raise ValueError('Original local SHA-256 changed')
    return records

path = ROOT / 'candidate_sources.json'
records = verified(path)
for item in records['candidates']:
    item.update(status='CAMERA_REVIEW_UNSUPPORTED_FOR_BASELINE_DEMO', cameraSuitability=False,
                qualificationEvidence=False, publicDisplayLicenseAvailable=True,
                modifications='None; original publisher video retained for camera review.')
    item['cameraReview'] = {
        'reviewedAt': checked_at, 'tool': 'claude-video-vision',
        'publicDemoEligible': False, 'queueCreated': False, 'inferenceRun': False,
        'licenseRequirement': 'Attribute Oleg Yunakov, link CC BY-SA 4.0 and source, disclose modifications, and keep adapted media under the same or compatible license.',
        'requiredRecordingView': 'Fixed elevated baseline view of the full singles court and both active participants.',
        'rejectionCode': 'DIAGONAL_SPECTATOR_VIEW',
        'modelAccuracyEvidence': None,
    }
    if item['id'] == 'alcaraz_wu_2026_06':
        item['cameraReview'].update(
            metadata={'durationSeconds': 7.196, 'width': 3840, 'height': 2160, 'reportedFps': 60},
            extraction={'fps': 'auto', 'resolution': 512, 'viewSample': None, 'extractedFrames': 14, 'reviewedFrames': 14, 'reviewedIntervalSeconds': [0, 6.5]},
            observation='Elevated diagonal stadium spectator view. Both players and court lines appear in the reviewed frames, with minor camera shifts. The camera is off the baseline axis and the net is diagonal; this does not establish the required supported baseline view.',
            sceneAnalysisRequired=False,
        )
    else:
        item['cameraReview'].update(
            metadata={'durationSeconds': 41.158, 'width': 1920, 'height': 1080, 'reportedFps': 60},
            sceneAnalysis={'requestedFilters': {'scene_changes': True, 'motion': True, 'transcription': True}, 'detectedScenes': [], 'transcription': []},
            extraction={'fps': 'auto', 'resolution': 512, 'viewSample': None, 'extractedFrames': 82, 'reviewedTimestampsSeconds': [0, 5, 10, 15, 20, 25, 30, 35, 40]},
            observation='Elevated diagonal spectator view with small camera shifts. The court is broadly visible, but the far participant is near the left image edge at the start. A foreground spectator intrudes at about 25 s, and ball persons enter the playing area between points. No reviewed timestamp shows the required baseline camera axis.',
            sceneAnalysisRequired=True,
        )
path.write_text(json.dumps(records, indent=2), encoding='utf-8')

path = ROOT / 'baseline_source_candidates.json'
records = verified(path)
item = records['candidates'][0]
item.update(status='CAMERA_REVIEW_REJECTED_NON_MATCH_VIDEO', cameraSuitability=False,
            publicDisplayLicenseAvailable=None,
            licenseVerificationStatus='EXTERNAL_SOURCE_LICENSE_REVIEW_PENDING',
            originalAuthorSource='https://www.youtube.com/watch?v=v4Fi5vDuwqI')
item['cameraReview'] = {
    'reviewedAt': checked_at, 'tool': 'claude-video-vision',
    'metadata': {'durationSeconds': 175.086, 'width': 1920, 'height': 1080, 'reportedFps': 30},
    'sceneAnalysis': {'requestedFilters': {'scene_changes': True, 'transcription': True}, 'sceneTimesSeconds': [18, 43, 54, 83, 106, 116, 171], 'transcription': []},
    'extraction': {'initialStrategy': 'Variable low FPS segments around scene boundaries; initial returned labels exceeded duration and were not used as source timestamps.', 'verifiedStrategy': 'Uniform 0.2 FPS, 512 px, 18 evenly spaced returned frames from the full clip.', 'reviewedTimestampsSeconds': list(range(0, 171, 10))},
    'contentIntervals': [
        {'startSeconds': 0, 'endSeconds': 18, 'content': 'Presentation title'},
        {'startSeconds': 18, 'endSeconds': 43, 'content': 'Screenshots of cognitive-highlights interfaces and player celebration images'},
        {'startSeconds': 43, 'endSeconds': 54, 'content': 'Research team slide'},
        {'startSeconds': 54, 'endSeconds': 83, 'content': 'Contributions slide'},
        {'startSeconds': 83, 'endSeconds': 106, 'content': 'Venue still-image collage with footage statistics'},
        {'startSeconds': 106, 'endSeconds': 116, 'content': 'Impact slide with article and website screenshots'},
        {'startSeconds': 116, 'endSeconds': 171, 'content': 'System overview diagram'},
        {'startSeconds': 171, 'endSeconds': 175.086, 'content': 'Thanks slide'},
    ],
    'observation': 'Conference slide presentation, not a raw singles recording. Static court photographs, interface screenshots and research diagrams provide no usable continuous rally interval.',
    'rejectionCode': 'PRESENTATION_SLIDES_NO_RAW_MATCH_INTERVAL',
    'licenseObservation': 'Commons lists CC BY 3.0 with IBM Research attribution but explicitly marks the external YouTube source as not yet license reviewed; public display eligibility is unverified.',
    'publicDemoEligible': False, 'queueCreated': False, 'inferenceRun': False,
    'qualificationEvidence': False, 'modelAccuracyEvidence': None,
}
path.write_text(json.dumps(records, indent=2), encoding='utf-8')

shortlist = json.loads((ROOT / 'camera_metadata_shortlist.json').read_text(encoding='utf-8'))
audit = {
    'schemaVersion': 1, 'reviewedAt': checked_at,
    'purpose': 'Bounded public-source camera search; no public demo or qualification claim.',
    'newOriginalVideosAcquired': 1, 'newSupportedPublicSourcesFound': 0,
    'existingOriginalsPreserved': ['alcaraz_wu_2026_06.webm', 'sabalenka_zheng_2024_20.webm'],
    'originalHashesRevalidated': True, 'jobQueueModified': False, 'mainDemoManifestModified': False,
    'qualificationHoldoutModified': False, 'inferenceRun': False,
    'shortlist': [],
    'result': 'No newly reviewed source satisfies the fixed elevated baseline singles camera requirement. The original two candidates remain unsupported diagonal views; IBM is a presentation with an unreviewed external license claim.',
    'nextMeaningfulAction': 'Obtain two genuinely different user-owned or explicitly public-licensed singles sessions recorded from a fixed elevated baseline with both players and court lines continuously visible. Review camera framing and source license before processing; do not reinterpret unsupported spectator views as successful supported demos.',
    'publicationRequirements': ['Document source, author, exact license and any modification.', 'Verify a clearly authorized public-display license; resolve external-source license review where applicable.', 'For CC BY-SA sources, preserve applicable share-alike terms for adapted media.', 'Only after camera acceptance, run the real pipeline and inspect usable court calibration and both active-player tracks.'],
}
for number, source in enumerate(shortlist, 1):
    title = source['title']
    if 'Davis Cup' in title:
        reason = 'Publisher posters show doubles participants, cropped single-end courts, or low side views; none is a promising full singles baseline recording.'
    elif 'Kalinskaya' in title:
        reason = 'Publisher poster shows either spectators away from the court or a ground-level side view; full video not acquired.'
    else:
        reason = 'Original acquired and reviewed; slide presentation without a raw match interval, with external license review pending.'
    audit['shortlist'].append({'sourcePage': source['imageinfo'].get('descriptionurl'), 'title': title, 'posterPath': str(ROOT / 'posters' / (str(number) + '.jpg')), 'screeningLevel': 'ORIGINAL_VIDEO_REVIEW' if 'Auto curation' in title else 'POSTER_SCREEN_ONLY', 'reason': reason})
(ROOT / 'supported_camera_search_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps({'originalHashesRevalidated': True, 'newSupportedPublicSourcesFound': 0, 'newOriginalVideosAcquired': 1, 'updatedRecords': ['candidate_sources.json', 'baseline_source_candidates.json', 'supported_camera_search_audit.json']}))
