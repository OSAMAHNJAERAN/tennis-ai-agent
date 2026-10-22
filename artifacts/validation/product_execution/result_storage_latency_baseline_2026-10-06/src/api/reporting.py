"""Self-contained, escaped reports and telemetry exports for one saved analysis."""

import base64
from collections import Counter
import csv
from html import escape
from itertools import islice
import math
from pathlib import Path

from src.api.media_source import validate_media_source
from src.api.result_storage import DiskSeries


def text(value):
    return escape(str(value), quote=True)


def numeric(value, digits=1):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else 'Unavailable'


def source_credit(source):
    source = validate_media_source(source)
    if source is None:
        return ''
    sharing = ' Share-alike derivative license applies to the adapted footage and evidence images.' if 'BY-SA' in source['derivativeLicense'] else ''
    return (f'{text(source["title"])} by {text(source["author"])} · '
            f'<a href="{text(source["sourcePage"])}" rel="noopener noreferrer">Source footage</a> · '
            f'<a href="{text(source["licenseUrl"])}" rel="noopener noreferrer">{text(source["license"])}</a>. '
            f'Annotated footage and included evidence images: {text(source["derivativeLicense"])}. '
            f'Changes: {text(source["modifications"])}{sharing}')


def csv_credit(value):
    # Quote formula-like attribution cells for readers opening CSV in Excel.
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


class _CsvRow:
    def write(self, value):
        # csv.writer returns this value from writerow; retain only one row.
        return value


def telemetry_csv_chunks(result, chunk_chars=64 * 1024):
    """Yield CSV without retaining a complete serialized copy of the result.

    Chunks contain whole rows and target at most ``chunk_chars`` characters.
    The header or one longer row can exceed that target. The saved result is
    still held by the caller; this bounds additional CSV serialization memory.
    """
    if not isinstance(chunk_chars, int) or isinstance(chunk_chars, bool) or chunk_chars < 1:
        raise ValueError('CSV chunk size must be a positive integer.')
    writer = csv.writer(_CsvRow())
    yield writer.writerow(['analysis_id', 'frame', 'normalized_seconds', 'ball_state', 'ball_x_px', 'ball_y_px',
                     'ball_confidence', 'ball_projected_court_x_m', 'ball_projected_court_y_m',
                     'player1_ground_x_m', 'player1_ground_y_m', 'player1_speed_kmh',
                     'player2_ground_x_m', 'player2_ground_y_m', 'player2_speed_kmh',
                     'media_source_id', 'media_title', 'media_author', 'media_source_page',
                     'media_license', 'media_license_url', 'media_derivative_license', 'media_modifications',
                     'media_source_sha256', 'media_input_sha256'])
    source = validate_media_source(result.get('mediaSource')) or {}
    credit = [csv_credit(source.get(key)) for key in ('id', 'title', 'author', 'sourcePage', 'license', 'licenseUrl',
                                                     'derivativeLicense', 'modifications', 'sourceSha256', 'inputSha256')]
    trajectories = result.get('trajectories') or {}
    player_series = [(trajectories.get(f'player{player}CourtPositions') or [],
                      trajectories.get(f'player{player}Speeds') or []) for player in (1, 2)]
    buffered, buffered_chars = [], 0
    for index, ball in enumerate(trajectories.get('ballTrajectory', [])):
        row = [result['summary']['id'], index, ball.get('timestamp_seconds'), ball.get('state'),
               ball.get('x_px'), ball.get('y_px'), ball.get('confidence'), ball.get('court_x_m'), ball.get('court_y_m')]
        for points, speeds in player_series:
            position = points[index] if index < len(points) else None
            row.extend([*(position or [None, None]), speeds[index] if index < len(speeds) else None])
        line = writer.writerow([*row, *credit])
        if buffered and buffered_chars + len(line) > chunk_chars:
            yield ''.join(buffered)
            buffered.clear()
            buffered_chars = 0
        buffered.append(line)
        buffered_chars += len(line)
    if buffered:
        yield ''.join(buffered)


def telemetry_csv(result):
    """Compatibility helper for callers that explicitly need one CSV string."""
    return ''.join(telemetry_csv_chunks(result))


def heatmap_svg(points, color):
    """Count occupied court-plane cells; never use airborne ball projection."""
    cells = Counter()
    for point in points:
        if isinstance(point, list) and len(point) == 2 and all(isinstance(v, (float, int)) and math.isfinite(v) for v in point):
            x, y = point
            if -3 <= x <= 13.97 and -4 <= y <= 27.77:
                cells[int((x + 3) * 2), int((y + 4) * 2)] += 1
    peak = max(cells.values(), default=1)
    rectangles = ''.join(f'<rect x="{x * 5}" y="{y * 5}" width="5" height="5" fill="{color}" opacity="{.15 + .7 * count / peak:.3f}"/>'
                         for (x, y), count in cells.items())
    return f'<svg role="img" aria-label="Player ground-position occupancy heatmap" viewBox="0 0 170 320">{rectangles}<g fill="none" stroke="#597568" stroke-width=".8"><rect x="30" y="40" width="109.7" height="237.7"/><path d="M43.7 40v237.7M126 40v237.7M43.7 94.85h82.3M43.7 222.85h82.3M84.85 94.85v128M30 158.85h109.7"/></g></svg>'


def speed_svg(samples, fps):
    count = len(samples)
    stride = max(1, math.ceil(count / 600))
    invalid_bins = set()
    finite_peak = None
    for index, value in enumerate(samples):
        if isinstance(value, (float, int)) and math.isfinite(value):
            if finite_peak is None or value > finite_peak:
                finite_peak = value
        else:
            invalid_bins.add(index // stride)
    if finite_peak is None:
        return '<p>Movement speed unavailable.</p>'
    peak = max(5., finite_peak)
    paths, segment = [], []
    values = iter(samples)
    for index in range(0, count, stride):
        interval = islice(values, stride)
        if index // stride in invalid_bins:
            # Consume the gap without summing any of it, as the list version
            # did. There are at most 600 validity flags for the whole chart.
            for _ in interval:
                pass
            if segment:
                paths.append(segment)
                segment = []
            continue
        # Built-in sum retains the original accumulation order and precision;
        # the iterator avoids retaining up to one match's stride.
        speed = sum(interval) / min(stride, count - index)
        segment.append(f'{30 + 540 * index / max(1, count - 1):.2f},{150 - 120 * speed / peak:.2f}')
    if segment:
        paths.append(segment)
    lines = ''.join(f'<polyline fill="none" stroke="#386f58" stroke-width="2" points="{" ".join(segment)}"/>' for segment in paths)
    return f'<svg role="img" aria-label="Movement speed over normalized video time" viewBox="0 0 600 185"><path d="M30 20v130h540" fill="none" stroke="#c6cdc6"/>{lines}<g font-size="12" fill="#44584b"><text x="30" y="15">{peak:.0f} km/h</text><text x="30" y="175">0 s</text><text x="525" y="175">{count / fps:.1f} s</text></g></svg>'


def evidence_images(directory, result):
    if directory is None or not (directory / 'annotated_web.mp4').is_file():
        return ''
    # Optional embedded JPEGs make this export usable without the server.
    try:
        import cv2
        video = cv2.VideoCapture(str(directory / 'annotated_web.mp4'))
        figures = []
        count, fps = result['summary']['frames'], result['summary']['fps']
        for frame in sorted({0, count // 2, max(0, count - 1)}):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = video.read()
            if not ok:
                continue
            height, width = image.shape[:2]
            image = cv2.resize(image, (min(960, width), max(1, round(height * min(960, width) / width))))
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                uri = base64.b64encode(encoded.tobytes()).decode('ascii')
                credit = source_credit(result.get('mediaSource'))
                figures.append(f'<figure><img src="data:image/jpeg;base64,{uri}" alt="Annotated evidence at frame {frame}"/><figcaption>Frame {frame} · {frame / fps:.3f} s{f"<br>{credit}" if credit else ""}</figcaption></figure>')
        video.release()
        return ''.join(figures)
    except (ImportError, OSError, ValueError):
        return ''


def image_people_summary(result):
    """Describe exported observations without equating proposals with players."""
    evidence = result.get('imagePeople')
    if not isinstance(evidence, dict) or not _is_array(evidence.get('frames')):
        return '<p>Image person detections were not exported for this analysis.</p>'
    frames = evidence['frames']
    boxes, positive, available = 0, 0, 0
    for frame in frames:
        if isinstance(frame, dict) and _is_array(frame.get('people')):
            count = len(frame['people'])
            boxes += count
            positive += count > 0
            available += 1
    total = result['summary']['frames']
    return (f'<p>Detected people: {boxes:,} observed boxes across {positive:,} of {total:,} frames. '
            f'Person evidence is available for {available:,} frames; an empty observation list means no valid person boxes were exported for that frame.</p>'
            '<p>These detections may include spectators, officials and false detections. Active participants and persistent person identity are not independently verified. '
            'Person boxes do not establish player movement, speed or court positions. Availability is not detection accuracy.</p>')


def _is_array(value):
    return isinstance(value, (list, DiskSeries))


def readable_report_chunks(result, directory: Path | None = None):
    """Yield a self-contained report with bounded charts and streamed rows."""
    summary = result['summary']
    trajectories = result.get('trajectories') or {}
    players = result.get('playerMetrics') or {}
    yield f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tennis analysis · {text(summary['label'])}</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#edf0e9;color:#24372b;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1080px;margin:0 auto;padding:48px 28px}}header,section{{background:#fff;padding:28px;margin-bottom:20px;border-radius:12px}}h1{{font-size:36px;line-height:1.15;letter-spacing:-1px}}h2{{font-size:22px}}small,p,dt{{color:#55675b}}dl{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}dt{{font-size:13px}}dd{{margin:0;font-size:21px;font-variant-numeric:tabular-nums}}.figures{{display:grid;grid-template-columns:3fr 1fr;align-items:center;gap:28px}}svg{{width:100%;max-height:290px}}figure{{margin:18px 0}}img{{width:100%;border-radius:8px}}figcaption{{font-size:13px}}code{{overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #dee4dc}}@media(max-width:650px){{main{{padding:18px 12px}}header,section{{padding:20px}}dl{{grid-template-columns:1fr 1fr}}.figures{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{padding:0}}section{{break-inside:avoid}}}}
    </style></head><body><main><header><small>TENNIS VIDEO ANALYSIS · SAVED REPORT</small><h1>{text(summary['label'])}</h1><p>Analysis ID: <code>{text(summary['id'])}</code><br>Processed: {text(summary['analyzedAt'])}<br>{summary['frames']} normalized frames · {numeric(summary['fps'], 3)} FPS · {numeric(summary['durationSeconds'])} seconds</p><p>Movement estimates describe the observed ground-position intervals. Physical ball speed and reconstructed flight are unavailable. Player labels identify tracked court roles; they are not recognized names.</p></header>'''
    for player in (1, 2):
        metrics = players.get(f'player_{player}') or (result.get('playerTracking') or {}).get(f'player_{player}', {}).get('metrics', {})
        samples = trajectories.get(f'player{player}Speeds') or []
        points = trajectories.get(f'player{player}CourtPositions') or []
        pose = (result.get('poseTracking') or {}).get('coveragePct', {}).get(f'player_{player}')
        racket = (result.get('racketTracking') or {}).get('coveragePct', {}).get(f'player_{player}')
        intervals = (result.get('playerTracking') or {}).get(f'player_{player}', {}).get('speedIntervals', [])
        interval_text = ', '.join(f'{value["startSeconds"]:.2f}–{value["endSeconds"]:.2f} s' for value in islice(intervals, 30)) or 'Not recorded in this contract version'
        if len(intervals) > 30:
            interval_text += f'; {len(intervals) - 30} further intervals in JSON'
        yield f'<section><h2>Player {player} · tracked court role</h2><dl><div><dt>Distance estimate</dt><dd>{numeric(metrics.get("total_distance_m"))} m</dd></div><div><dt>Average speed estimate</dt><dd>{numeric(metrics.get("avg_speed_kmh"))} km/h</dd></div><div><dt>Peak speed estimate</dt><dd>{numeric(metrics.get("max_speed_kmh"))} km/h</dd></div><div><dt>Ground-position availability</dt><dd>{numeric(metrics.get("coverage_pct"))}%</dd></div><div><dt>Pose frame availability</dt><dd>{numeric(pose)}%</dd></div><div><dt>Racket frame availability</dt><dd>{numeric(racket)}%</dd></div></dl><div class="figures">{speed_svg(samples, summary["fps"])}{heatmap_svg(points, "#386f58" if player == 1 else "#487f9d")}</div><p>Valid speed intervals: {text(interval_text)}. Missing intervals are excluded. Availability is not a measure of accuracy.</p></section>'
    states = Counter(ball.get('state', 'MISSING') for ball in trajectories.get('ballTrajectory', []))
    state_text = ' · '.join(f'{state.lower()}: {count} frames' for state, count in sorted(states.items()))
    people_quality = image_people_summary(result)
    court = result.get('court') or {}
    yield f'<section><h2>Tracking and input quality</h2><p>{text(state_text)}</p><p>Ball tracking availability does not certify correct detections. Court fit residual: {numeric(court.get("reprojectionErrorPx"), 2)} px. This measures fit consistency, not independent calibration error.</p>{people_quality}<ul>'
    for warning in result.get('warnings', []):
        yield f'<li>{text(warning)}</li>'
    yield '</ul></section><section><h2>Validated events</h2>'
    events = iter(result.get('events', []))
    sentinel = object()
    event = next(events, sentinel)
    if event is sentinel:
        yield '<p>No qualified contact or bounce events are available for this clip.</p>'
    else:
        yield '<table><tr><th>Event</th><th>Time</th><th>Confidence</th></tr>'
        while event is not sentinel:
            yield f'<tr><td>{text(event.get("event_type"))}</td><td>{numeric(event.get("timestamp_s"), 3)} s</td><td>{numeric(event.get("confidence"), 2)}</td></tr>'
            event = next(events, sentinel)
        yield '</table>'
    yield '</section>'
    provenance = result.get('provenance') or {}
    bundle = provenance.get('bundleVersion') or 'Version not recorded in this historical result'
    images = evidence_images(directory, result)
    credit = source_credit(result.get('mediaSource'))
    source = validate_media_source(result.get('mediaSource'))
    excerpt = source.get('excerpt') if source else None
    excerpt_note = (f'<p>Excerpt: source {numeric(excerpt["startSeconds"], 3)}–{numeric(excerpt["endSeconds"], 3)} s; '
                    f'frames {excerpt["startFrame"]}–{excerpt["endFrameExclusive"]} (end exclusive). '
                    'Report timestamps start at the normalized analysis clip.</p>') if excerpt else ''
    source_section = f'<section><h2>Footage credit and sharing</h2><p>{credit}</p>{excerpt_note}</section>' if credit else ''
    yield f'''{source_section}{f'<section><h2>Timestamped evidence</h2>{images}</section>' if images else ''}<section><h2>Technical appendix and limitations</h2><p>Bundle: <code>{text(bundle)}</code><br>Contract: {text(result.get('contractVersion', 'historical'))}<br>Pipeline throughput: {numeric((result.get('pipelineMetrics') or {}).get('pipelineFps'), 2)} frames/s on the processing host.</p><p>Image coordinates use normalized video pixels; ground coordinates use meters. The airborne ball projection is a viewing ray intersecting the court plane. Skeletons and racket boxes are image observations. No segmentation or independently validated physical ball flight is claimed. Model confidence and export coverage are not accuracy scores. Single-camera mostly static singles footage is the current development scope; full-match capacity is not qualified.</p><p>Checkpoint hashes and all available measurement intervals are retained in the analysis JSON. Source/annotated video and CSV downloads remain associated with this analysis ID.</p>{f'<p>Export credit: {credit}</p>' if credit else ''}</section></main></body></html>'''


def readable_report(result, directory: Path | None = None):
    """Compatibility helper for callers that explicitly need one HTML string."""
    return ''.join(readable_report_chunks(result, directory))
