import type { AnalysisRun, LandingZone } from '../data/contracts';
import type { Point2D } from './telemetryGeometry';

export type LandingPartition = LandingZone['partition'];
export type LandingPresentation = { visible: boolean; partition: LandingPartition };
export const DEFAULT_LANDING_PRESENTATION: LandingPresentation = { visible: true, partition: 'tactical_3x3' };

/** Reproject metric boundaries with the presented frame's calibration. */
export function projectCourtPolygon(points: readonly Point2D[], matrix: number[][] | null): Point2D[] | null {
  if (!matrix || matrix.length !== 3 || matrix.some(row => row.length !== 3 || row.some(value => !Number.isFinite(value)))) return null;
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix as [[number, number, number], [number, number, number], [number, number, number]];
  const inverse = [[e * i - f * h, c * h - b * i, b * f - c * e],
    [f * g - d * i, a * i - c * g, c * d - a * f], [d * h - e * g, b * g - a * h, a * e - b * d]];
  const determinant = a * inverse[0]![0]! + b * inverse[1]![0]! + c * inverse[2]![0]!;
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) return null;
  let sign = 0;
  const result: Point2D[] = [];
  for (const [x, y] of points) {
    const denominator = inverse[2]![0]! * x + inverse[2]![1]! * y + inverse[2]![2]!;
    if (!Number.isFinite(denominator) || Math.abs(denominator) < 1e-12 || (sign && Math.sign(denominator) !== sign)) return null;
    sign = Math.sign(denominator);
    const point: Point2D = [(inverse[0]![0]! * x + inverse[0]![1]! * y + inverse[0]![2]!) / denominator,
      (inverse[1]![0]! * x + inverse[1]![1]! * y + inverse[1]![2]!) / denominator];
    if (!point.every(Number.isFinite)) return null;
    result.push(point);
  }
  return result;
}

const equalPoint = (first: unknown, second: Point2D) => Array.isArray(first) && first.length === 2
  && first.every((value, index) => typeof value === 'number' && Number.isFinite(value) && Math.abs(value - second[index]!) < 1e-6);

/** Bind once; time lookup considers only the preceding one second of events.
 * Missing evidence or calibration suppresses both video and mini-court fills.
 */
export function createLandingZoneIndex(run: AnalysisRun) {
  const events = new Map(run.events.map(event => [event.event_id, event]));
  const records = (run.landingZones ?? []).filter(row => {
    const event = events.get(row.event_id); const evidence = event?.evidence;
    return row.analysis_id === run.summary.id && row.analysis_revision === run.analysisRevision
      && event?.event_type === 'BOUNCE' && evidence?.active_play === true && evidence.position_type === 'ground_contact'
      && evidence.review_status === row.review_status && evidence.analysis_id === row.analysis_id
      && evidence.analysis_revision === row.analysis_revision && evidence.calibration_id === row.calibration_id
      && evidence.ground_contact_source === row.source && evidence.position_uncertainty_m === row.position_uncertainty_m
      && equalPoint(event.court_position_m, row.contact_position_m) && equalPoint(evidence.contact_position_px, row.contact_position_px)
      && Math.abs(event.timestamp_s - row.timestamp_s) < 1e-6;
  }).sort((a, b) => a.timestamp_s - b.timestamp_s);
  const lowerBound = (timestamp: number) => {
    let left = 0, right = records.length;
    while (left < right) { const middle = (left + right) >>> 1; if (records[middle]!.timestamp_s < timestamp) left = middle + 1; else right = middle; }
    return left;
  };
  return (timestamp: number, frame: number, partition: LandingPartition): LandingZone[] => {
    if (!run.court?.isValid || !Number.isFinite(timestamp)) return [];
    const sample = run.cameraTracking?.frames[frame];
    if (!sample?.isValid || sample.frame_index !== frame || !sample.imageToCourt) return [];
    const visible: LandingZone[] = [];
    for (let index = lowerBound(timestamp - 1); index < records.length; ++index) {
      const row = records[index]!;
      if (row.timestamp_s > timestamp) break;
      if (row.partition !== partition || timestamp >= row.timestamp_s + row.duration_s) continue;
      const eventFrame = events.get(row.event_id)!.frame;
      const origin = run.cameraTracking!.frames[eventFrame];
      if (!origin?.isValid || origin.frame_index !== eventFrame || eventFrame > frame) continue;
      // A loss of calibration ends the display interval even if it recovers.
      const segment = run.courtSegments?.find(value => value.startFrame <= eventFrame && eventFrame < value.endFrameExclusive);
      if (!segment || segment.calibrationStatus !== 'VALID_COURT_PLANE' || frame >= segment.endFrameExclusive) continue;
      const contact = projectCourtPolygon([row.contact_position_m], sample.imageToCourt);
      const polygons = row.zones.map(zone => projectCourtPolygon(zone.polygon_m, sample.imageToCourt));
      if (!contact || polygons.some(value => value === null)) continue;
      visible.push({ ...row, contact_position_px: [...contact[0]!] as [number, number],
        zones: row.zones.map((zone, index) => ({ ...zone, polygon_px: polygons[index]!.map(point => [...point] as [number, number]) })) });
    }
    return visible;
  };
}
