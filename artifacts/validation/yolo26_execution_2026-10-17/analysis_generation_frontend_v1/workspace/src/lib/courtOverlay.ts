import type { AnalysisRun } from '../data/contracts';
import type { Point2D } from './telemetryGeometry';

/** Current image landmarks only; a lost camera never retains anchor markers. */
export function courtOverlayPoints(run: AnalysisRun, frameIndex: number): Point2D[] {
  if (!run.court?.isValid) return [];
  if (run.contractVersion === '2.0' && !run.cameraTracking) return [];
  const points = run.court.courtKeypointsPx;
  if (!run.cameraTracking?.enabled) return points;
  const sample = run.cameraTracking.frames[frameIndex];
  const matrix = sample?.anchorToFramePx;
  if (!sample?.isValid || sample.frame_index !== frameIndex || !matrix || matrix.length !== 3
      || matrix.some(row => row.length !== 3 || row.some(value => !Number.isFinite(value)))) return [];
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix as [[number, number, number], [number, number, number], [number, number, number]];
  const transformed: Point2D[] = [];
  for (const [x, y] of points) {
    const denominator = g * x + h * y + i;
    if (!Number.isFinite(denominator) || Math.abs(denominator) < 1e-10) return [];
    const point: Point2D = [(a * x + b * y + c) / denominator, (d * x + e * y + f) / denominator];
    if (!point.every(Number.isFinite)) return [];
    transformed.push(point);
  }
  return transformed;
}
