import type { AnalysisRun, BallTrajectoryPoint, MatchEvent, TrackingState } from "../data/contracts";

export const TRACKING_STATES: TrackingState[] = [
  "DETECTED",
  "TRACKED",
  "PREDICTED",
  "INTERPOLATED",
  "OCCLUDED",
  "MISSING",
];

export function trackingDistribution(points: BallTrajectoryPoint[]) {
  const counts = new Map<TrackingState, number>(TRACKING_STATES.map((state) => [state, 0]));
  for (const point of points) counts.set(point.state, (counts.get(point.state) ?? 0) + 1);
  return TRACKING_STATES.map((state) => ({ state, count: counts.get(state) ?? 0 }));
}

export function deriveCoverage(points: BallTrajectoryPoint[]) {
  if (!points.length) return null;
  const observed = points.filter((point) => point.state !== "MISSING" && point.x_px != null && point.y_px != null).length;
  return (observed / points.length) * 100;
}

export function getAdjacentEvent(events: MatchEvent[], selectedId: number | null, direction: -1 | 1) {
  if (!events.length) return null;
  const selectedIndex = events.findIndex((event) => event.event_id === selectedId);
  const anchor = selectedIndex < 0 ? (direction === 1 ? -1 : 0) : selectedIndex;
  const nextIndex = Math.min(events.length - 1, Math.max(0, anchor + direction));
  return events[nextIndex] ?? null;
}

export function eventParticipation(run: AnalysisRun, playerId: number) {
  return run.events.filter((event) => event.player_id === playerId).length;
}

export function currentFrameIndex(time: number, fps: number, frameCount: number) {
  return Math.min(frameCount - 1, Math.max(0, Math.round(time * fps)));
}

export function mapCourtPoint(point: [number, number], width: number, height: number, padding = 20) {
  return {
    x: padding + (point[0] / 10.97) * (width - padding * 2),
    y: padding + (point[1] / 23.77) * (height - padding * 2),
  };
}

