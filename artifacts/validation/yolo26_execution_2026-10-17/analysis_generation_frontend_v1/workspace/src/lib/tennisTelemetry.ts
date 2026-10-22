import type { AnalysisRun, BallTrajectoryPoint, TrackingState } from "../data/contracts";

// Mirrors src/court/court_geometry.py. Units are meters; origin is the far-left doubles corner.
export const COURT = { width: 10.97, length: 23.77, singles: 8.23, alley: 1.37, service: 6.4, netCenter: 0.914, netPost: 1.067 } as const;
export const FIELD = { minX: -3.5, maxX: 14.47, minY: -5, maxY: 28.77 } as const;
export const PLAYER_COLORS = ["#c6f66b", "#7cd6ec"] as const;
export type CourtPoint = readonly [number, number];
export type PlayerFilter = "combined" | "player1" | "player2";
export type FrameRange = readonly [number, number];
export type SpatialSample = { position: CourtPoint; state: TrackingState; confidence: number | null; speed: number | null };

export function validPoint(point: CourtPoint | null | undefined): point is CourtPoint {
  return !!point && point.every(Number.isFinite) && point[0] >= FIELD.minX && point[0] <= FIELD.maxX && point[1] >= FIELD.minY && point[1] <= FIELD.maxY;
}

export function courtToWorld(point: CourtPoint, height = 0): [number, number, number] {
  return [point[0] - COURT.width / 2, height, point[1] - COURT.length / 2];
}

export function distance(a: CourtPoint, b: CourtPoint) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function interpolatePosition(points: ReadonlyArray<CourtPoint | null>, frame: number, maxJump = 1): CourtPoint | null {
  const index = Math.max(0, Math.floor(frame));
  const a = points[index];
  if (!validPoint(a)) return null;
  const t = frame - index;
  const b = points[index + 1];
  if (!t || !validPoint(b) || distance(a, b) > maxJump) return a;
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

export function normalizeTelemetry(run: AnalysisRun) {
  const count = run.summary.frames;
  const usable = run.court?.isValid === true;
  const players: Array<Array<CourtPoint | null>> = [run.trajectories?.player1CourtPositions, run.trajectories?.player2CourtPositions].map(points =>
    Array.from({ length: count }, (_, index) => usable && validPoint(points?.[index]) ? points![index]! : null),
  );
  const ball: Array<SpatialSample | null> = Array.from({ length: count }, () => null);
  if (usable) for (const sample of run.trajectories?.ballTrajectory ?? []) {
    const point: CourtPoint | null = sample.court_x_m == null || sample.court_y_m == null ? null : [sample.court_x_m, sample.court_y_m];
    if (sample.frame_index < 0 || sample.frame_index >= count || !validPoint(point) || ["MISSING", "OCCLUDED"].includes(sample.state)) continue;
    ball[sample.frame_index] = { position: point, state: sample.state, confidence: sample.confidence, speed: sample.speed_kmh };
  }
  return { players, ball, ballPositions: ball.map(sample => sample?.position ?? null), fps: run.summary.fps, count };
}
export type NormalizedTelemetry = ReturnType<typeof normalizeTelemetry>;

export function splitTrail(points: ReadonlyArray<CourtPoint | null>, end: number, count: number, maxJump: number): CourtPoint[][] {
  const paths: CourtPoint[][] = [];
  let path: CourtPoint[] = [];
  for (let i = Math.max(0, Math.floor(end) - count); i <= Math.floor(end); i++) {
    const point = points[i];
    const last = path.at(-1);
    if (!validPoint(point) || (last && distance(last, point) > maxJump)) {
      if (path.length > 1) paths.push(path);
      path = [];
    }
    if (validPoint(point)) path.push(point);
  }
  if (path.length > 1) paths.push(path);
  return paths;
}

export function playerSpeed(points: ReadonlyArray<CourtPoint | null>, frame: number, fps: number): number | null {
  const end = Math.floor(frame);
  const start = Math.max(0, end - Math.max(1, Math.round(fps / 4)));
  if (start === end) return null;
  let meters = 0;
  for (let i = start + 1; i <= end; i++) {
    const a = points[i - 1]; const b = points[i];
    if (!validPoint(a) || !validPoint(b) || distance(a, b) * fps > 14) return null;
    meters += distance(a, b);
  }
  return meters / ((end - start) / fps) * 3.6;
}

export function samplePoints(telemetry: NormalizedTelemetry, filter: PlayerFilter, range: FrameRange): CourtPoint[] {
  return telemetry.players.flatMap((points, index) => {
    if (filter !== "combined" && filter !== `player${index + 1}`) return [];
    return points.slice(range[0], range[1] + 1).filter(validPoint);
  });
}

export type SpatialStats = ReturnType<typeof spatialStats>;
export function spatialStats(points: CourtPoint[]) {
  const total = points.length;
  const share = (predicate: (p: CourtPoint) => boolean) => total ? points.filter(predicate).length / total * 100 : 0;
  return {
    total,
    behindBaseline: share(p => p[1] < 0 || p[1] > COURT.length),
    backcourt: share(p => p[1] <= COURT.length / 2 - COURT.service || p[1] >= COURT.length / 2 + COURT.service),
    frontcourt: share(p => p[1] > COURT.length / 2 - COURT.service && p[1] < COURT.length / 2 + COURT.service),
    center: share(p => Math.abs(p[0] - COURT.width / 2) <= COURT.singles / 6),
    left: share(p => p[0] < COURT.width / 2),
  };
}

export function densityGrid(points: CourtPoint[], columns = 25, rows = 45, sigma = 0.9) {
  // Bin first, then evaluate a Gaussian kernel. Cost stays bounded for long recordings.
  const counts = new Map<number, number>();
  const dx = (FIELD.maxX - FIELD.minX) / columns;
  const dy = (FIELD.maxY - FIELD.minY) / rows;
  for (const point of points) {
    if (!validPoint(point)) continue;
    const col = Math.min(columns - 1, Math.floor((point[0] - FIELD.minX) / dx));
    const row = Math.min(rows - 1, Math.floor((point[1] - FIELD.minY) / dy));
    const key = row * columns + col;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const bins = [...counts].map(([key, count]) => ({ x: FIELD.minX + (key % columns + 0.5) * dx, y: FIELD.minY + (Math.floor(key / columns) + 0.5) * dy, count }));
  return Array.from({ length: columns * rows }, (_, key) => {
    const x = FIELD.minX + (key % columns + 0.5) * dx;
    const y = FIELD.minY + (Math.floor(key / columns) + 0.5) * dy;
    const value = bins.reduce((sum, bin) => {
      const d2 = (x - bin.x) ** 2 + (y - bin.y) ** 2;
      return d2 > 9 * sigma * sigma ? sum : sum + bin.count * Math.exp(-d2 / (2 * sigma * sigma));
    }, 0);
    return { x, y, value, width: dx, depth: dy };
  });
}

export function ballAtFrame(samples: BallTrajectoryPoint[], frame: number) {
  return samples.find(sample => sample.frame_index === Math.floor(frame)) ?? null;
}

export function buildCoachEvidence(run: AnalysisRun, telemetry: NormalizedTelemetry, range: FrameRange) {
  return {
    runId: run.summary.id,
    fps: run.summary.fps,
    range: { startFrame: range[0], endFrame: range[1], durationSeconds: (range[1] - range[0] + 1) / run.summary.fps },
    geometryValid: run.court?.isValid === true,
    players: (["player1", "player2"] as const).map((filter, index) => ({ id: index + 1, ...spatialStats(samplePoints(telemetry, filter, range)) })),
    events: run.events.filter(event => event.frame >= range[0] && event.frame <= range[1]).slice(0, 100).map(event => ({ id: event.event_id, type: event.event_type, frame: event.frame, confidence: event.confidence, playerId: event.player_id ?? null })),
    limitations: ["Single-camera court-plane projection; no measured ball height or spin.", "Player position samples describe occupancy, not point-winning causality.", "No reliable rally outcome supplied. Event labels are model estimates."],
  };
}
