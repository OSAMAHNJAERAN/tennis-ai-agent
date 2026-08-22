import { z } from "zod";

export const TrackingStateSchema = z.enum([
  "DETECTED",
  "TRACKED",
  "PREDICTED",
  "INTERPOLATED",
  "OCCLUDED",
  "MISSING",
]);

export const LineDecisionSchema = z.enum([
  "IN",
  "OUT",
  "SERVE_IN",
  "SERVE_FAULT",
  "REVIEW_REQUIRED",
  "UNKNOWN",
]);

export type TrackingState = z.infer<typeof TrackingStateSchema>;
export type LineDecision = z.infer<typeof LineDecisionSchema>;

const Point2DSchema = z.tuple([z.number(), z.number()]);

export const RunCapabilitiesSchema = z.object({
  detections: z.boolean(),
  trajectories: z.boolean(),
  court: z.boolean(),
  players: z.boolean(),
  ballMetrics: z.boolean(),
  events: z.boolean(),
  lineCalls: z.boolean(),
  scoring: z.boolean(),
  video: z.boolean(),
});

export const AnalysisRunSummarySchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  phase: z.string().min(1),
  status: z.enum(["COMPLETED", "PARTIAL", "FAILED"]),
  analyzedAt: z.string().datetime(),
  sourceVideoPath: z.string(),
  frames: z.number().int().nonnegative(),
  fps: z.number().positive(),
  durationSeconds: z.number().nonnegative(),
  eventCount: z.number().int().nonnegative(),
  lineCallCount: z.number().int().nonnegative(),
  artifacts: z.array(z.string()),
  capabilities: RunCapabilitiesSchema,
});

export const MatchEventSchema = z.object({
  event_id: z.number().int(),
  event_type: z.string(),
  frame: z.number().int().nonnegative(),
  timestamp_s: z.number().nonnegative(),
  player_id: z.number().int().nullable().optional(),
  ball_position_px: Point2DSchema.nullable().optional(),
  court_position_m: Point2DSchema.nullable().optional(),
  confidence: z.number().min(0).max(1),
  trajectory_state: TrackingStateSchema,
  evidence: z.record(z.string(), z.unknown()).optional(),
});

export const BallTrajectoryPointSchema = z.object({
  frame_index: z.number().int().nonnegative(),
  timestamp_seconds: z.number().nonnegative(),
  x_px: z.number().nullable(),
  y_px: z.number().nullable(),
  court_x_m: z.number().nullable(),
  court_y_m: z.number().nullable(),
  speed_kmh: z.number().nullable(),
  confidence: z.number().min(0).max(1).nullable(),
  state: TrackingStateSchema,
});

export const CourtGeometrySchema = z.object({
  courtKeypointsPx: z.array(Point2DSchema),
  canonicalKeypointsM: z.array(Point2DSchema),
  homographyMatrix: z.array(z.array(z.number())),
  reprojectionErrorPx: z.number(),
  isValid: z.boolean(),
});

export const LineCallSchema = z.object({
  event_id: z.number().int(),
  bounce_frame: z.number().int().nonnegative(),
  decision: LineDecisionSchema,
  decision_context: z.string(),
  nearest_line: z.string(),
  bounce_position_px: Point2DSchema,
  bounce_position_m: Point2DSchema,
  tracker_state: TrackingStateSchema,
  center_signed_distance_cm: z.number(),
  contact_patch_model: z.string(),
  contact_patch_radius_cm: z.number(),
  ball_edge_margin_cm: z.number(),
  position_uncertainty_cm: z.number(),
  spatial_tier: z.string(),
  confidence: z.number().min(0).max(1),
  reason: z.string(),
  refinement_method: z.string(),
  line_strip_info: z.record(z.string(), z.unknown()).optional(),
});

export const PlayerMetricsSchema = z.object({
  player_1: z.object({
    total_distance_m: z.number().nonnegative(),
    avg_speed_kmh: z.number().nonnegative(),
    coverage_pct: z.number().min(0).max(100),
  }),
  player_2: z.object({
    total_distance_m: z.number().nonnegative(),
    avg_speed_kmh: z.number().nonnegative(),
    coverage_pct: z.number().min(0).max(100),
  }),
});

export const FlightSegmentSchema = z.object({
  segment_id: z.number().int(),
  start_frame: z.number().int(),
  end_frame: z.number().int(),
  start_event: z.string().nullable(),
  end_event: z.string().nullable(),
  duration_s: z.number(),
  total_2d_distance_m: z.number(),
  mean_speed_kmh: z.number(),
  peak_speed_kmh: z.number(),
  launch_speed_kmh: z.number(),
  confidence_tier: z.string(),
  confidence_score: z.number(),
});

export const BallMetricsSchema = z.object({
  scientific_status: z.string(),
  speed_overview: z.object({
    coordinate_system: z.string(),
    time_base: z.string(),
    units: z.string(),
    average_speed_kmh: z.number(),
    maximum_speed_kmh: z.number(),
    segments_count: z.number().int(),
    scientific_disclaimer: z.string(),
  }),
  flight_segments: z.array(FlightSegmentSchema),
});

export const PipelineMetricsSchema = z.object({
  pipelineFps: z.number().nullable(),
  eventCount: z.number().int(),
  lineCallCount: z.number().int(),
  courtReprojectionErrorPx: z.number().nullable(),
  raw: z.record(z.string(), z.unknown()).nullable(),
});

export const MatchStateSchema = z.object({
  match_id: z.string(),
  format: z.string(),
  set_format: z.string(),
  player_1: z.record(z.string(), z.unknown()),
  player_2: z.record(z.string(), z.unknown()),
  server_id: z.number().int(),
  receiver_id: z.number().int(),
  current_set: z.number().int(),
  current_game: z.number().int(),
  serve_attempt: z.number().int(),
  service_side: z.string(),
  tie_break_active: z.boolean(),
  point_state: z.string(),
  ball_state: z.string(),
  completed_sets: z.array(z.unknown()),
  match_complete: z.boolean(),
  winner_id: z.number().int().nullable(),
  last_point_winner: z.number().int().nullable(),
  last_point_reason: z.string().nullable(),
  state_version: z.number().int(),
});

const DetectionFrameSchema = z.object({
  frame_index: z.number().int(),
  player_1: z.object({ bbox: z.array(z.number()).length(4) }).nullable().optional(),
  player_2: z.object({ bbox: z.array(z.number()).length(4) }).nullable().optional(),
  ball: z
    .object({
      position_px: Point2DSchema.nullable(),
      confidence: z.number().nullable(),
      state: TrackingStateSchema,
    })
    .nullable()
    .optional(),
});

export const AnalysisRunSchema = z.object({
  summary: AnalysisRunSummarySchema,
  detections: z
    .object({ metadata: z.record(z.string(), z.unknown()).nullable(), frames: z.array(DetectionFrameSchema) })
    .nullable(),
  trajectories: z
    .object({
      ballTrajectory: z.array(BallTrajectoryPointSchema),
      player1CourtPositions: z.array(Point2DSchema.nullable()),
      player2CourtPositions: z.array(Point2DSchema.nullable()),
    })
    .nullable(),
  court: CourtGeometrySchema.nullable(),
  playerMetrics: PlayerMetricsSchema.nullable(),
  ballMetrics: BallMetricsSchema.nullable(),
  events: z.array(MatchEventSchema),
  lineCalls: z.array(LineCallSchema),
  pipelineMetrics: PipelineMetricsSchema.nullable(),
  matchState: MatchStateSchema.nullable(),
  scoringEvents: z.array(z.unknown()),
  scoreHistory: z.array(z.unknown()),
  runConfig: z.record(z.string(), z.unknown()).nullable(),
  videoUrl: z.string().nullable(),
});

export const SnapshotSchema = z.object({
  schemaVersion: z.literal(1),
  generatedAt: z.string().datetime(),
  defaultRunId: z.string(),
  runs: z.array(AnalysisRunSummarySchema),
  details: z.record(z.string(), AnalysisRunSchema),
});

export type AnalysisRunSummary = z.infer<typeof AnalysisRunSummarySchema>;
export type AnalysisRun = z.infer<typeof AnalysisRunSchema>;
export type MatchState = z.infer<typeof MatchStateSchema>;
export type PlayerMetrics = z.infer<typeof PlayerMetricsSchema>;
export type BallMetrics = z.infer<typeof BallMetricsSchema>;
export type MatchEvent = z.infer<typeof MatchEventSchema>;
export type BallTrajectoryPoint = z.infer<typeof BallTrajectoryPointSchema>;
export type CourtGeometry = z.infer<typeof CourtGeometrySchema>;
export type LineCall = z.infer<typeof LineCallSchema>;
export type PipelineMetrics = z.infer<typeof PipelineMetricsSchema>;
export type ConfigurationMetadata = Record<string, unknown>;

