import { z } from "zod";
import { NativeForecastSchema } from './nativeForecast';
import { MediaSourceSchema } from './publicDemos';

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
  pose: z.boolean().optional(),
  racket: z.boolean().optional(),
  masks: z.boolean().optional(),
  world3d: z.boolean().optional(),
  report: z.boolean().optional(),
  csv: z.boolean().optional(),
  imagePeople: z.boolean().optional(),
});

export const AnalysisRunSummarySchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  phase: z.string().min(1),
  status: z.enum(["COMPLETED", "PARTIAL", "FAILED"]),
  analyzedAt: z.string().datetime({ offset: true }),
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
  bounce_position_px: Point2DSchema.nullable(),
  bounce_position_m: Point2DSchema.nullable(),
  tracker_state: z.union([TrackingStateSchema, z.literal("UNKNOWN")]),
  center_signed_distance_cm: z.number().nullable(),
  contact_patch_model: z.string(),
  contact_patch_radius_cm: z.number().nullable(),
  ball_edge_margin_cm: z.number().nullable(),
  position_uncertainty_cm: z.number().nullable(),
  spatial_tier: z.string(),
  confidence: z.number().min(0).max(1),
  reason: z.string(),
  refinement_method: z.string(),
  line_strip_info: z.record(z.string(), z.unknown()).nullable().optional(),
  reason_code: z.string().nullable().optional(),
});

export const PlayerMetricsSchema = z.object({
  player_1: z.object({
    total_distance_m: z.number().nonnegative(),
    avg_speed_kmh: z.number().nonnegative(),
    coverage_pct: z.number().min(0).max(100),
    max_speed_kmh: z.number().nonnegative().nullable().optional(),
  }),
  player_2: z.object({
    total_distance_m: z.number().nonnegative(),
    avg_speed_kmh: z.number().nonnegative(),
    coverage_pct: z.number().min(0).max(100),
    max_speed_kmh: z.number().nonnegative().nullable().optional(),
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
  court_registration_valid: z.boolean().optional(),
  player_1: z.object({ bbox: z.array(z.number()).length(4), confidence: z.number().min(0).max(1).nullable().optional() }).nullable().optional(),
  player_2: z.object({ bbox: z.array(z.number()).length(4), confidence: z.number().min(0).max(1).nullable().optional() }).nullable().optional(),
  ball: z
    .object({
      position_px: Point2DSchema.nullable(),
      confidence: z.number().nullable(),
      state: TrackingStateSchema,
    })
    .nullable()
    .optional(),
});

export const PoseJointSchema = z.object({
  name: z.string(),
  x_px: z.number().finite().nullable(),
  y_px: z.number().finite().nullable(),
  confidence: z.number().min(0).max(1).nullable(),
  visible: z.boolean(),
});

export const PoseTrackingSchema = z.object({
  coordinateSystem: z.literal('normalized_video_pixels'),
  jointNames: z.array(z.string()),
  boneTopology: z.array(z.tuple([z.number().int().nonnegative(), z.number().int().nonnegative()])),
  frames: z.array(z.object({
    frame_index: z.number().int().nonnegative(),
    timestamp_seconds: z.number().nonnegative(),
    player_1: z.array(PoseJointSchema).nullable(),
    player_2: z.array(PoseJointSchema).nullable(),
  })),
  coveragePct: z.object({ player_1: z.number().min(0).max(100), player_2: z.number().min(0).max(100) }),
  accuracyValidated: z.literal(false),
});

const RacketObservationSchema = z.object({
  state: TrackingStateSchema,
  bbox_xyxy: z.tuple([z.number(), z.number(), z.number(), z.number()]).nullable(),
  position_px: Point2DSchema.nullable(),
  confidence: z.number().min(0).max(1).nullable(),
  velocity_px_per_s: Point2DSchema.nullable(),
  motion_direction_deg: z.number().nullable(),
  source: z.string().nullable(),
  ownership: z.literal('PLAYER_CROP_ASSOCIATION'),
  contact_event: z.null(),
});

export const RacketTrackingSchema = z.object({
  coordinateSystem: z.literal('normalized_video_pixels'),
  frames: z.array(z.object({
    frame_index: z.number().int().nonnegative(),
    timestamp_seconds: z.number().nonnegative(),
    player_1: RacketObservationSchema.nullable(),
    player_2: RacketObservationSchema.nullable(),
  })),
  coveragePct: z.object({ player_1: z.number().min(0).max(100), player_2: z.number().min(0).max(100) }),
  accuracyValidated: z.literal(false),
});

const ImagePersonSchema = z.object({
  bbox: z.tuple([z.number().finite(), z.number().finite(), z.number().finite(), z.number().finite()])
    .refine(([x1, y1, x2, y2]) => 0 <= x1 && x1 < x2 && 0 <= y1 && y1 < y2),
  confidence: z.number().positive().max(1),
  sourceTrackId: z.number().int().nonnegative().nullable(),
  courtRole: z.union([z.literal(1), z.literal(2)]).nullable(),
  participantStatus: z.enum(['UNKNOWN', 'COURT_ROLE_CANDIDATE']),
}).refine(person => person.participantStatus === (person.courtRole == null ? 'UNKNOWN' : 'COURT_ROLE_CANDIDATE'));

export const ImagePeopleSchema = z.object({
  coordinateSystem: z.literal('normalized_video_pixels'),
  identityScope: z.literal('DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID'),
  participantSemantics: z.string(),
  accuracyValidated: z.literal(false),
  missingSemantics: z.string(),
  frames: z.array(z.object({
    frame_index: z.number().int().nonnegative(),
    timestamp_seconds: z.number().nonnegative(),
    people: z.array(ImagePersonSchema).nullable(),
  })),
});

const TrackingIntervalSchema = z.object({
  startFrame: z.number().int().nonnegative(),
  endFrameExclusive: z.number().int().nonnegative(),
  startSeconds: z.number().nonnegative(),
  endSeconds: z.number().nonnegative(),
});

const PlayerTrackSchema = z.object({
  role: z.enum(['near', 'far']),
  name: z.string().nullable(),
  identityScope: z.string(),
  metrics: z.object({
    total_distance_m: z.number().nonnegative().nullable(),
    avg_speed_kmh: z.number().nonnegative().nullable(),
    max_speed_kmh: z.number().nonnegative().nullable(),
    coverage_pct: z.number().min(0).max(100),
  }),
  accuracyValidated: z.literal(false),
  summaryScopePreserved: z.boolean().optional(),
  groundPositionSource: z.string(),
  positionIntervals: z.array(TrackingIntervalSchema),
  speedIntervals: z.array(TrackingIntervalSchema),
  uncertaintySemantics: z.string(),
  samples: z.array(z.object({
    frame_index: z.number().int().nonnegative(),
    timestamp_seconds: z.number().nonnegative(),
    ground_position_m: Point2DSchema.nullable(),
    speed_kmh: z.number().nonnegative().nullable(),
    position_fit_rmse_m: z.number().nonnegative().nullable(),
  })),
});

const PlayerTrackingSchema = z.object({ player_1: PlayerTrackSchema.nullable(), player_2: PlayerTrackSchema.nullable() });

const CourtSegmentSchema = TrackingIntervalSchema.extend({
  calibrationStatus: z.enum(['VALID_COURT_PLANE', 'UNAVAILABLE_COURT_PLANE']),
  accuracyValidated: z.literal(false),
});

const CameraTrackingSchema = z.object({
  enabled: z.boolean(),
  coordinateSemantics: z.string(),
  accuracyValidated: z.literal(false),
  frames: z.array(z.object({
    frame_index: z.number().int().nonnegative(),
    timestamp_seconds: z.number().nonnegative(),
    isValid: z.boolean(),
    imageToCourt: z.array(z.array(z.number())).nullable(),
    anchorToFramePx: z.array(z.array(z.number())).nullable(),
    rejectionReason: z.string().nullable(),
    registrationResidualReferencePx: z.number().nonnegative().nullable(),
    referenceWidth: z.number().positive().nullable(),
    accuracyValidated: z.literal(false),
  })),
});

export const LandingZoneSchema = z.object({
  schema_version: z.literal('1.0'),
  analysis_id: z.string().min(1),
  analysis_revision: z.number().int().nonnegative(),
  calibration_id: z.string().min(1),
  event_id: z.number().int(),
  timestamp_s: z.number().finite().nonnegative(),
  duration_s: z.number().min(.6).max(1),
  partition: z.enum(['service', 'tactical_3x3', 'tactical_4x3']),
  status: z.enum(['inside', 'uncertain', 'outside_partition']),
  label: z.string(),
  contact_position_m: z.tuple([z.number().finite(), z.number().finite()]),
  contact_position_px: z.tuple([z.number().finite(), z.number().finite()]),
  position_uncertainty_m: z.number().finite().nonnegative(),
  source: z.string().min(1),
  review_status: z.enum(['confirmed', 'corrected']),
  line_call: z.enum(['in', 'out', 'uncertain']).nullable(),
  zones: z.array(z.object({
    id: z.string(), label: z.string(),
    polygon_m: z.array(z.tuple([z.number().finite(), z.number().finite()])).length(4),
    polygon_px: z.array(z.tuple([z.number().finite(), z.number().finite()])).length(4),
  })),
});
export type LandingZone = z.infer<typeof LandingZoneSchema>;

const PointParticipantSchema = z.union([z.literal(1), z.literal(2)]);
const PointCounterSchema = z.number().int().nonnegative();
const PointDurationSchema = z.number().finite().nonnegative().nullable();

const PointServeStatisticsSchema = z.object({
  serverId: PointParticipantSchema,
  contactExecutions: PointCounterSchema,
  serviceLets: PointCounterSchema,
  firstServesAttempted: PointCounterSchema,
  secondServesAttempted: PointCounterSchema,
  firstServesIn: PointCounterSchema,
  firstServesFaulted: PointCounterSchema,
  secondServesIn: PointCounterSchema,
  doubleFaults: PointCounterSchema,
  unresolvedExecutions: PointCounterSchema,
  firstServePercentage: z.number().finite().min(0).max(100).nullable(),
  firstServePercentageDenominator: PointCounterSchema,
});

// Native point evidence is provisional independently of whether rule processing
// was enabled. A flag cannot qualify the underlying observations or score.
export const PointAnalysisSchema = z.object({
  schemaVersion: z.literal('1.0'),
  status: z.literal('UNQUALIFIED'),
  analysisId: z.string().min(1),
  analysisRevision: PointCounterSchema,
  scoringRevision: PointCounterSchema,
  sourceSha256: z.string().regex(/^[a-f0-9]{64}$/i),
  independentAccuracyValidated: z.literal(false),
  authoritativeEventsEnabled: z.boolean(),
  summary: z.object({
    totalPointsPlayed: PointCounterSchema.nullable(),
    partialOrReviewPoints: PointCounterSchema,
    totalRallies: PointCounterSchema,
    averageRallyLength: z.number().finite().nonnegative().nullable(),
    longestRallyLength: PointCounterSchema.nullable(),
    effectivePlaySeconds: PointDurationSchema,
    timedCompletedRallies: PointCounterSchema,
    completedRalliesDurationDenominator: PointCounterSchema,
    scoreSummary: z.string().nullable(),
    matchComplete: z.boolean().nullable(),
    winnerId: PointParticipantSchema.nullable(),
  }),
  serveByPlayer: z.object({ player_1: PointServeStatisticsSchema, player_2: PointServeStatisticsSchema }),
  points: z.array(z.object({
    id: z.string().min(1), status: z.string(),
    anchorEventId: z.number().int().nullable(), terminalEventId: z.number().int().nullable(),
    serverId: PointParticipantSchema.nullable(), receiverId: PointParticipantSchema.nullable(),
    setNumber: PointCounterSchema.nullable(), gameNumber: PointCounterSchema.nullable(),
    serviceSide: z.string().nullable(), winnerId: PointParticipantSchema.nullable(),
    endingReason: z.string().nullable(), isReviewPending: z.boolean(),
    totalStrokesIncludingServe: PointCounterSchema, rallyHitsExcludingServe: PointCounterSchema,
    durationSeconds: PointDurationSchema,
  })),
  rallies: z.array(z.object({
    id: z.string().min(1), pointId: z.string().min(1), status: z.string(),
    serverId: PointParticipantSchema.nullable(), receiverId: PointParticipantSchema.nullable(),
    serveAttempt: PointParticipantSchema.nullable(), winnerId: PointParticipantSchema.nullable(),
    endingReason: z.string().nullable(), startFrame: PointCounterSchema.nullable(),
    terminalEventId: z.number().int().nullable(), durationSeconds: PointDurationSchema,
    durationReason: z.string().nullable(), totalStrokesIncludingServe: PointCounterSchema,
    rallyHitsExcludingServe: PointCounterSchema,
  })),
  serveExecutions: z.array(z.object({
    id: z.string().min(1), pointId: z.string().min(1), status: z.string(),
    officialAttempt: PointParticipantSchema, replayIndex: PointCounterSchema,
    contactEventId: z.number().int(), resultEventId: z.number().int().nullable(),
    serverId: PointParticipantSchema, serverEnd: z.string(), serviceSide: z.string(),
    rallyId: z.string().nullable(),
  })),
});
export type PointAnalysis = z.infer<typeof PointAnalysisSchema>;

export const AnalysisRunSchema = z.object({
  contractVersion: z.literal('2.0').optional(),
  summary: AnalysisRunSummarySchema,
  detections: z
    .object({ metadata: z.record(z.string(), z.unknown()).nullable(), frames: z.array(DetectionFrameSchema) })
    .nullable(),
  trajectories: z
    .object({
      ballTrajectory: z.array(BallTrajectoryPointSchema),
      player1CourtPositions: z.array(Point2DSchema.nullable()),
      player2CourtPositions: z.array(Point2DSchema.nullable()),
      player1Speeds: z.array(z.number().nonnegative().nullable()).optional(),
      player2Speeds: z.array(z.number().nonnegative().nullable()).optional(),
    })
    .nullable(),
  court: CourtGeometrySchema.nullable(),
  playerMetrics: PlayerMetricsSchema.nullable(),
  ballMetrics: BallMetricsSchema.nullable(),
  events: z.array(MatchEventSchema),
  landingZones: z.array(LandingZoneSchema).optional(),
  analysisRevision: z.number().int().nonnegative().optional(),
  analysisGeneration: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  lineCalls: z.array(LineCallSchema),
  pipelineMetrics: PipelineMetricsSchema.nullable(),
  matchState: MatchStateSchema.nullable(),
  scoringEvents: z.array(z.unknown()),
  scoreHistory: z.array(z.unknown()),
  runConfig: z.record(z.string(), z.unknown()).nullable(),
  videoUrl: z.string().nullable(),
  annotatedVideoUrl: z.string().nullable().optional(),
  warnings: z.array(z.string()).optional(),
  poseTracking: PoseTrackingSchema.nullable().optional(),
  racketTracking: RacketTrackingSchema.nullable().optional(),
  imagePeople: ImagePeopleSchema.nullable().optional(),
  playerTracking: PlayerTrackingSchema.nullable().optional(),
  courtSegments: z.array(CourtSegmentSchema).nullable().optional(),
  cameraTracking: CameraTrackingSchema.nullable().optional(),
  mediaMetadata: z.record(z.string(), z.unknown()).optional(),
  provenance: z.record(z.string(), z.unknown()).optional(),
  coordinateSemantics: z.record(z.string(), z.unknown()).optional(),
  unavailableSections: z.record(z.string(), z.string()).optional(),
  exports: z.record(z.string(), z.string().nullable()).optional(),
  mediaSource: MediaSourceSchema.nullable().optional(),
  pointAnalysis: PointAnalysisSchema.nullable().optional(),
  nativeForecast: NativeForecastSchema.nullable().optional(),
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
