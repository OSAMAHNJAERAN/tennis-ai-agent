import type { AnalysisRun, LandingZone } from '../data/contracts';
import { snapshot } from '../data/repository';

/** Explicitly synthetic reviewed-contact fixture; never a published example. */
export function landingFixture(): AnalysisRun {
  const base = snapshot.details.phase5_scoring_1!;
  const matrix = [[.1, 0, 0], [0, .1, 0], [0, 0, 1]];
  const evidence = { active_play: true, position_type: 'ground_contact', review_status: 'confirmed',
    ground_contact_source: 'synthetic_software_fixture', position_uncertainty_m: .1,
    analysis_id: base.summary.id, analysis_revision: 2, calibration_id: 'frame_50', contact_position_px: [20, 140] };
  const event = { event_id: 5, event_type: 'BOUNCE', frame: 50, timestamp_s: 2, confidence: 1,
    court_position_m: [2, 14] as [number, number], trajectory_state: 'DETECTED' as const, evidence };
  const row = (partition: LandingZone['partition'], id: string, label: string, polygon: [number, number][]): LandingZone => ({
    schema_version: '1.0', analysis_id: base.summary.id, analysis_revision: 2, calibration_id: 'frame_50',
    event_id: 5, timestamp_s: 2, duration_s: .8, partition, status: 'inside', label,
    contact_position_m: [2, 14], contact_position_px: [20, 140], position_uncertainty_m: .1,
    source: 'synthetic_software_fixture', review_status: 'confirmed', line_call: null,
    zones: [{ id, label, polygon_m: polygon, polygon_px: polygon.map(([x, y]) => [x * 10, y * 10]) }],
  });
  return { ...base, analysisRevision: 2, summary: { ...base.summary, frames: 100, fps: 25, durationSeconds: 4 },
    events: [event], court: { ...base.court!, isValid: true, homographyMatrix: matrix },
    landingZones: [row('tactical_3x3', 'near_r0_c0', 'Near tactical row 1, column 1',
      [[1.37, 11.885], [1.37 + 8.23 / 3, 11.885], [1.37 + 8.23 / 3, 11.885 + 11.885 / 3], [1.37, 11.885 + 11.885 / 3]]),
    row('service', 'near_service_left', 'Near left service box', [[1.37, 11.885], [5.485, 11.885], [5.485, 18.285], [1.37, 18.285]])],
    cameraTracking: { enabled: true, coordinateSemantics: 'synthetic court', accuracyValidated: false,
      frames: Array.from({ length: 100 }, (_, index) => ({ frame_index: index, timestamp_seconds: index / 25,
        isValid: true, imageToCourt: matrix, anchorToFramePx: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        rejectionReason: null, registrationResidualReferencePx: null, referenceWidth: null, accuracyValidated: false })) },
    courtSegments: [{ startFrame: 0, endFrameExclusive: 100, startSeconds: 0, endSeconds: 4,
      calibrationStatus: 'VALID_COURT_PLANE', accuracyValidated: false }],
  };
}
