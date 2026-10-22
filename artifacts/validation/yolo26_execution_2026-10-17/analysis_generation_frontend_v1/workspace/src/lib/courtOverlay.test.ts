import { describe, expect, it } from 'vitest';
import type { AnalysisRun } from '../data/contracts';
import { snapshot } from '../data/repository';
import { courtOverlayPoints } from './courtOverlay';

const base = snapshot.details.phase5_scoring_1!;
const run = { ...base, court: { ...base.court!, isValid: true, courtKeypointsPx: [[10, 20], [30, 40]] } } as AnalysisRun;
const cameraSample = { frame_index: 1, timestamp_seconds: 1 / run.summary.fps, isValid: true,
  imageToCourt: null, anchorToFramePx: [[1, 0, 50], [0, 1, 12], [0, 0, 1]],
  rejectionReason: null, registrationResidualReferencePx: null, referenceWidth: null, accuracyValidated: false as const };

describe('court image overlay', () => {
  it('follows registered camera motion in the presented frame', () => {
    const moved = { ...run, cameraTracking: { enabled: true, coordinateSemantics: 'court plane',
      accuracyValidated: false as const, frames: [cameraSample, cameraSample] } };
    expect(courtOverlayPoints(moved, 1)).toEqual([[60, 32], [80, 52]]);
  });
  it('withholds invalid, missing or mismatched camera intervals', () => {
    const lost = { ...run, cameraTracking: { enabled: true, coordinateSemantics: 'court plane',
      accuracyValidated: false as const, frames: [{ ...cameraSample, isValid: false }] } };
    expect(courtOverlayPoints(lost, 0)).toEqual([]);
    expect(courtOverlayPoints(lost, 2)).toEqual([]);
    expect(courtOverlayPoints({ ...lost, cameraTracking: { ...lost.cameraTracking, frames: [cameraSample] } }, 0)).toEqual([]);
    expect(courtOverlayPoints({ ...run, court: { ...run.court!, isValid: false } }, 0)).toEqual([]);
  });
  it('rejects projective horizons and malformed transforms', () => {
    const camera = { enabled: true, coordinateSemantics: 'court plane', accuracyValidated: false as const,
      frames: [{ ...cameraSample, frame_index: 0, anchorToFramePx: [[1, 0, 0], [0, 1, 0], [0, 0, 0]] }] };
    expect(courtOverlayPoints({ ...run, cameraTracking: camera }, 0)).toEqual([]);
    expect(courtOverlayPoints({ ...run, cameraTracking: { ...camera, frames: [{ ...camera.frames[0]!, anchorToFramePx: [[1]] }] } }, 0)).toEqual([]);
  });
});
