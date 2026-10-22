import { describe, expect, it } from 'vitest';
import { parseLiveRun } from '../data/repository';
import { landingFixture } from '../test/landingFixture';
import { createLandingZoneIndex, projectCourtPolygon } from './landingZones';

describe('reviewed landing zones', () => {
  it('shows complete selected cells after contact for the bounded source-time interval', () => {
    const at = createLandingZoneIndex(landingFixture());
    expect(at(1.999, 49, 'tactical_3x3')).toEqual([]);
    const row = at(2.4, 60, 'tactical_3x3')[0]!;
    expect(row.zones[0]!.polygon_m).toHaveLength(4);
    expect(row.contact_position_px[0]).toBeCloseTo(20, 10);
    expect(row.contact_position_px[1]).toBeCloseTo(140, 10);
    expect(at(2.8, 70, 'tactical_3x3')).toEqual([]);
    expect(at(2.4, 60, 'service')[0]!.zones[0]!.id).toBe('near_service_left');
    expect(at(2.4, 60, 'tactical_4x3')).toEqual([]);
  });
  it.each(['revision', 'airborne', 'dead', 'unreviewed', 'source', 'position', 'calibration'])(
    'withholds conflicting %s evidence', change => {
      const run = landingFixture();
      if (change === 'revision') run.analysisRevision = 3;
      if (change === 'airborne') run.events[0]!.evidence!.position_type = 'airborne_projection';
      if (change === 'dead') run.events[0]!.evidence!.active_play = false;
      if (change === 'unreviewed') run.events[0]!.evidence!.review_status = 'provisional';
      if (change === 'source') run.events[0]!.evidence!.ground_contact_source = 'different_source';
      if (change === 'position') run.events[0]!.court_position_m = [9, 14];
      if (change === 'calibration') run.cameraTracking!.frames[60]!.isValid = false;
      expect(createLandingZoneIndex(run)(2.4, 60, 'tactical_3x3')).toEqual([]);
    });
  it('reprojects the same metric cell and contact during camera motion', () => {
    const run = landingFixture();
    run.cameraTracking!.frames[60]!.imageToCourt = [[.05, 0, 0], [0, .05, 0], [0, 0, 1]];
    const row = createLandingZoneIndex(run)(2.4, 60, 'tactical_3x3')[0]!;
    expect(row.contact_position_px[0]).toBeCloseTo(40, 10);
    expect(row.contact_position_px[1]).toBeCloseTo(280, 10);
    expect(row.zones[0]!.polygon_px[0]![0]).toBeCloseTo(27.4, 10);
    expect(row.zones[0]!.polygon_px[0]![1]).toBeCloseTo(237.7, 10);
    expect(row.zones[0]!.polygon_m).toEqual(run.landingZones![0]!.zones[0]!.polygon_m);
  });
  it('ends an old highlight when calibration is lost and later recovers', () => {
    const run = landingFixture();
    run.courtSegments![0]!.endFrameExclusive = 55;
    run.courtSegments!.push({ ...run.courtSegments![0]!, startFrame: 56, endFrameExclusive: 100 });
    expect(createLandingZoneIndex(run)(2.4, 60, 'tactical_3x3')).toEqual([]);
  });
  it('rejects singular matrices and polygons crossing a projective horizon', () => {
    expect(projectCourtPolygon([[1, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]])).toBeNull();
    expect(projectCourtPolygon([[1, 1], [-1, 1]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]])).toBeNull();
  });
  it('keeps usable video when an optional zone artifact is malformed', () => {
    const run = landingFixture();
    const parsed = parseLiveRun({ ...run, landingZones: [{ ...run.landingZones![0]!, zones: [{ polygon_px: [[1, 2]] }] }] });
    expect(parsed.videoUrl).toBe(run.videoUrl);
    expect(parsed.landingZones).toEqual([]);
    expect(parsed.warnings).toContain('Analysis unavailable for this section: landingZones.');
  });
});
