import { afterEach, describe, expect, it, vi } from 'vitest';
import { LiveAnalysisRepository, parseLiveRun, snapshot } from './repository';
import { AnalysisRunSchema } from './contracts';

afterEach(() => vi.unstubAllGlobals());
describe('live analysis repository', () => {
  it('preserves immutable generation metadata and all supplied media and export pins', () => {
    const base = snapshot.details[snapshot.defaultRunId]!;
    const id = 'match_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    const generation = '1a'.repeat(32);
    const videoUrl = `/api/analyses/${id}/media/source?generation=${generation}`;
    const annotatedVideoUrl = `/api/analyses/${id}/media/annotated?generation=${generation}`;
    const exports = { reportJson: `/api/analyses/${id}/download/report?generation=${generation}`,
      csv: null, annotatedVideo: `/api/analyses/${id}/download/annotated?generation=${generation}` };
    const parsed = parseLiveRun({ ...base, summary: { ...base.summary, id },
      analysisGeneration: generation, analysisRevision: 0, videoUrl, annotatedVideoUrl, exports });
    expect(parsed.analysisGeneration).toBe(generation);
    expect(parsed.analysisRevision).toBe(0);
    expect(parsed.videoUrl).toBe(videoUrl);
    expect(parsed.annotatedVideoUrl).toBe(annotatedVideoUrl);
    expect(parsed.exports).toEqual(exports);
  });

  it.each([null, '', 'a'.repeat(63), 'a'.repeat(65), 'A'.repeat(64), 'g'.repeat(64)])(
    'rejects malformed immutable generation metadata %s', analysisGeneration => {
      const base = snapshot.details[snapshot.defaultRunId]!;
      expect(AnalysisRunSchema.safeParse({ ...base, analysisGeneration }).success).toBe(false);
      expect(() => parseLiveRun({ ...base, analysisGeneration })).toThrow('invalid metadata');
    },
  );

  it.each([0, 3])('preserves an explicitly supplied nonnegative analysis revision %s', analysisRevision => {
    const base = snapshot.details[snapshot.defaultRunId]!;
    const value = { ...base, analysisGeneration: '1a'.repeat(32), analysisRevision };
    expect(AnalysisRunSchema.parse(value).analysisRevision).toBe(analysisRevision);
    expect(parseLiveRun(value).analysisRevision).toBe(analysisRevision);
  });

  it.each([-1, .5, true, false, null, '0'])(
    'rejects malformed analysis revision metadata %s', analysisRevision => {
      const base = snapshot.details[snapshot.defaultRunId]!;
      const value = { ...base, analysisGeneration: '1a'.repeat(32), analysisRevision };
      expect(AnalysisRunSchema.safeParse(value).success).toBe(false);
      expect(() => parseLiveRun(value)).toThrow('invalid metadata');
    },
  );

  it('keeps legacy and static results available without a generation', () => {
    const base = snapshot.details[snapshot.defaultRunId]!;
    const parsed = parseLiveRun(base);
    expect(parsed.analysisGeneration).toBeUndefined();
    expect(parsed.analysisRevision).toBeUndefined();
    expect(parsed.videoUrl).toBe(base.videoUrl);
    expect(parsed.summary).toEqual(base.summary);
  });

  it('loads the requested analysis and never substitutes a demo on failure', async () => {
    const id = 'match_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    const run = snapshot.details[snapshot.defaultRunId]!;
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ...run, summary: { ...run.summary, id } }) });
    vi.stubGlobal('fetch', fetch);
    expect((await new LiveAnalysisRepository().getRun(id)).summary.id).toBe(id);
    expect(fetch).toHaveBeenCalledWith(`/api/analyses/${id}/result`, { signal: undefined });
    fetch.mockResolvedValue({ ok: false, json: async () => ({ detail: 'Analysis is still processing.' }) });
    await expect(new LiveAnalysisRepository().getRun(id)).rejects.toThrow('still processing');
  });

  it('isolates malformed optional sections and preserves the video', () => {
    const run = snapshot.details[snapshot.defaultRunId]!;
    const parsed = parseLiveRun({ ...run, playerMetrics: { player_1: { total_distance_m: -1 } }, events: [{ confidence: 4 }] });
    expect(parsed.playerMetrics).toBeNull();
    expect(parsed.events).toEqual([]);
    expect(parsed.videoUrl).toBe(run.videoUrl);
    expect(parsed.warnings).toHaveLength(2);
  });

  it('rejects a valid result that belongs to a different session', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => snapshot.details[snapshot.defaultRunId] }));
    await expect(new LiveAnalysisRepository().getRun('match_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')).rejects.toThrow('different analysis');
  });

  it('rejects invalid video metadata rather than inventing a time base', () => {
    const run = snapshot.details[snapshot.defaultRunId]!;
    expect(() => parseLiveRun({ ...run, summary: { ...run.summary, fps: 0 } })).toThrow('invalid metadata');
  });

  it('keeps replay available when optional frame evidence is malformed', () => {
    const run = snapshot.details[snapshot.defaultRunId]!;
    const parsed = parseLiveRun({ ...run, poseTracking: { frames: 'invalid' }, racketTracking: { frames: null }, cameraTracking: { enabled: true } });
    expect(parsed.videoUrl).toBe(run.videoUrl);
    expect(parsed.detections).toEqual(run.detections);
    expect(parsed.poseTracking).toBeNull();
    expect(parsed.racketTracking).toBeNull();
    expect(parsed.cameraTracking).toBeNull();
    expect(parsed.warnings).toHaveLength(3);
  });

  it('preserves generic person observations independently of court calibration and active-player roles', () => {
    const run = snapshot.details[snapshot.defaultRunId]!;
    const imagePeople = {
      coordinateSystem: 'normalized_video_pixels', identityScope: 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID',
      participantSemantics: 'May include spectators and officials.', accuracyValidated: false,
      missingSemantics: 'Missing observations remain null.',
      frames: [
        { frame_index: 0, timestamp_seconds: 0, people: [{ bbox: [1, 2, 10, 20], confidence: .8, sourceTrackId: null, courtRole: null, participantStatus: 'UNKNOWN' }] },
        { frame_index: 1, timestamp_seconds: 1 / 30, people: [] },
        { frame_index: 2, timestamp_seconds: 2 / 30, people: null },
      ],
    };
    const input = { ...run, summary: { ...run.summary, frames: 3, fps: 30 }, court: null, playerTracking: null,
      detections: { ...run.detections, metadata: { width: 100, height: 100 } }, imagePeople };
    const parsed = parseLiveRun(input);
    expect(parsed.imagePeople?.frames[0]?.people?.[0]?.sourceTrackId).toBeNull();
    expect(parsed.imagePeople?.frames[0]?.people?.[0]?.participantStatus).toBe('UNKNOWN');
    expect(parsed.imagePeople?.frames[1]?.people).toEqual([]);
    expect(parsed.imagePeople?.frames[2]?.people).toBeNull();
    expect(parsed.court).toBeNull();
    expect(parsed.playerTracking).toBeNull();
    expect(parsed.videoUrl).toBe(run.videoUrl);
    expect(parseLiveRun({ ...input, imagePeople: { ...imagePeople, frames: [{ ...imagePeople.frames[0], timestamp_seconds: 5 }, ...imagePeople.frames.slice(1)] } }).imagePeople).toBeNull();
    expect(parseLiveRun({ ...input, imagePeople: { ...imagePeople, identityScope: 'VERIFIED_PERSON_ID' } }).imagePeople).toBeNull();
    expect(parseLiveRun({ ...input, imagePeople: { ...imagePeople, frames: [{ ...imagePeople.frames[0], people: [{ ...imagePeople.frames[0]!.people![0], bbox: [1, 2, 101, 20] }] }, ...imagePeople.frames.slice(1)] } }).imagePeople).toBeNull();
  });

  it('retains one player\'s valid movement when the other player has no measurements', () => {
    const run = snapshot.details[snapshot.defaultRunId]!;
    const player = {
      role: 'near', name: null, identityScope: 'NEAR_FAR_ROLE_WITHIN_CAMERA_SEGMENT',
      metrics: { total_distance_m: 2.5, avg_speed_kmh: 9, max_speed_kmh: 12, coverage_pct: 50 },
      accuracyValidated: false, groundPositionSource: 'CALIBRATED_BBOX_BOTTOM_CENTER_PROXY',
      positionIntervals: [{ startFrame: 0, endFrameExclusive: 30, startSeconds: 0, endSeconds: 1 }],
      speedIntervals: [], uncertaintySemantics: 'Independent speed accuracy is unqualified.',
      samples: [{ frame_index: 0, timestamp_seconds: 0, ground_position_m: [5, 24], speed_kmh: null, position_fit_rmse_m: null }],
    };
    const parsed = parseLiveRun({ ...run, playerMetrics: null, playerTracking: { player_1: player, player_2: null } });
    expect(parsed.playerTracking?.player_1?.metrics.total_distance_m).toBe(2.5);
    expect(parsed.playerTracking?.player_1?.samples[0]?.speed_kmh).toBeNull();
    expect(parsed.playerTracking?.player_2).toBeNull();
    expect(parsed.playerMetrics).toBeNull();
  });

  it('accepts ISO timestamps from the Python API including an explicit UTC offset', () => {
    const run = snapshot.details[snapshot.defaultRunId]!;
    expect(parseLiveRun({ ...run, summary: { ...run.summary, analyzedAt: '2026-09-19T10:32:18.711126+00:00' } }).summary.fps).toBe(30);
  });

  it('loads only lightweight run summaries for the session picker', async () => {
    const run = snapshot.runs[0]!;
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => [{ ...run, id: 'match_new' }] });
    vi.stubGlobal('fetch', fetch);
    expect((await new LiveAnalysisRepository().listRuns())[0]?.id).toBe('match_new');
    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch.mock.calls[0]?.[0]).toBe('/api/runs');
  });
});
