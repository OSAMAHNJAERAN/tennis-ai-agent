import { describe, expect, it } from "vitest";
import { AnalysisRunSchema, LineDecisionSchema, SnapshotSchema, TrackingStateSchema } from "./contracts";
import { analysisRepository, RunNotFoundError, snapshot } from "./repository";

const expectedRunIds = [
  "baseline_run_1",
  "phase2_ball_1",
  "phase2_yolo11_final",
  "phase3_events_1",
  "phase4_1_line_calls",
  "phase4_line_calls_1",
  "phase5_scoring_1",
];

describe("offline analysis repository", () => {
  it("validates the versioned snapshot", () => {
    expect(SnapshotSchema.safeParse(snapshot).success).toBe(true);
  });

  it.each(expectedRunIds)("normalizes %s to the current run contract", async (id) => {
    const run = await analysisRepository.getRun(id);
    expect(AnalysisRunSchema.safeParse(run).success).toBe(true);
    expect(run.summary.id).toBe(id);
    expect(run.summary.frames).toBe(214);
  });

  it("preserves every required scientific enum", () => {
    for (const state of ["DETECTED", "TRACKED", "PREDICTED", "INTERPOLATED", "OCCLUDED", "MISSING"]) {
      expect(TrackingStateSchema.parse(state)).toBe(state);
    }
    for (const decision of ["IN", "OUT", "SERVE_IN", "SERVE_FAULT", "REVIEW_REQUIRED", "UNKNOWN"]) {
      expect(LineDecisionSchema.parse(decision)).toBe(decision);
    }
  });

  it("keeps unavailable historical artifacts explicit", async () => {
    const baseline = await analysisRepository.getRun("baseline_run_1");
    expect(baseline.ballMetrics).toBeNull();
    expect(baseline.events).toEqual([]);
    expect(baseline.lineCalls).toEqual([]);
    expect(analysisRepository.getVideoUrl("baseline_run_1")).toBeNull();
  });

  it("rejects invalid run ids", async () => {
    await expect(analysisRepository.getRun("not-a-real-run")).rejects.toBeInstanceOf(RunNotFoundError);
  });
});

