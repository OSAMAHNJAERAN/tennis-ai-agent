import { describe, expect, it } from "vitest";
import { snapshot } from "../data/repository";
import { COURT, buildCoachEvidence, courtToWorld, densityGrid, interpolatePosition, normalizeTelemetry, playerSpeed, samplePoints, spatialStats, splitTrail } from "./tennisTelemetry";

const source = snapshot.details.phase5_scoring_1!;
describe("spatial telemetry", () => {
  it("centers the regulation court in X/Z and preserves baseline run-off", () => {
    expect(courtToWorld([COURT.width / 2, COURT.length / 2])).toEqual([0, 0, 0]);
    expect(courtToWorld([2, 26])[2]).toBeGreaterThan(COURT.length / 2);
    expect(courtToWorld([0, 0], 1)).toEqual([-5.485, 1, -11.885]);
  });
  it("interpolates adjacent observations without filling missing detections or large jumps", () => {
    expect(interpolatePosition([[2, 25], [2.2, 25.2]], .5)).toEqual([2.1, 25.1]);
    expect(interpolatePosition([null, [2, 25]], .5)).toBeNull();
    expect(interpolatePosition([[2, 25], null], 1)).toBeNull();
    expect(interpolatePosition([[2, 25], [10, 25]], .5)).toEqual([2, 25]);
  });
  it("indexes sparse ball data by source frame and never displays missing or occluded samples", () => {
    const run = structuredClone(source);
    const point = run.trajectories!.ballTrajectory[0]!;
    run.trajectories!.ballTrajectory = [{ ...point, frame_index: 8 }, { ...point, frame_index: 12, state: "MISSING" }, { ...point, frame_index: 13, state: "OCCLUDED" }];
    const normalized = normalizeTelemetry(run);
    expect(normalized.ball[0]).toBeNull(); expect(normalized.ball[8]?.position).toEqual([point.court_x_m, point.court_y_m]);
    expect(normalized.ball[12]).toBeNull(); expect(normalized.ball[13]).toBeNull();
  });
  it("gates invalid court mappings and out-of-field points", () => {
    const run = structuredClone(source); run.court!.isValid = false;
    const normalized = normalizeTelemetry(run);
    expect(samplePoints(normalized, "combined", [0, 213])).toHaveLength(0);
    expect(normalized.ball.every(point => point === null)).toBe(true);
  });
  it("breaks trails at gaps and discontinuities", () => {
    expect(splitTrail([[1, 1], [1.1, 1.1], null, [2, 2], [2.1, 2.1], [10, 10]], 5, 10, 1)).toEqual([[[1, 1], [1.1, 1.1]], [[2, 2], [2.1, 2.1]]]);
  });
  it("computes time-based player speed and rejects discontinuous segments", () => {
    expect(playerSpeed([[1, 1], [1.1, 1], [1.2, 1]], 2, 30)).toBeCloseTo(10.8);
    expect(playerSpeed([[1, 1], null, [1.2, 1]], 2, 30)).toBeNull();
    expect(playerSpeed([[1, 1], [10, 10]], 1, 30)).toBeNull();
  });
  it("counts behind-baseline occupancy with explicit sample denominators", () => {
    const stats = spatialStats([[2, 26], [2, 25], [5.485, 10], [5.485, -1]]);
    expect(stats.behindBaseline).toBe(75); expect(stats.frontcourt).toBe(25); expect(stats.center).toBe(50);
    expect(spatialStats([]).total).toBe(0);
  });
  it("keeps kernel density proportional to sample count and finite for empty input", () => {
    const one = densityGrid([[2, 26]]); const two = densityGrid([[2, 26], [2, 26]]);
    expect(Math.max(...two.map(p => p.value))).toBeCloseTo(Math.max(...one.map(p => p.value)) * 2);
    expect(densityGrid([]).every(p => p.value === 0)).toBe(true);
    expect(one.some(p => p.y > COURT.length && p.value > .1)).toBe(true);
  });
  it("ranges coaching evidence and excludes source paths, raw detections, and credentials", () => {
    const telemetry = normalizeTelemetry(source);
    const evidence = buildCoachEvidence(source, telemetry, [40, 70]);
    expect(evidence.range.durationSeconds).toBeCloseTo(31 / 30);
    expect(evidence.events.every(e => e.frame >= 40 && e.frame <= 70)).toBe(true);
    expect(evidence.players[0]!.total).toBeLessThanOrEqual(31);
    expect(evidence).not.toHaveProperty("sourceVideoPath");
    expect(evidence).not.toHaveProperty("runConfig");
    expect(samplePoints(telemetry, "combined", [0, 213]).length).toBeGreaterThan(214);
  });
});
