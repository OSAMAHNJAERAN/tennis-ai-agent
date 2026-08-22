import { describe, expect, it } from "vitest";
import type { BallTrajectoryPoint, MatchEvent } from "../data/contracts";
import { deriveCoverage, getAdjacentEvent, mapCourtPoint, trackingDistribution } from "./analytics";
import { formatConfidence, formatDuration, formatNumber } from "./format";

const point = (state: BallTrajectoryPoint["state"], x: number | null): BallTrajectoryPoint => ({
  frame_index: 0,
  timestamp_seconds: 0,
  x_px: x,
  y_px: x,
  court_x_m: null,
  court_y_m: null,
  speed_kmh: null,
  confidence: null,
  state,
});

const event = (id: number): MatchEvent => ({
  event_id: id,
  event_type: "BOUNCE",
  frame: id * 10,
  timestamp_s: id,
  confidence: 0.9,
  trajectory_state: "DETECTED",
});

describe("analytics utilities", () => {
  it("formats valid and unavailable values", () => {
    expect(formatNumber(16.47, " FPS")).toBe("16.5 FPS");
    expect(formatNumber(null)).toBe("Unavailable");
    expect(formatConfidence(0.954)).toBe("95%");
    expect(formatDuration(7.133)).toBe("0:07.1");
  });

  it("derives coverage without treating missing points as observed", () => {
    expect(deriveCoverage([point("DETECTED", 2), point("INTERPOLATED", 4), point("MISSING", null)])).toBeCloseTo(66.67, 1);
    expect(deriveCoverage([])).toBeNull();
  });

  it("counts scientific tracking states", () => {
    const distribution = trackingDistribution([point("DETECTED", 2), point("DETECTED", 3), point("PREDICTED", 4)]);
    expect(distribution.find((item) => item.state === "DETECTED")?.count).toBe(2);
    expect(distribution.find((item) => item.state === "PREDICTED")?.count).toBe(1);
  });

  it("maps canonical court corners into the padded SVG bounds", () => {
    expect(mapCourtPoint([0, 0], 420, 680, 34)).toEqual({ x: 34, y: 34 });
    expect(mapCourtPoint([10.97, 23.77], 420, 680, 34)).toEqual({ x: 386, y: 646 });
  });

  it("steps semantic events and clamps at timeline boundaries", () => {
    const events = [event(1), event(2), event(3)];
    expect(getAdjacentEvent(events, 1, 1)?.event_id).toBe(2);
    expect(getAdjacentEvent(events, 3, 1)?.event_id).toBe(3);
    expect(getAdjacentEvent(events, null, -1)?.event_id).toBe(1);
  });
});

