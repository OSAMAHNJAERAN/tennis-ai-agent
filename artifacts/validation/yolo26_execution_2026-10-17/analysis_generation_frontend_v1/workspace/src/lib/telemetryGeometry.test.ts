import { describe, expect, it } from "vitest";
import { calculateMediaRect, canvasBitmapSize, createCoordinateTransform } from "./telemetryGeometry";

describe("telemetry coordinate geometry", () => {
  it("scales a 16:9 source into a matching container", () => {
    const transform = createCoordinateTransform({ width: 1920, height: 1080 }, { width: 960, height: 540 });
    expect(transform.mediaRect).toEqual({ x: 0, y: 0, width: 960, height: 540, scale: 0.5 });
    expect(transform.point([960, 540])).toEqual([480, 270]);
    expect(transform.box([100, 200, 500, 600])).toEqual({ x: 50, y: 100, width: 200, height: 200 });
  });

  it("accounts for vertical letterboxing in a square container", () => {
    const transform = createCoordinateTransform({ width: 1920, height: 1080 }, { width: 1000, height: 1000 });
    expect(transform.mediaRect.scale).toBeCloseTo(0.5208333);
    expect(transform.mediaRect.y).toBeCloseTo(218.75);
    expect(transform.point([960, 540])).toEqual([500, 500]);
    expect(transform.normalizedPoint([0.5, 0.5])).toEqual([500, 500]);
  });

  it.each([
    ["4:3", { width: 1024, height: 768 }],
    ["3:2", { width: 1500, height: 1000 }],
    ["21:9", { width: 2520, height: 1080 }],
    ["portrait", { width: 1080, height: 1920 }],
  ])("keeps %s media centered and fully visible", (_label, source) => {
    const rect = calculateMediaRect(source, { width: 900, height: 540 });
    expect(rect.x).toBeGreaterThanOrEqual(0);
    expect(rect.y).toBeGreaterThanOrEqual(0);
    expect(rect.x + rect.width).toBeLessThanOrEqual(900.000001);
    expect(rect.y + rect.height).toBeLessThanOrEqual(540.000001);
  });

  it("supports cover cropping with negative offsets", () => {
    const rect = calculateMediaRect({ width: 1920, height: 1080 }, { width: 1000, height: 1000 }, "cover");
    expect(rect.height).toBe(1000);
    expect(rect.x).toBeLessThan(0);
  });

  it("returns a safe empty rectangle for uninitialized dimensions", () => {
    expect(calculateMediaRect({ width: 0, height: 1080 }, { width: 1000, height: 600 })).toEqual({
      x: 0,
      y: 0,
      width: 0,
      height: 0,
      scale: 0,
    });
  });

  it.each([1, 1.25, 1.5, 2])("creates a DPR-safe bitmap at DPR %s", (dpr) => {
    expect(canvasBitmapSize({ width: 800, height: 450 }, dpr)).toEqual({
      width: Math.round(800 * dpr),
      height: Math.round(450 * dpr),
    });
  });
});
