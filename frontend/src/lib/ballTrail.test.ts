import { describe, expect, it } from 'vitest';
import { createBallTrailIndex } from './ballTrail';

type Point = { frame_index: number; x_px: number | null; y_px: number | null; label: string };
const point = (frame: number, label = String(frame), x: number | null = frame, y: number | null = frame): Point =>
  ({ frame_index: frame, x_px: x, y_px: y, label });

// The previous renderer is the compatibility oracle, separate from the index.
function previousTail<T extends Pick<Point, 'frame_index' | 'x_px' | 'y_px'>>(samples: ReadonlyArray<T>, frame: number): T[] {
  return samples.filter(value => value.frame_index <= frame && value.frame_index >= frame - 24 && value.x_px != null && value.y_px != null)
    .sort((a, b) => a.frame_index - b.frame_index);
}

describe('ball overlay trail index', () => {
  it('keeps sparse observations and stable duplicate order without changing the source', () => {
    const samples = [point(27), point(24, '24-first'), point(0), point(25, '25-first'),
      point(23), point(24, '24-second'), point(25, '25-missing', null), point(25, '25-second'), point(50)];
    samples.forEach(Object.freeze);
    Object.freeze(samples);
    const trail = createBallTrailIndex(samples);
    expect(trail(27).map(value => value.label)).toEqual(['23', '24-first', '24-second', '25-first', '25-second', '27']);
    expect(trail(27)).toEqual(previousTail(samples, 27));
    expect(samples.map(value => value.label)).toEqual(['27', '24-first', '0', '25-first', '23', '24-second', '25-missing', '25-second', '50']);
    expect(trail(0)).toEqual([samples[2]]);
  });

  it('uses the inclusive 25-frame window at clip boundaries and fractional query times', () => {
    const samples = Array.from({ length: 60 }, (_, frame) => point(frame));
    const trail = createBallTrailIndex(samples);
    expect(trail(0).map(value => value.frame_index)).toEqual([0]);
    expect(trail(24).map(value => value.frame_index)).toEqual(Array.from({ length: 25 }, (_, frame) => frame));
    expect(trail(25).map(value => value.frame_index)).toEqual(Array.from({ length: 25 }, (_, frame) => frame + 1));
    expect(trail(24.5).map(value => value.frame_index)).toEqual(Array.from({ length: 24 }, (_, frame) => frame + 1));
    expect(trail(59).map(value => value.frame_index)).toEqual(Array.from({ length: 25 }, (_, frame) => frame + 35));
    expect(trail(83)).toEqual([samples[59]]);
    expect(trail(84)).toEqual([]);
    expect(trail(-1)).toEqual([]);
    expect(createBallTrailIndex([])(0)).toEqual([]);
  });

  it('retains zero coordinates, omits null coordinates, and does not replace absent frames', () => {
    const samples = [point(10, 'zero', 0, 0), point(11, 'missing-x', null), point(12, 'missing-y', 12, null), point(14)];
    const trail = createBallTrailIndex(samples);
    expect(trail(14).map(value => value.label)).toEqual(['zero', '14']);
    expect(trail(14)[0]).toBe(samples[0]);
    // Frame 13 was not exported; no observation is manufactured to join the gap.
    expect(trail(14).some(value => value.frame_index === 13)).toBe(false);
  });

  it('skips actual array holes just as the previous filter did', () => {
    const samples: Point[] = [];
    samples[5] = point(8);
    samples[20] = point(6, 'first');
    samples[23] = point(6, 'second');
    const trail = createBallTrailIndex(samples);
    expect(trail(8).map(value => value.label)).toEqual(['first', 'second', '8']);
    expect(trail(8)).toEqual(previousTail(samples, 8));
    expect(samples.length).toBe(24);
    expect(0 in samples).toBe(false);
  });

  it.each(['ordered', 'out-of-order with duplicates'] as const)('bounds sample reads for a two-hour 60 FPS %s recording', order => {
    const count = 7200 * 60;
    const reads = { frame: 0, coordinate: 0 };
    class Observation {
      readonly frame: number;
      readonly missing: boolean;
      constructor(frame: number) { this.frame = frame; this.missing = frame % 97 === 0; }
      get frame_index() { ++reads.frame; return this.frame; }
      get x_px() { ++reads.coordinate; return this.missing ? null : this.frame % 100; }
      get y_px() { ++reads.coordinate; return this.missing ? null : this.frame % 90; }
    }
    const samples: Observation[] = [];
    for (let input = 0; input < count; ++input) {
      const frame = order === 'ordered' ? input : input * 17 % count;
      if (order !== 'ordered' && (frame === 101 || frame === 216003)) continue;
      samples.push(new Observation(frame));
    }
    if (order !== 'ordered') samples.push(new Observation(216000), new Observation(count - 17));
    expect(samples).toHaveLength(432000);
    const trail = createBallTrailIndex(samples);
    const checkpoints = [0, 24, 25, 110, 216008, count - 1, 97, 216000];
    for (const frame of checkpoints) {
      reads.frame = 0; reads.coordinate = 0;
      const actual = trail(frame);
      const queryReads = { ...reads };
      // Allow two binary searches and the local observations, never a full scan.
      expect(queryReads.frame, `frame reads at ${frame}`).toBeLessThanOrEqual(2 * Math.ceil(Math.log2(count)) + 2);
      expect(queryReads.coordinate, `coordinate reads at ${frame}`).toBeLessThanOrEqual(2 * 27);
      expect(actual).toEqual(previousTail(samples, frame));
    }
  });
});
