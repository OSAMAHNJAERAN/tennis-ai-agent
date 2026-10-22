import type { BallTrajectoryPoint } from '../data/contracts';

type ImageBallPoint = Pick<BallTrajectoryPoint, 'frame_index' | 'x_px' | 'y_px'>;

export function createBallTrailIndex<T extends ImageBallPoint>(samples: ReadonlyArray<T>) {
  // Dense, ordered exports can reuse their reference array. Legacy sparse or
  // unordered inputs need one stable sorted copy, never a sort on every draw.
  let previous = -Infinity;
  let visited = 0;
  const ordered = samples.every(point => {
    ++visited;
    const frame = point.frame_index;
    const valid = Number.isFinite(frame) && frame >= previous;
    previous = frame;
    return valid;
  });
  const indexed: ReadonlyArray<T> = ordered && visited === samples.length
    ? samples
    : samples.filter(point => Number.isFinite(point.frame_index)).sort((a, b) => a.frame_index - b.frame_index);

  function boundary(frame: number, inclusive: boolean) {
    let low = 0;
    let high = indexed.length;
    while (low < high) {
      const middle = low + Math.floor((high - low) / 2);
      // The index is dense and middle is strictly below its length.
      const value = indexed[middle]!.frame_index;
      if (value < frame || (inclusive && value === frame)) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  // Work follows the local window plus two binary searches. Duplicates remain
  // stable and require work proportional to the observations in that window.
  return (frame: number): T[] => {
    if (!Number.isFinite(frame)) return [];
    return indexed.slice(boundary(frame - 24, false), boundary(frame, true))
      .filter(point => point.x_px != null && point.y_px != null);
  };
}
