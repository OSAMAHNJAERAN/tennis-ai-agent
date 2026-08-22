import { useId } from "react";
import { mapCourtPoint } from "../lib/analytics";

export function OccupancyMap({ points, label }: { points: Array<[number, number] | null>; label: string }) {
  const titleId = useId();
  const width = 240;
  const height = 360;
  return (
    <svg className="occupancy-map" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={titleId}>
      <title id={titleId}>{label} grayscale occupancy map derived from tracked court positions</title>
      <rect x="12" y="12" width={width - 24} height={height - 24} rx="8" />
      <line x1={width / 2} y1="12" x2={width / 2} y2={height - 12} />
      {points.filter((point, index): point is [number, number] => Boolean(point) && index % 2 === 0).map((point, index) => {
        const mapped = mapCourtPoint([Math.max(0, Math.min(10.97, point[0])), Math.max(0, Math.min(23.77, point[1]))], width, height, 12);
        return <circle key={index} cx={mapped.x} cy={mapped.y} r="6" />;
      })}
    </svg>
  );
}

