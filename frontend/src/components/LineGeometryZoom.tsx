import { useId } from "react";
import type { LineCall } from "../data/contracts";

export function LineGeometryZoom({ call }: { call: LineCall }) {
  const titleId = useId();
  const margin = Math.max(-250, Math.min(250, call.ball_edge_margin_cm));
  const centerX = 260 + margin * .7;
  const uncertainty = Math.max(4, call.position_uncertainty_cm * 2.2);
  const radius = Math.max(12, call.contact_patch_radius_cm * 3.2);
  return (
    <svg className="line-geometry-zoom" viewBox="0 0 520 250" role="img" aria-labelledby={titleId}>
      <title id={titleId}>Zoomed {call.nearest_line} geometry with contact patch, signed margin, and uncertainty</title>
      <rect className="zoom-surface" x="1" y="1" width="518" height="248" rx="12" />
      <rect className="line-strip" x="244" y="24" width="32" height="202" />
      <line className="line-center" x1="260" y1="24" x2="260" y2="226" />
      <rect className="uncertainty-band" x={centerX - uncertainty} y="75" width={uncertainty * 2} height="100" rx="8" />
      <circle className="contact-patch" cx={centerX} cy="125" r={radius} />
      <line className="margin-arrow" x1="260" y1="205" x2={centerX} y2="205" />
      <circle className="margin-end" cx={centerX} cy="205" r="4" />
      <text x="24" y="34">OUTSIDE</text>
      <text x="496" y="34" textAnchor="end">LEGAL SIDE</text>
      <text x="260" y="240" textAnchor="middle">{call.ball_edge_margin_cm.toFixed(1)} CM SIGNED MARGIN</text>
    </svg>
  );
}

