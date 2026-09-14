import { useMemo } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Cube } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";
import { CoachPanel } from "../components/spatial/CoachPanel";
import { SpatialHeatmapPanel } from "../components/spatial/SpatialHeatmapPanel";
import { normalizeTelemetry, type FrameRange } from "../lib/tennisTelemetry";
import { useReducedMotion } from "../hooks/useReducedMotion";

export default function CoachWorkspace() {
  const { run } = useAnalysis();
  const reducedMotion = useReducedMotion();
  const telemetry = useMemo(() => run ? normalizeTelemetry(run) : null, [run]);
  const range = useMemo<FrameRange>(() => [0, Math.max(0, (run?.summary.frames ?? 1) - 1)], [run]);
  if (!run || !telemetry) return null;
  return <div className="astra-page"><header className="astra-page-heading"><div><span className="astra-eyebrow">ASTRA / COACHING</span><h1>Better questions.<em> Clearer decisions.</em></h1><p>Ground your next adjustment in the selected analysis.</p></div><Link className="astra-text-button" to={`/match-center?run=${encodeURIComponent(run.summary.id)}`}><ArrowLeft />Return to replay</Link></header><div className="astra-workspace-grid coach-workspace-grid"><section className="coach-spatial-reference"><header><Cube size={18} /><h2>Your positioning evidence</h2></header><SpatialHeatmapPanel key={run.summary.id} run={run} range={range} reducedMotion={reducedMotion} /></section><CoachPanel key={run.summary.id} run={run} telemetry={telemetry} range={range} /></div></div>;
}
