import { CheckCircle, Crosshair, Path, Scan, Timer } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";
import { SynchronizedTelemetryView } from "../components/SynchronizedTelemetryView";
import { DataError, SkeletonPage } from "../components/ui";

export function MatchAnalysis() {
  const { run, loading, error, selectedEventId, selectEvent, retry } = useAnalysis();
  if (loading) return <SkeletonPage />;
  if (!run || error) return <DataError message={error?.message ?? "Unable to load the analysis."} onRetry={retry} />;
  const distance = run.playerMetrics ? run.playerMetrics.player_1.total_distance_m + run.playerMetrics.player_2.total_distance_m : null;
  return <div className="astra-page">
    <header className="astra-page-heading"><div><span className="astra-eyebrow">ASTRA / TENNIS INTELLIGENCE</span><h1>See the game. <em>Read the court.</em></h1><p>Explore every movement in 3D. Turn recorded match evidence into your next adjustment.</p></div><div className="astra-session-chip"><span><CheckCircle size={14} weight="fill" />{run.summary.status === "COMPLETED" ? "Analysis complete" : run.summary.status === "PARTIAL" ? "Partial analysis" : "Analysis failed"}</span><small>SINGLE CAMERA · RECORDED SESSION</small></div></header>
    <div className="astra-overview-strip" aria-label="Analysis summary">
      <div><Timer weight="duotone" /><span><small>Analyzed duration</small><strong>{run.summary.durationSeconds.toFixed(2)}<small>seconds</small></strong></span></div>
      <div><Crosshair weight="duotone" /><span><small>Detected events</small><strong>{run.events.length.toString().padStart(2, "0")}<small>to explore</small></strong></span></div>
      <div><Path weight="duotone" /><span><small>Combined movement</small><strong>{distance?.toFixed(1) ?? "—"}<small>meters</small></strong></span></div>
      <div><Scan weight="duotone" /><span><small>Court reprojection</small><strong>{run.court?.isValid ? run.court.reprojectionErrorPx.toFixed(2) : "—"}<small>px residual</small></strong></span></div>
    </div>
    <SynchronizedTelemetryView run={run} selectedEventId={selectedEventId} onSelectEvent={selectEvent} />
  </div>;
}
export default MatchAnalysis;
