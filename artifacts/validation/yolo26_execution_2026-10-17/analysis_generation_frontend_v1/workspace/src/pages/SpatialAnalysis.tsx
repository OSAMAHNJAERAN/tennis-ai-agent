import { CheckCircle, Crosshair, Path, Scan, Timer } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";
import { SynchronizedTelemetryView } from "../components/SynchronizedTelemetryView";
import { DataError, SkeletonPage } from "../components/ui";
import '../styles/upload.css';
import type { FrameRange } from '../lib/tennisTelemetry';

const DOWNLOADS = [
  ['readableReport', 'readable-report', 'Download report'],
  ['annotatedVideo', 'annotated', 'Download annotated video'],
  ['reportJson', 'report', 'Analysis JSON'],
  ['csv', 'csv', 'Movement CSV'],
  ['originalVideo', 'original', 'Original video'],
] as const;

export function MatchAnalysis({ active = true, range, onRangeChange }: { active?: boolean; range?: FrameRange; onRangeChange?: (value: FrameRange) => void }) {
  const { run, loading, error, selectedEventId, selectEvent, retry } = useAnalysis();
  if (loading) return <SkeletonPage />;
  if (!run || error) return <DataError message={error?.message ?? "Unable to load the analysis."} onRetry={retry} />;
  const distance = run.playerMetrics ? run.playerMetrics.player_1.total_distance_m + run.playerMetrics.player_2.total_distance_m : null;
  const downloads = DOWNLOADS.flatMap(([key, path, label]) => {
    const supplied = run.exports?.[key];
    const href = supplied === undefined && !run.analysisGeneration
      ? `/api/analyses/${run.summary.id}/download/${path}` : supplied;
    return href ? [{ key, label, href }] : [];
  });
  return <div className="astra-page">
    <header className="astra-page-heading"><div><span className="astra-eyebrow">ASTRA / TENNIS INTELLIGENCE</span><h1>Match replay</h1><p>Review the video alongside player movement and court evidence.</p></div><div className="astra-session-chip"><span><CheckCircle size={14} weight="fill" />{run.summary.status === "COMPLETED" ? "Analysis complete" : run.summary.status === "PARTIAL" ? "Partial analysis" : "Analysis failed"}</span><small>SINGLE CAMERA · RECORDED SESSION</small></div></header>
    <div className="astra-overview-strip" aria-label="Analysis summary">
      <div><Timer weight="duotone" /><span><small>Analyzed duration</small><strong>{run.summary.durationSeconds.toFixed(2)}<small>seconds</small></strong></span></div>
      <div><Crosshair weight="duotone" /><span><small>Detected events</small><strong>{run.summary.capabilities.events ? run.events.length.toString().padStart(2, "0") : "—"}<small>{run.summary.capabilities.events ? "to explore" : "unavailable"}</small></strong></span></div>
      <div><Path weight="duotone" /><span><small>Movement estimate</small><strong>{distance?.toFixed(1) ?? "—"}<small>meters · both players</small></strong></span></div>
      <div><Scan weight="duotone" /><span><small>Court reprojection</small><strong>{run.court?.isValid ? run.court.reprojectionErrorPx.toFixed(2) : "—"}<small>px residual</small></strong></span></div>
    </div>
    {run.warnings?.length ? <details className="analysis-notices"><summary>Measurement availability and confidence</summary><ul>{run.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul></details> : null}
    {run.annotatedVideoUrl && downloads.length ? <div className="analysis-downloads">{downloads.map(download => <a key={download.key} href={download.href}>{download.label}</a>)}</div> : null}
    <SynchronizedTelemetryView range={range} onRangeChange={onRangeChange} active={active} run={run} selectedEventId={selectedEventId} onSelectEvent={selectEvent} />
  </div>;
}
export default MatchAnalysis;
