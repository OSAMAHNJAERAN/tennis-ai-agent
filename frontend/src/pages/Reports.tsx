import { DownloadSimple, FileCode, FileCsv, FilePdf, Printer, ShieldWarning } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";
import { Card, Metric, PageHeader, SectionHeading, StatusBadge } from "../components/ui";
import { deriveCoverage } from "../lib/analytics";
import { formatDateTime, formatNumber, formatPercent } from "../lib/format";

function downloadFile(filename: string, contents: string, type: string) {
  const blob = new Blob([contents], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function csvCell(value: unknown) {
  const text = value == null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

export default function Reports() {
  const { run } = useAnalysis();
  if (!run) return null;
  const coverage = run.trajectories ? deriveCoverage(run.trajectories.ballTrajectory) : null;
  const exportJson = () => downloadFile(`${run.summary.id}.json`, JSON.stringify(run, null, 2), "application/json");
  const exportCsv = () => {
    const headers = ["event_id", "event_type", "frame", "timestamp_s", "player_id", "confidence", "trajectory_state"];
    const rows = run.events.map((event) => headers.map((header) => csvCell(event[header as keyof typeof event])).join(","));
    downloadFile(`${run.summary.id}-events.csv`, [headers.join(","), ...rows].join("\n"), "text/csv;charset=utf-8");
  };

  return (
    <div className="page reports-page">
      <PageHeader eyebrow="Portable outputs" title="Reports" description="Produce traceable client-side exports and a print-ready summary from the selected offline run. No data leaves this browser." action={<button className="button button-primary no-print" type="button" onClick={() => window.print()}><Printer size={16} /> Print report</button>} />
      <Card className="report-cover">
        <div className="report-cover-index">T88 / {run.summary.phase}</div>
        <div><p className="eyebrow">Selected analysis report</p><h2>{run.summary.label}</h2><p>Generated from the versioned normalized snapshot. Source analyzed {formatDateTime(run.summary.analyzedAt)}.</p></div>
        <StatusBadge tone="signal">{run.summary.status}</StatusBadge>
      </Card>
      <section className="report-kpis" aria-label="Selected run summary">
        <Metric primary label="Pipeline throughput" value={formatNumber(run.pipelineMetrics?.pipelineFps, " FPS")} meta={`${run.summary.frames} source frames`} />
        <Metric label="Tracking coverage" value={formatPercent(coverage)} />
        <Metric label="Semantic events" value={run.events.length} />
        <Metric label="Line calls" value={run.lineCalls.length} />
        <Metric label="Court error" value={formatNumber(run.court?.reprojectionErrorPx, " px")} />
      </section>
      <section className="report-detail-grid">
        <Card className="panel-padding">
          <SectionHeading eyebrow="Match state" title="Scoring summary" detail="Unavailable fields remain explicit." />
          <dl className="report-facts"><div><dt>Match ID</dt><dd>{run.matchState?.match_id ?? "Unavailable"}</dd></div><div><dt>Point state</dt><dd>{run.matchState?.point_state ?? "Unavailable"}</dd></div><div><dt>Server</dt><dd>{run.matchState ? `Player ${run.matchState.server_id}` : "Unavailable"}</dd></div><div><dt>Serve attempt</dt><dd>{run.matchState?.serve_attempt ?? "Unavailable"}</dd></div><div><dt>Match complete</dt><dd>{run.matchState ? (run.matchState.match_complete ? "Yes" : "No") : "Unavailable"}</dd></div></dl>
        </Card>
        <Card className="panel-padding">
          <SectionHeading eyebrow="Artifacts" title="Availability ledger" detail={`${run.summary.artifacts.length} committed source artifact references.`} />
          <div className="artifact-ledger">{Object.entries(run.summary.capabilities).map(([name, available]) => <div key={name}><span>{name}</span><strong>{available ? "Available" : "Unavailable"}</strong></div>)}</div>
        </Card>
      </section>
      <section className="export-section no-print">
        <SectionHeading eyebrow="Client-side delivery" title="Export formats" detail="Exports are assembled locally from the selected run." />
        <div className="export-grid">
          <Card className="export-card"><span className="export-icon"><FileCode size={24} weight="duotone" /></span><div><h3>Normalized JSON</h3><p>Complete selected-run contract with scientific enums and metadata.</p></div><button className="button button-primary" type="button" onClick={exportJson}><DownloadSimple size={16} /> Download JSON</button></Card>
          <Card className="export-card"><span className="export-icon"><FileCsv size={24} weight="duotone" /></span><div><h3>Events CSV</h3><p>Portable semantic event log with frame and time references.</p></div><button className="button" type="button" onClick={exportCsv} disabled={!run.events.length}><DownloadSimple size={16} /> Download CSV</button></Card>
          <Card className="export-card export-disabled"><span className="export-icon"><FilePdf size={24} weight="duotone" /></span><div><h3>Server PDF</h3><p>Unavailable because this frontend has no report-generation service.</p></div><button className="button" type="button" disabled><ShieldWarning size={16} /> Server required</button></Card>
        </div>
      </section>
    </div>
  );
}
