import { CheckCircle, Cpu, Database, FolderOpen, HardDrives, PlugsConnected, WarningCircle } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";
import { Card, PageHeader, SectionHeading, StatusBadge } from "../components/ui";
import { formatNumber, humanize } from "../lib/format";

function objectAt(config: Record<string, unknown> | null, key: string) {
  const value = config?.[key];
  return value && typeof value === "object" ? value as Record<string, unknown> : null;
}

function valueAt(config: Record<string, unknown> | null, key: string) {
  const value = config?.[key];
  return value == null ? "Unavailable" : Array.isArray(value) ? value.join(" × ") : String(value);
}

export default function System() {
  const { run } = useAnalysis();
  if (!run) return null;
  const player = objectAt(run.runConfig, "player_detection");
  const ball = objectAt(run.runConfig, "ball_detection");
  const court = objectAt(run.runConfig, "court_detection");
  const temporal = objectAt(run.runConfig, "temporal_tracking");
  const line = objectAt(run.runConfig, "line_calling");
  const components = [
    { name: "Player detector", architecture: "YOLO11m", path: valueAt(player, "model"), detail: `ByteTrack · confidence ${valueAt(player, "confidence")}` },
    { name: "Ball detector", architecture: "YOLO11s", path: valueAt(ball, "model"), detail: `Image size ${valueAt(ball, "imgsz")} · low confidence ${valueAt(ball, "low_conf")}` },
    { name: "Court keypoints", architecture: "ResNet-50 / FC(28)", path: valueAt(court, "model"), detail: `Input ${valueAt(court, "input_size")} · 14 landmarks` },
  ];

  return (
    <div className="page system-page">
      <PageHeader eyebrow="Integration ledger" title="System" description="Read-only visibility into the selected pipeline configuration, model paths, processing measurements, schema availability, and the frontend’s offline boundary." action={<StatusBadge tone="signal">Snapshot schema v1</StatusBadge>} />
      <section className="system-health-grid">
        <Card><span className="health-icon"><Database size={20} /></span><div><strong>Static repository</strong><small>Validated Zod snapshot</small></div><CheckCircle size={18} weight="fill" /></Card>
        <Card><span className="health-icon"><PlugsConnected size={20} /></span><div><strong>Runtime network</strong><small>Zero requests by design</small></div><CheckCircle size={18} weight="fill" /></Card>
        <Card><span className="health-icon"><Cpu size={20} /></span><div><strong>Processing</strong><small>{formatNumber(run.pipelineMetrics?.pipelineFps, " FPS")}</small></div><CheckCircle size={18} weight="fill" /></Card>
        <Card><span className="health-icon"><HardDrives size={20} /></span><div><strong>Source artifacts</strong><small>{run.summary.artifacts.length} indexed files</small></div><CheckCircle size={18} weight="fill" /></Card>
      </section>
      <div className="system-main-grid">
        <Card className="panel-padding model-registry">
          <SectionHeading eyebrow="Model registry" title="Detection and geometry" detail="Paths are copied from the selected run configuration." />
          {components.map((component) => <div className="model-record" key={component.name}><div className="model-monogram">{component.name.slice(0, 2).toUpperCase()}</div><div><span>{component.name}</span><strong>{component.architecture}</strong><code>{component.path}</code><small>{component.detail}</small></div><StatusBadge>Configured</StatusBadge></div>)}
        </Card>
        <Card className="panel-padding config-panel">
          <SectionHeading eyebrow="Tracking configuration" title="Temporal and spatial thresholds" detail="Configuration remains inspection-only." />
          <dl className="config-list">
            {Object.entries(temporal ?? {}).map(([key, value]) => <div key={key}><dt>{humanize(key)}</dt><dd>{String(value)}</dd></div>)}
            {Object.entries(line ?? {}).map(([key, value]) => <div key={key}><dt>{humanize(key)}</dt><dd>{String(value)}</dd></div>)}
          </dl>
        </Card>
      </div>
      <section className="system-lower-grid">
        <Card className="panel-padding artifact-panel">
          <SectionHeading eyebrow="Run manifest" title="Artifact availability" detail={run.summary.label} />
          <div className="artifact-file-grid">{run.summary.artifacts.map((artifact) => <span key={artifact}><FolderOpen size={15} /><code>{artifact}</code></span>)}</div>
        </Card>
        <Card className="panel-padding integration-panel">
          <SectionHeading eyebrow="Frontend boundary" title="Replaceable repository" detail="The UI depends on three methods only." />
          <div className="interface-code"><code>listRuns(): Promise&lt;AnalysisRunSummary[]&gt;</code><code>getRun(id): Promise&lt;AnalysisRun&gt;</code><code>getVideoUrl(id): string | null</code></div>
          <div className="integration-note"><WarningCircle size={18} /><p>Historical artifacts normalize during the frontend-only sync step. Missing server PDF, backend search, and live processing capabilities stay disabled.</p></div>
        </Card>
      </section>
    </div>
  );
}

