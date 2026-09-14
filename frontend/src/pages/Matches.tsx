import { ArrowRight, FileCode, FilmStrip, StackSimple } from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";
import { useAnalysis } from "../app/AnalysisContext";
import { Card, PageHeader, StatusBadge } from "../components/ui";
import { formatDateTime, formatDuration } from "../lib/format";

export default function Matches() {
  const { runs, selectedRunId, selectRun } = useAnalysis();
  const navigate = useNavigate();
  const openRun = (id: string) => {
    selectRun(id);
    navigate(`/analysis?run=${id}`);
  };

  return (
    <div className="page matches-page">
      <PageHeader eyebrow="Artifact library" title="Analysis runs" description="Seven real pipeline generations, normalized into one stable frontend contract. Feature gaps remain explicit as the pipeline evolves." />
      <div className="matches-summary-bar" role="status">
        <span><StackSimple size={18} weight="duotone" /><strong>{runs.length}</strong> runs</span>
        <span><FilmStrip size={18} weight="duotone" /><strong>{runs.filter((run) => run.capabilities.video).length}</strong> video artifacts</span>
        <span><FileCode size={18} weight="duotone" /><strong>{runs.reduce((sum, run) => sum + run.artifacts.length, 0)}</strong> indexed artifacts</span>
      </div>
      <section className="run-library" aria-label="Available analysis runs">
        {runs.map((run, index) => (
          <Card className={`run-card ${run.id === selectedRunId ? "run-card-selected" : ""}`} key={run.id}>
            <div className="run-index">{String(index + 1).padStart(2, "0")}</div>
            <div className="run-card-main">
              <div className="run-title-line"><div><span>{run.phase}</span><h2>{run.label}</h2></div><StatusBadge tone={run.id === selectedRunId ? "signal" : "neutral"}>{run.status}</StatusBadge></div>
              <dl className="run-facts">
                <div><dt>Analyzed</dt><dd>{formatDateTime(run.analyzedAt)}</dd></div>
                <div><dt>Duration</dt><dd>{formatDuration(run.durationSeconds)}</dd></div>
                <div><dt>Frames</dt><dd>{run.frames}</dd></div>
                <div><dt>Artifacts</dt><dd>{run.artifacts.length}</dd></div>
              </dl>
              <div className="capability-row" aria-label="Available features">
                {Object.entries(run.capabilities).map(([name, available]) => <span key={name} className={available ? "capability-available" : "capability-missing"}>{name}</span>)}
              </div>
            </div>
            <button className="run-open" type="button" onClick={() => openRun(run.id)} aria-label={`Open ${run.label}`}>
              <ArrowRight size={18} aria-hidden="true" />
            </button>
          </Card>
        ))}
      </section>
    </div>
  );
}

