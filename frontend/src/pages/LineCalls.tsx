import { Crosshair, Ruler, ShieldCheck, Waveform } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";
import { DecisionBadge } from "../components/DecisionBadge";
import { LineGeometryZoom } from "../components/LineGeometryZoom";
import { ReviewWorkflow } from "../components/ReviewWorkflow";
import { Card, EmptyState, PageHeader, SectionHeading, StatusBadge } from "../components/ui";
import { formatConfidence, formatNumber, humanize } from "../lib/format";

export default function LineCalls() {
  const { run, selectedEventId, selectEvent } = useAnalysis();
  if (!run) return null;
  const selectedCall = run.lineCalls.find((call) => call.event_id === selectedEventId) ?? run.lineCalls[0] ?? null;

  return (
    <div className="page line-calls-page">
      <PageHeader eyebrow="Spatial adjudication" title="Line calls" description="Each decision pairs contact geometry with a signed edge margin, uncertainty, tracker state, contact model, and a plain-language reason." action={<StatusBadge tone="signal">{run.lineCalls.length} decisions</StatusBadge>} />
      <ReviewWorkflow calls={run.lineCalls} />
      {!selectedCall ? <EmptyState title="No line-call artifact" detail="Select Phase 4 or later to inspect spatial adjudication." /> : (
        <div className="line-call-workspace">
          <Card className="line-call-list-panel">
            <SectionHeading eyebrow="Decision index" title="Bounce record" detail="Select a call to inspect its spatial evidence." />
            <div className="decision-list">{run.lineCalls.map((call, index) => (
              <button type="button" key={call.event_id} className={call.event_id === selectedCall.event_id ? "decision-list-active" : ""} onClick={() => selectEvent(call.event_id)}>
                <span className="decision-number">{String(index + 1).padStart(2, "0")}</span>
                <span><DecisionBadge decision={call.decision} /><strong>Frame {call.bounce_frame}</strong><small>{humanize(call.nearest_line)}</small></span>
                <strong>{formatNumber(call.ball_edge_margin_cm, " cm")}</strong>
              </button>
            ))}</div>
          </Card>
          <div className="line-evidence-column">
            <Card className="geometry-panel">
              <div className="geometry-head"><div><p className="eyebrow">Contact geometry</p><h2>{humanize(selectedCall.nearest_line)}</h2></div><DecisionBadge decision={selectedCall.decision} /></div>
              <LineGeometryZoom call={selectedCall} />
              <p className="geometry-reason">{selectedCall.reason}</p>
            </Card>
            <div className="line-metric-grid">
              <Card><Crosshair size={18} /><span>Signed edge margin</span><strong>{formatNumber(selectedCall.ball_edge_margin_cm, " cm")}</strong><small>{selectedCall.ball_edge_margin_cm < 0 ? "Outside legal boundary" : "Inside legal boundary"}</small></Card>
              <Card><Waveform size={18} /><span>Position uncertainty</span><strong>{formatNumber(selectedCall.position_uncertainty_cm, " cm")}</strong><small>{humanize(selectedCall.spatial_tier)} spatial tier</small></Card>
              <Card><Ruler size={18} /><span>Contact patch</span><strong>{formatNumber(selectedCall.contact_patch_radius_cm, " cm")}</strong><small>{humanize(selectedCall.contact_patch_model)}</small></Card>
              <Card><ShieldCheck size={18} /><span>Decision confidence</span><strong>{formatConfidence(selectedCall.confidence)}</strong><small>{humanize(selectedCall.tracker_state)} tracker state</small></Card>
            </div>
            <Card className="method-record"><dl><div><dt>Refinement method</dt><dd>{humanize(selectedCall.refinement_method)}</dd></div><div><dt>Decision context</dt><dd>{humanize(selectedCall.decision_context)}</dd></div><div><dt>Center signed distance</dt><dd>{formatNumber(selectedCall.center_signed_distance_cm, " cm")}</dd></div><div><dt>Bounce coordinate</dt><dd>{selectedCall.bounce_position_m.map((value) => value.toFixed(3)).join(", ")} m</dd></div></dl></Card>
          </div>
        </div>
      )}
    </div>
  );
}

