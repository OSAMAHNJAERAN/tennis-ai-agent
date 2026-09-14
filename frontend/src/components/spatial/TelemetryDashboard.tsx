import { lazy, Suspense, useEffect, useId, useMemo, useRef, useState } from "react";
import { Cube, DownloadSimple, Eye, MapTrifold, Path, Television, Waves } from "@phosphor-icons/react";
import type { AnalysisRun } from "../../data/contracts";
import { useReplay } from "../../hooks/useReplay";
import { useReducedMotion } from "../../hooks/useReducedMotion";
import { buildCoachEvidence, normalizeTelemetry, playerSpeed, samplePoints, spatialStats, type FrameRange } from "../../lib/tennisTelemetry";
import { humanize } from "../../lib/format";
import { VideoEvidencePlayer } from "../VideoEvidencePlayer";
import { TacticalCourt } from "./TacticalCourt";
import { SpatialHeatmapPanel } from "./SpatialHeatmapPanel";
import { MatchReplayTimeline } from "./MatchReplayTimeline";
import { CoachPanel } from "./CoachPanel";

const TennisCourt3D = lazy(() => import("./TennisCourt3D"));
type View = "video" | "court" | "3d" | "heatmap";
const VIEWS = [{ id: "video", label: "Video Feed", icon: Television }, { id: "court", label: "2D Court Model", icon: MapTrifold }, { id: "3d", label: "3D Court", icon: Cube }, { id: "heatmap", label: "Spatial Heatmap", icon: Waves }] as const;
type Props = { run: AnalysisRun; selectedEventId: number | null; onSelectEvent: (id: number | null) => void };

export function TelemetryDashboard(props: Props) { return <Workspace key={props.run.summary.id} {...props} />; }

function Workspace({ run, selectedEventId, onSelectEvent }: Props) {
  const id = useId(); const tabs = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<View>("3d");
  const [trails, setTrails] = useState(true); const [arc, setArc] = useState(false);
  const reducedMotion = useReducedMotion();
  const [range, setRange] = useState<FrameRange>([0, Math.max(0, run.summary.frames - 1)]);
  const telemetry = useMemo(() => normalizeTelemetry(run), [run]);
  const selectedEvent = run.events.find(event => event.event_id === selectedEventId) ?? null;
  const replay = useReplay(run.summary.frames, run.summary.fps, selectedEvent?.frame ?? 0);
  const { seek, setPlaying } = replay;
  useEffect(() => { if (selectedEvent) { seek(selectedEvent.frame); setPlaying(false); } }, [selectedEvent, seek, setPlaying]);
  const currentFrame = Math.floor(replay.frame);
  const ball = telemetry.ball[currentFrame];
  const mapped = run.court?.isValid && telemetry.players.some(points => points.some(Boolean));
  const stats = useMemo(() => ["player1", "player2"].map(filter => spatialStats(samplePoints(telemetry, filter as "player1" | "player2", range))), [range, telemetry]);
  const selectedCall = run.lineCalls.find(call => call.event_id === selectedEventId);
  const selectEvent = (eventId: number) => {
    const event = run.events.find(value => value.event_id === eventId);
    if (event) { replay.seek(event.frame); replay.setPlaying(false); }
    onSelectEvent(eventId);
  };
  function changeTab(event: React.KeyboardEvent) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const index = VIEWS.findIndex(item => item.id === view);
    const next = event.key === "Home" ? 0 : event.key === "End" ? VIEWS.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + VIEWS.length) % VIEWS.length;
    setView(VIEWS[next]!.id); tabs.current?.querySelectorAll<HTMLButtonElement>("button")[next]?.focus();
  }
  function exportEvidence() {
    const report = { schemaVersion: 1, exportedAt: new Date().toISOString(), ...buildCoachEvidence(run, telemetry, range) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${run.summary.id}-spatial-evidence.json`; anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className="astra-workspace" aria-label="Synchronized Court Telemetry View">
    <div className="astra-workspace-bar"><div><span className="astra-section-label"><i />SPATIAL WORKSPACE</span><span className="workspace-run">{run.summary.frames} frames · {run.summary.fps} FPS source</span></div><button type="button" className="astra-text-button" onClick={exportEvidence}><DownloadSimple size={16} />Export evidence</button></div>
    <div className="astra-workspace-grid">
      <div className="astra-analysis-main">
        <div className="astra-view-tabs" ref={tabs} role="tablist" aria-label="Telemetry representation" onKeyDown={changeTab}>{VIEWS.map(({ id: value, label, icon: Icon }) => <button key={value} id={`${id}-tab-${value}`} type="button" role="tab" aria-selected={view === value} aria-controls={`${id}-panel-${value}`} tabIndex={view === value ? 0 : -1} onClick={() => setView(value)}><Icon size={17} weight={view === value ? "fill" : "regular"} /><span>{label}</span>{value === "3d" ? <small>NEW</small> : null}</button>)}</div>
        <div className="astra-spatial-stage">
          {view !== "heatmap" ? <div className="spatial-stage-header"><span><i className={replay.playing ? "is-playing" : ""} />{replay.playing ? "REPLAYING" : "REPLAY READY"}</span><div><span className="player-one">● P1</span><span className="player-two">◆ P2</span><span className="ball-legend">○ Ball</span></div></div> : null}
          <div id={`${id}-panel-video`} role="tabpanel" aria-labelledby={`${id}-tab-video`} hidden={view !== "video"} className="astra-view-panel">
            <VideoEvidencePlayer run={run} selectedEventId={selectedEventId} onSelectEvent={onSelectEvent} active={view === "video"} transport={{ time: replay.time, playing: replay.playing, rate: replay.rate, toggle: replay.toggle, seekTime: time => replay.seek(time * run.summary.fps) }} />
          </div>
          <div id={`${id}-panel-court`} role="tabpanel" aria-labelledby={`${id}-tab-court`} hidden={view !== "court"} className="astra-view-panel">{view === "court" ? mapped ? <TacticalCourt run={run} telemetry={telemetry} frame={replay.frame} trails={trails} selectedEventId={selectedEventId} onSelectEvent={selectEvent} /> : <Unavailable /> : null}</div>
          <div id={`${id}-panel-3d`} role="tabpanel" aria-labelledby={`${id}-tab-3d`} hidden={view !== "3d"} className="astra-view-panel">{view === "3d" ? mapped ? <Suspense fallback={<div className="spatial-loading"><Cube size={32} /><span>Constructing your court…</span></div>}><TennisCourt3D run={run} telemetry={telemetry} playhead={replay.playhead} frame={replay.frame} playing={replay.playing} trails={trails} arc={arc} reducedMotion={reducedMotion} onFallback={() => setView("court")} /></Suspense> : <Unavailable /> : null}</div>
          <div id={`${id}-panel-heatmap`} role="tabpanel" aria-labelledby={`${id}-tab-heatmap`} hidden={view !== "heatmap"} className="astra-view-panel">{view === "heatmap" ? <SpatialHeatmapPanel run={run} range={range} reducedMotion={reducedMotion} /> : null}</div>
          {(view === "3d" || view === "court") && mapped ? <div className="spatial-stage-options"><button type="button" aria-pressed={trails} onClick={() => setTrails(value => !value)}><Path size={15} />Movement trails</button>{view === "3d" ? <button type="button" aria-pressed={arc} onClick={() => setArc(value => !value)}><Eye size={15} />Illustrative arc</button> : null}<span>{arc && view === "3d" ? "Illustrative height · not measured" : "Court-plane projection · meters"}</span></div> : null}
        </div>
        <MatchReplayTimeline run={run} replay={replay} selectedEventId={selectedEventId} onSelectEvent={selectEvent} />
        <div className="astra-player-strip">{[0, 1].map(index => {
          const metrics = index === 0 ? run.playerMetrics?.player_1 : run.playerMetrics?.player_2;
          const speed = playerSpeed(telemetry.players[index] ?? [], replay.frame, run.summary.fps);
          return <article key={index} className={`astra-player-stat player-stat-${index}`}><div className="player-stat-identity"><i>P{index + 1}</i><span>Player {index + 1}<small>{telemetry.players[index]?.[currentFrame] ? "Position tracked" : "Position unavailable"}</small></span></div><dl><div><dt>Speed now</dt><dd>{speed == null ? "—" : speed.toFixed(1)}<small> km/h</small></dd></div><div><dt>Total distance</dt><dd>{metrics?.total_distance_m.toFixed(1) ?? "—"}<small> m</small></dd></div><div><dt>Behind baseline</dt><dd>{stats[index]?.total ? stats[index]!.behindBaseline.toFixed(0) : "—"}<small>%</small></dd></div></dl></article>;
        })}</div>
      </div>
      <CoachPanel key={`${run.summary.id}-${range[0]}-${range[1]}`} run={run} telemetry={telemetry} range={range} />
    </div>
    <div className="astra-evidence-row">
      <section className="astra-range-panel"><span className="astra-eyebrow">Analysis interval</span><h3>Focus the evidence.</h3><p>Apply a frame range to density, position breakdowns, coaching, and export.</p><div className="range-inputs"><label>From frame<input type="number" aria-label="Analysis start frame" min={0} max={range[1]} value={range[0]} onChange={event => setRange([Math.max(0, Math.min(range[1], Math.round(Number(event.target.value) || 0))), range[1]])} /></label><span>→</span><label>To frame<input type="number" aria-label="Analysis end frame" min={range[0]} max={replay.lastFrame} value={range[1]} onChange={event => setRange([range[0], Math.min(replay.lastFrame, Math.max(range[0], Math.round(Number(event.target.value) || 0)))])} /></label><button type="button" onClick={() => setRange([0, replay.lastFrame])}>Full clip</button></div></section>
      <section className="astra-event-inspector"><span className="astra-eyebrow">Event inspector</span><h3>{selectedEvent ? humanize(selectedEvent.event_type) : "A closer look at the moment."}</h3>{selectedEvent ? <><div className="event-inspector-facts"><span>Frame <b>{selectedEvent.frame}</b></span><span>Confidence <b>{(selectedEvent.confidence * 100).toFixed(0)}%</b></span><span>Player <b>{selectedEvent.player_id ? `P${selectedEvent.player_id}` : "Unknown"}</b></span><span>State <b>{humanize(selectedEvent.trajectory_state)}</b></span></div><p>{selectedCall ? `${humanize(selectedCall.decision)}: ${selectedCall.reason}` : "No line-call decision is attached to this event. Compare the video and court projection to review its evidence."}</p></> : <p>Select a contact or bounce in the timeline to inspect its frame, model confidence, and any attached line call.</p>}</section>
    </div>
    <footer className="astra-telemetry-footer"><span><i />{run.court?.isValid ? "Court mapping available" : "Court mapping unavailable"}</span><span>Frame {currentFrame} / {replay.lastFrame}</span><span>Ball: {ball ? humanize(ball.state) : "Unavailable"}</span><span>Ball speed: {ball?.speed == null ? "Unavailable" : `${ball.speed.toFixed(1)} km/h · court plane`}</span><span>Source: single camera</span></footer>
  </section>;
}
function Unavailable() { return <div className="spatial-empty" role="status"><Cube size={34} /><h3>Court replay unavailable</h3><p>This analysis does not include valid mapped player trajectories. Select a run with spatial telemetry to explore the court.</p></div>; }
