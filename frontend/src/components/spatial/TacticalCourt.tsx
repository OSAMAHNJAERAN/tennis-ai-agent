import { useId } from "react";
import type { AnalysisRun } from "../../data/contracts";
import { COURT, FIELD, interpolatePosition, splitTrail, validPoint, type CourtPoint, type NormalizedTelemetry } from "../../lib/tennisTelemetry";

export function CourtLines2D() {
  const { width, length, singles, alley, service } = COURT;
  return <g className="astra-court-lines" fill="none" stroke="#c7d9d0" strokeWidth="0.045">
    <rect width={width} height={length} />
    {[alley, width - alley].map(x => <path key={x} d={`M${x} 0V${length}`} />)}
    {[length / 2 - service, length / 2 + service].map(y => <path key={y} d={`M${alley} ${y}h${singles}`} />)}
    <path d={`M${width / 2} ${length / 2 - service}v${service * 2} M${width / 2} 0v.15 M${width / 2} ${length}v-.15`} />
    <path d={`M-.914 ${length / 2}H${width + 0.914}`} stroke="#edf2e7" strokeWidth="0.11" />
  </g>;
}

export function TacticalCourt({ run, telemetry, frame, trails, selectedEventId, onSelectEvent }: { run: AnalysisRun; telemetry: NormalizedTelemetry; frame: number; trails: boolean; selectedEventId: number | null; onSelectEvent: (id: number) => void }) {
  const id = useId();
  const playerPoints = telemetry.players.map(points => interpolatePosition(points, frame));
  const ball = interpolatePosition(telemetry.ballPositions, frame, 5);
  const path = (points: CourtPoint[]) => points.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ");
  return <div className="tactical-stage">
    <div className="tactical-stage-copy"><span className="astra-eyebrow">Court plane</span><h3>Every position.<br />One perspective.</h3><p>Player foot positions and projected ball movement in meters, including baseline run-off.</p><div className="tactical-key"><span className="player-one">● P1</span><span className="player-two">◆ P2</span><span>○ Ball</span></div><small>Trails show recent movement.<br />Gaps remain visible.</small></div>
    <svg className="astra-tactical-court" viewBox={`${FIELD.minX} ${FIELD.minY} ${FIELD.maxX - FIELD.minX} ${FIELD.maxY - FIELD.minY}`} role="img" aria-labelledby={id}>
      <title id={id}>Tactical tennis court at frame {Math.floor(frame)} with player and ball positions</title>
      <rect x={FIELD.minX} y={FIELD.minY} width={FIELD.maxX - FIELD.minX} height={FIELD.maxY - FIELD.minY} rx="0.5" fill="#1a3032" />
      <rect width={COURT.width} height={COURT.length} fill="#345951" />
      <CourtLines2D />
      {trails ? [...telemetry.players, telemetry.ballPositions].flatMap((points, i) => splitTrail(points, frame, i === 2 ? 28 : 60, i === 2 ? 5 : 1).map((segment, j) => <path key={`${i}-${j}`} d={path(segment)} className={`tactical-trail tactical-trail-${i}`} fill="none" strokeWidth="0.08" strokeDasharray={i === 2 ? ".2 .12" : undefined} />)) : null}
      {run.events.filter(event => event.frame <= frame && validPoint(event.court_position_m)).map(event => <g key={event.event_id} transform={`translate(${event.court_position_m![0]} ${event.court_position_m![1]})`} className="tactical-event" role="button" tabIndex={0} aria-label={`Inspect ${event.event_type.toLowerCase()} at frame ${event.frame}`} onClick={() => onSelectEvent(event.event_id)} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelectEvent(event.event_id); } }}><circle r=".6" fill="transparent" /><circle r={event.event_id === selectedEventId ? ".38" : ".2"} fill="none" stroke="#f1d693" strokeWidth=".055" /><path d="M-.12 0h.24M0-.12v.24" stroke="#f1d693" strokeWidth=".04" /></g>)}
      {playerPoints.map((p, i) => p ? <g key={i} transform={`translate(${p[0]} ${p[1]})`} className={`tactical-player tactical-player-${i}`}>
        <circle r=".6" opacity=".15" /><circle r=".3" stroke="#102124" strokeWidth=".09" />
        <text y="-.7" textAnchor="middle" fontSize=".65" fontWeight="700" fill="currentColor">P{i + 1}</text>
      </g> : null)}
      {ball ? <g transform={`translate(${ball[0]} ${ball[1]})`}><circle r=".16" fill="#f0ff9d" /><circle r=".3" fill="none" stroke="#f0ff9d" strokeWidth=".025" /></g> : null}
    </svg>
  </div>;
}
