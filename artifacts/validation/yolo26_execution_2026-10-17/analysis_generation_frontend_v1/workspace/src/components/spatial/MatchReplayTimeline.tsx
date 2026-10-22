import { ArrowCounterClockwise, CaretLeft, CaretRight, Pause, Play, Repeat } from "@phosphor-icons/react";
import type { AnalysisRun } from "../../data/contracts";
import type { ReplayTransport } from "../../hooks/useReplay";
import { humanize } from "../../lib/format";

export function MatchReplayTimeline({ run, replay, selectedEventId, onSelectEvent }: { run: AnalysisRun; replay: ReplayTransport; selectedEventId: number | null; onSelectEvent: (id: number) => void }) {
  return <div className="astra-replay" aria-label="Synchronized replay controls">
    <div className="replay-ticks"><span>0:00</span><span>{(replay.lastFrame / run.summary.fps / 2).toFixed(2)}s</span><span>{(replay.lastFrame / run.summary.fps).toFixed(2)}s</span></div>
    <div className="replay-track">
      <div className="replay-event-marks" aria-hidden="true">{run.events.map(event => <i key={event.event_id} className={event.event_type === "BOUNCE" ? "bounce-mark" : "contact-mark"} style={{ left: `${event.frame / Math.max(1, replay.lastFrame) * 100}%` }} />)}</div>
      <input type="range" min={0} max={replay.lastFrame} step={1} value={Math.floor(replay.frame)} onChange={event => replay.seek(Number(event.target.value))} aria-label="Replay frame" aria-valuetext={`Frame ${Math.floor(replay.frame)}, ${replay.time.toFixed(3)} seconds`} disabled={run.summary.frames < 2} />
    </div>
    <div className="replay-controls">
      <div className="replay-primary"><button type="button" className="replay-play" aria-label={replay.playing ? "Pause replay" : "Play replay"} onClick={replay.toggle} disabled={run.summary.frames < 2}>{replay.playing ? <Pause size={18} weight="fill" /> : <Play size={18} weight="fill" />}</button><button type="button" aria-label="Previous replay frame" onClick={() => replay.step(-1)}><CaretLeft size={18} /></button><button type="button" aria-label="Next replay frame" onClick={() => replay.step(1)}><CaretRight size={18} /></button><span className="replay-time">{replay.time.toFixed(2)}<small> / {(replay.lastFrame / run.summary.fps).toFixed(2)} s</small></span></div>
      <div className="replay-secondary"><label>Speed<select aria-label="Replay speed" value={replay.rate} onChange={event => replay.setRate(Number(event.target.value))}>{[.25, .5, 1, 1.5, 2].map(rate => <option key={rate} value={rate}>{rate}×</option>)}</select></label><button type="button" aria-label="Loop replay" aria-pressed={replay.loop} onClick={() => replay.setLoop(value => !value)}><Repeat size={18} /></button><button type="button" aria-label="Restart replay" onClick={() => replay.seek(0)}><ArrowCounterClockwise size={18} /></button><span className="replay-frame">F {Math.floor(replay.frame).toString().padStart(3, "0")}</span></div>
    </div>
    {run.events.length ? <div className="replay-events" aria-label="Replay event timeline">{run.events.map(event => <button key={event.event_id} type="button" aria-pressed={selectedEventId === event.event_id} onClick={() => onSelectEvent(event.event_id)}><i className={event.event_type === "BOUNCE" ? "bounce-mark" : "contact-mark"} /><span>{humanize(event.event_type)}</span><small>{event.timestamp_s.toFixed(2)}s</small></button>)}</div> : <p className="replay-no-events">No event annotations in this analysis. Frame replay is still available.</p>}
  </div>;
}
