import * as ToggleGroup from "@radix-ui/react-toggle-group";
import {
  ArrowCounterClockwise,
  CaretLeft,
  CaretRight,
  CornersOut,
  Pause,
  Play,
  SlidersHorizontal,
  SpeakerHigh,
  SpeakerX,
} from "@phosphor-icons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisRun } from "../data/contracts";
import { currentFrameIndex, getAdjacentEvent } from "../lib/analytics";
import { formatDuration, humanize } from "../lib/format";

type OverlayKey = "players" | "ball" | "trajectory" | "court" | "events" | "calls";

export function VideoEvidencePlayer({
  run,
  selectedEventId,
  onSelectEvent,
}: {
  run: AnalysisRun;
  selectedEventId: number | null;
  onSelectEvent: (id: number | null) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameCallbackRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(run.summary.durationSeconds);
  const [speed, setSpeed] = useState(1);
  const [overlays, setOverlays] = useState<OverlayKey[]>(["players", "ball", "trajectory", "court", "events", "calls"]);
  const fps = run.summary.fps;
  const frameIndex = currentFrameIndex(currentTime, fps, run.summary.frames);
  const selectedEvent = useMemo(() => run.events.find((event) => event.event_id === selectedEventId) ?? null, [run.events, selectedEventId]);

  const drawOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const index = currentFrameIndex(video.currentTime, fps, run.summary.frames);
    const frame = run.detections?.frames[index];
    ctx.lineWidth = Math.max(3, canvas.width / 640);
    ctx.font = `600 ${Math.max(18, canvas.width / 75)}px Geist Mono, monospace`;

    if (overlays.includes("court") && run.court) {
      ctx.fillStyle = "#dbfe00";
      for (const [x, y] of run.court.courtKeypointsPx) {
        ctx.beginPath();
        ctx.arc(x, y, Math.max(4, canvas.width / 360), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (overlays.includes("players") && frame) {
      [frame.player_1, frame.player_2].forEach((player, playerIndex) => {
        if (!player?.bbox) return;
        const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = player.bbox;
        ctx.strokeStyle = playerIndex === 0 ? "#ffffff" : "#dbfe00";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = playerIndex === 0 ? "#ffffff" : "#dbfe00";
        ctx.fillText(`P${playerIndex + 1}`, x1, Math.max(24, y1 - 9));
      });
    }
    if (overlays.includes("trajectory") && run.trajectories) {
      const tail = run.trajectories.ballTrajectory.slice(Math.max(0, index - 18), index + 1).filter((point) => point.x_px != null && point.y_px != null);
      ctx.strokeStyle = "#dbfe00";
      ctx.lineWidth = Math.max(2, canvas.width / 800);
      ctx.beginPath();
      tail.forEach((point, tailIndex) => {
        if (point.x_px == null || point.y_px == null) return;
        if (tailIndex === 0) ctx.moveTo(point.x_px, point.y_px);
        else ctx.lineTo(point.x_px, point.y_px);
      });
      ctx.stroke();
    }
    if (overlays.includes("ball") && frame?.ball?.position_px) {
      const [x, y] = frame.ball.position_px;
      ctx.fillStyle = "#dbfe00";
      ctx.strokeStyle = "#262626";
      ctx.lineWidth = Math.max(2, canvas.width / 1000);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(7, canvas.width / 240), 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    if (overlays.includes("events")) {
      const event = run.events.find((item) => Math.abs(item.frame - index) <= 1);
      if (event) {
        ctx.fillStyle = "#262626";
        ctx.fillRect(26, 26, Math.min(420, canvas.width - 52), 58);
        ctx.fillStyle = "#dbfe00";
        ctx.fillText(humanize(event.event_type), 44, 63);
      }
    }
    if (overlays.includes("calls")) {
      const call = run.lineCalls.find((item) => Math.abs(item.bounce_frame - index) <= 1);
      if (call) {
        ctx.fillStyle = "#dbfe00";
        ctx.fillRect(26, canvas.height - 88, Math.min(500, canvas.width - 52), 58);
        ctx.fillStyle = "#262626";
        ctx.fillText(`${humanize(call.decision)} · ${call.ball_edge_margin_cm.toFixed(1)} CM`, 44, canvas.height - 51);
      }
    }
  }, [fps, overlays, run]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let active = true;
    const tick = () => {
      if (!active) return;
      setCurrentTime(video.currentTime);
      drawOverlay();
      if (typeof video.requestVideoFrameCallback === "function") {
        frameCallbackRef.current = video.requestVideoFrameCallback(tick);
      } else {
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    tick();
    return () => {
      active = false;
      if (frameCallbackRef.current != null && typeof video.cancelVideoFrameCallback === "function") video.cancelVideoFrameCallback(frameCallbackRef.current);
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [drawOverlay]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !selectedEvent) return;
    video.currentTime = selectedEvent.timestamp_s;
    setCurrentTime(selectedEvent.timestamp_s);
  }, [selectedEvent]);

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play();
    else video.pause();
  };

  const seek = (time: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, Math.min(duration, time));
    setCurrentTime(video.currentTime);
    drawOverlay();
  };

  const stepEvent = (direction: -1 | 1) => {
    const event = getAdjacentEvent(run.events, selectedEventId, direction);
    if (event) onSelectEvent(event.event_id);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement) return;
    if (event.key === " ") { event.preventDefault(); togglePlayback(); }
    if (event.key === "ArrowLeft") { event.preventDefault(); seek(currentTime - (event.shiftKey ? 1 : 1 / fps)); }
    if (event.key === "ArrowRight") { event.preventDefault(); seek(currentTime + (event.shiftKey ? 1 : 1 / fps)); }
  };

  return (
    <div className="evidence-player" onKeyDown={onKeyDown} tabIndex={0} aria-label="Video evidence player. Space plays or pauses. Arrow keys step frames.">
      <div className="video-stage">
        <video
          ref={videoRef}
          src={run.videoUrl ?? undefined}
          muted={muted}
          playsInline
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
          onClick={togglePlayback}
          aria-label="Source match video"
        />
        <canvas ref={canvasRef} aria-hidden="true" />
        <div className="frame-chip"><strong>F {frameIndex}</strong><span>{currentTime.toFixed(3)} s</span></div>
      </div>
      <div className="timeline-wrap">
        <input className="video-scrubber" aria-label="Playback position" type="range" min="0" max={duration || 0} step={1 / fps} value={Math.min(currentTime, duration || 0)} onChange={(event) => seek(Number(event.target.value))} />
        <div className="semantic-ticks" aria-hidden="true">{run.events.map((event) => <span key={event.event_id} className={event.event_id === selectedEventId ? "tick-selected" : ""} style={{ left: `${(event.timestamp_s / duration) * 100}%` }} />)}</div>
      </div>
      <div className="player-controls">
        <div className="control-cluster">
          <button className="icon-button control-primary" type="button" onClick={togglePlayback} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={18} weight="fill" /> : <Play size={18} weight="fill" />}</button>
          <button className="icon-button" type="button" onClick={() => seek(0)} aria-label="Restart"><ArrowCounterClockwise size={18} /></button>
          <button className="icon-button" type="button" onClick={() => stepEvent(-1)} disabled={!run.events.length} aria-label="Previous event"><CaretLeft size={18} /></button>
          <button className="icon-button" type="button" onClick={() => stepEvent(1)} disabled={!run.events.length} aria-label="Next event"><CaretRight size={18} /></button>
          <span className="time-readout">{formatDuration(currentTime)} / {formatDuration(duration)}</span>
        </div>
        <div className="control-cluster control-options">
          <label className="speed-control"><span>Speed</span><select value={speed} onChange={(event) => { const next = Number(event.target.value); setSpeed(next); if (videoRef.current) videoRef.current.playbackRate = next; }}>{[0.25, 0.5, 1, 1.5, 2].map((value) => <option key={value} value={value}>{value}×</option>)}</select></label>
          <button className="icon-button" type="button" onClick={() => setMuted((value) => !value)} aria-label={muted ? "Unmute" : "Mute"}>{muted ? <SpeakerX size={18} /> : <SpeakerHigh size={18} />}</button>
          <button className="icon-button" type="button" onClick={() => videoRef.current?.requestFullscreen()} aria-label="Enter fullscreen"><CornersOut size={18} /></button>
        </div>
      </div>
      <div className="overlay-controls">
        <span><SlidersHorizontal size={16} /> Overlays</span>
        <ToggleGroup.Root type="multiple" value={overlays} onValueChange={(values) => setOverlays(values as OverlayKey[])} aria-label="Video overlay controls">
          {(["players", "ball", "trajectory", "court", "events", "calls"] as OverlayKey[]).map((key) => <ToggleGroup.Item key={key} value={key} aria-label={`Toggle ${key} overlay`}>{humanize(key)}</ToggleGroup.Item>)}
        </ToggleGroup.Root>
      </div>
    </div>
  );
}
