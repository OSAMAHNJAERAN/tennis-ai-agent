import * as ToggleGroup from "@radix-ui/react-toggle-group";
import {
  ArrowCounterClockwise,
  CaretLeft,
  CaretRight,
  CornersIn,
  CornersOut,
  Pause,
  Play,
  SlidersHorizontal,
  SpeakerHigh,
  SpeakerX,
} from "@phosphor-icons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisRun } from "../data/contracts";
import { currentFrameIndex } from "../lib/analytics";
import {
  canvasBitmapSize,
  createCoordinateTransform,
  type CoordinateTransform,
  type Point2D,
} from "../lib/telemetryGeometry";
import { formatDuration, humanize } from "../lib/format";

type OverlayKey = "players" | "ball" | "trajectory" | "court" | "events" | "calls";
type VideoStatus = "loading" | "ready" | "error";

type VideoMetadata = {
  width: number;
  height: number;
  duration: number;
};

const OVERLAY_KEYS: OverlayKey[] = ["players", "ball", "trajectory", "court", "events", "calls"];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function VideoEvidencePlayer({
  run,
  selectedEventId,
  onSelectEvent,
  active = true,
  onTimeChange,
  onMetadata,
  transport,
}: {
  run: AnalysisRun;
  selectedEventId: number | null;
  onSelectEvent: (id: number | null) => void;
  active?: boolean;
  onTimeChange?: (time: number) => void;
  onMetadata?: (metadata: VideoMetadata) => void;
  transport?: { time: number; playing: boolean; rate: number; toggle: () => void; seekTime: (time: number) => void };
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const transformRef = useRef<CoordinateTransform | null>(null);
  const dprRef = useRef(1);
  const frameCallbackRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const drawOverlayRef = useRef<() => void>(() => undefined);
  const onTimeChangeRef = useRef(onTimeChange);
  const lastPublishedTimeRef = useRef(-1);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [status, setStatus] = useState<VideoStatus>(run.videoUrl ? "loading" : "error");
  const [sourceSize, setSourceSize] = useState({ width: 0, height: 0 });
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(run.summary.durationSeconds);
  const [speed, setSpeed] = useState(1);
  const [overlays, setOverlays] = useState<OverlayKey[]>(["players", "ball", "trajectory"]);
  const fps = run.summary.fps;
  const frameIndex = currentFrameIndex(currentTime, fps, run.summary.frames);
  const selectedEvent = useMemo(
    () => run.events.find((event) => event.event_id === selectedEventId) ?? null,
    [run.events, selectedEventId],
  );

  const externalTime = transport?.time;
  const externalPlaying = transport?.playing;
  const externalRate = transport?.rate;
  useEffect(() => {
    const video = videoRef.current;
    if (!video || externalPlaying == null || status !== "ready") return;
    video.playbackRate = externalRate ?? 1;
    if (externalPlaying) void video.play().catch(() => { /* The independent telemetry clock remains usable if media playback is blocked. */ });
    else video.pause();
  }, [externalPlaying, externalRate, status]);

  useEffect(() => {
    onTimeChangeRef.current = onTimeChange;
  }, [onTimeChange]);

  const publishTime = useCallback((force = false) => {
    const video = videoRef.current;
    if (!video) return;
    const nextTime = video.currentTime;
    if (!force && Math.abs(nextTime - lastPublishedTimeRef.current) < 0.035) return;
    lastPublishedTimeRef.current = nextTime;
    setCurrentTime(nextTime);
    onTimeChangeRef.current?.(nextTime);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || externalTime == null) return;
    if (Math.abs(video.currentTime - externalTime) > (externalPlaying ? 0.12 : 0.001)) {
      video.currentTime = externalTime;
      publishTime(true);
      drawOverlayRef.current();
    }
  }, [externalTime, externalPlaying, status, publishTime]);

  const resizeCanvas = useCallback(() => {
    const video = videoRef.current;
    const stage = stageRef.current;
    const canvas = canvasRef.current;
    if (!video || !stage || !canvas || !video.videoWidth || !video.videoHeight) return;

    const bounds = stage.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const cssSize = { width: bounds.width, height: bounds.height };
    const dpr = clamp(window.devicePixelRatio || 1, 1, 3);
    const bitmap = canvasBitmapSize(cssSize, dpr);

    if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
    if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
    canvas.style.width = `${cssSize.width}px`;
    canvas.style.height = `${cssSize.height}px`;
    dprRef.current = dpr;
    transformRef.current = createCoordinateTransform(
      { width: video.videoWidth, height: video.videoHeight },
      cssSize,
      "contain",
    );
    drawOverlayRef.current();
  }, []);

  const hydrateVideoMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;
    const nextDuration = Number.isFinite(video.duration) ? video.duration : run.summary.durationSeconds;
    const metadata = { width: video.videoWidth, height: video.videoHeight, duration: nextDuration };
    setSourceSize({ width: metadata.width, height: metadata.height });
    setDuration(metadata.duration);
    setStatus("ready");
    onMetadata?.(metadata);
    requestAnimationFrame(resizeCanvas);
  }, [onMetadata, resizeCanvas, run.summary.durationSeconds]);

  const drawOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const transform = transformRef.current;
    if (!video || !canvas || !transform || !transform.mediaRect.width || !transform.mediaRect.height) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const dpr = dprRef.current;
    const { container, mediaRect } = transform;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, container.width, container.height);
    context.save();
    context.beginPath();
    context.rect(mediaRect.x, mediaRect.y, mediaRect.width, mediaRect.height);
    context.clip();

    const index = currentFrameIndex(video.currentTime, fps, run.summary.frames);
    const frame = run.detections?.frames[index];
    const strokeWidth = clamp(mediaRect.width / 560, 1.25, 2);
    const labelFontSize = clamp(mediaRect.width / 72, 10, 13);
    context.lineJoin = "round";
    context.lineCap = "round";

    if (overlays.includes("court") && run.court?.courtKeypointsPx.length) {
      for (const sourcePoint of run.court.courtKeypointsPx) {
        const [x, y] = transform.point(sourcePoint);
        context.beginPath();
        context.fillStyle = "rgba(190, 255, 80, 0.82)";
        context.strokeStyle = "rgba(20, 20, 15, 0.9)";
        context.lineWidth = 1;
        context.arc(x, y, clamp(mediaRect.width / 260, 2.5, 4.5), 0, Math.PI * 2);
        context.fill();
        context.stroke();
      }
    }

    if (overlays.includes("players") && frame) {
      [frame.player_1, frame.player_2].forEach((player, playerIndex) => {
        if (!player?.bbox) return;
        const box = transform.box(player.bbox as [number, number, number, number]);
        const color = playerIndex === 0 ? "#f5f5eb" : "#beff50";
        context.strokeStyle = color;
        context.lineWidth = strokeWidth;
        context.strokeRect(box.x, box.y, box.width, box.height);

        const chipWidth = labelFontSize * 2.3;
        const chipHeight = labelFontSize + 8;
        const chipX = clamp(box.x, mediaRect.x + 2, mediaRect.x + mediaRect.width - chipWidth - 2);
        const aboveY = box.y - chipHeight - 2;
        const chipY = aboveY >= mediaRect.y + 2 ? aboveY : box.y + 2;
        context.fillStyle = "rgba(20, 20, 15, 0.92)";
        context.fillRect(chipX, chipY, chipWidth, chipHeight);
        context.fillStyle = color;
        context.font = `600 ${labelFontSize}px Geist Mono, monospace`;
        context.textBaseline = "middle";
        context.fillText(`P${playerIndex + 1}`, chipX + 6, chipY + chipHeight / 2 + 0.5);
      });
    }

    if (overlays.includes("trajectory") && run.trajectories?.ballTrajectory.length) {
      const tail = run.trajectories.ballTrajectory
        .filter((point) => point.frame_index <= index && point.frame_index >= index - 24 && point.x_px != null && point.y_px != null)
        .sort((a, b) => a.frame_index - b.frame_index);
      for (let pointIndex = 1; pointIndex < tail.length; pointIndex += 1) {
        const previous = tail[pointIndex - 1];
        const current = tail[pointIndex];
        if (previous?.x_px == null || previous.y_px == null || current?.x_px == null || current.y_px == null) continue;
        const [x1, y1] = transform.point([previous.x_px, previous.y_px]);
        const [x2, y2] = transform.point([current.x_px, current.y_px]);
        const alpha = 0.12 + (pointIndex / Math.max(1, tail.length - 1)) * 0.76;
        context.beginPath();
        context.strokeStyle = `rgba(190, 255, 80, ${alpha.toFixed(3)})`;
        context.lineWidth = clamp(mediaRect.width / 700, 1.25, 2.25);
        context.moveTo(x1, y1);
        context.lineTo(x2, y2);
        context.stroke();
      }
    }

    if (overlays.includes("ball") && frame?.ball?.position_px && frame.ball.state !== "MISSING") {
      const [x, y] = transform.point(frame.ball.position_px);
      const radius = clamp(mediaRect.width / 190, 4.5, 7);
      context.beginPath();
      context.fillStyle = frame.ball.state === "INTERPOLATED" ? "rgba(190, 255, 80, 0.36)" : "#beff50";
      context.strokeStyle = "#14140f";
      context.lineWidth = 1.5;
      if (frame.ball.state === "INTERPOLATED" || frame.ball.state === "PREDICTED") context.setLineDash([3, 2]);
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.setLineDash([]);
    }

    if (overlays.includes("events")) {
      const event = run.events.find((item) => Math.abs(item.frame - index) <= 1);
      if (event) {
        const eventPoint = event.ball_position_px ?? frame?.ball?.position_px ?? null;
        const [anchorX, anchorY] = eventPoint
          ? transform.point(eventPoint)
          : ([mediaRect.x + 16, mediaRect.y + 16] as Point2D);
        const label = humanize(event.event_type);
        context.font = `600 ${labelFontSize}px Geist Mono, monospace`;
        const labelWidth = clamp(context.measureText(label).width + 20, 72, mediaRect.width - 20);
        const labelHeight = labelFontSize + 12;
        const labelX = clamp(anchorX + 10, mediaRect.x + 10, mediaRect.x + mediaRect.width - labelWidth - 10);
        const labelY = clamp(anchorY - labelHeight - 10, mediaRect.y + 10, mediaRect.y + mediaRect.height - labelHeight - 10);
        context.fillStyle = "rgba(20, 20, 15, 0.92)";
        context.fillRect(labelX, labelY, labelWidth, labelHeight);
        context.fillStyle = "#beff50";
        context.textBaseline = "middle";
        context.fillText(label, labelX + 10, labelY + labelHeight / 2);
      }
    }

    if (overlays.includes("calls")) {
      const call = run.lineCalls.find((item) => Math.abs(item.bounce_frame - index) <= 1);
      if (call) {
        const [x, y] = transform.point(call.bounce_position_px);
        context.beginPath();
        context.strokeStyle = "#beff50";
        context.lineWidth = 2;
        context.arc(x, y, 10, 0, Math.PI * 2);
        context.moveTo(x - 14, y);
        context.lineTo(x + 14, y);
        context.moveTo(x, y - 14);
        context.lineTo(x, y + 14);
        context.stroke();
      }
    }

    context.restore();
  }, [fps, overlays, run]);

  useEffect(() => {
    drawOverlayRef.current = drawOverlay;
  }, [drawOverlay]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(stage);
    resizeCanvas();
    return () => observer.disconnect();
  }, [resizeCanvas]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onReady = () => hydrateVideoMetadata();
    video.addEventListener("loadedmetadata", onReady);
    video.addEventListener("loadeddata", onReady);
    video.addEventListener("canplay", onReady);
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) requestAnimationFrame(onReady);
    return () => {
      video.removeEventListener("loadedmetadata", onReady);
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("canplay", onReady);
    };
  }, [hydrateVideoMetadata]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !playing) return;
    let running = true;
    let lastPublish = 0;

    const tick = (timestamp: number) => {
      if (!running) return;
      if (active) drawOverlay();
      if (timestamp - lastPublish >= 50) {
        lastPublish = timestamp;
        publishTime();
      }
      if (active && typeof video.requestVideoFrameCallback === "function") {
        frameCallbackRef.current = video.requestVideoFrameCallback(tick);
      } else {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    tick(performance.now());
    return () => {
      running = false;
      if (frameCallbackRef.current != null && typeof video.cancelVideoFrameCallback === "function") {
        video.cancelVideoFrameCallback(frameCallbackRef.current);
      }
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      frameCallbackRef.current = null;
      rafRef.current = null;
    };
  }, [active, drawOverlay, playing, publishTime]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !selectedEvent || externalTime != null) return;
    video.currentTime = selectedEvent.timestamp_s;
    publishTime(true);
    drawOverlay();
  }, [drawOverlay, publishTime, selectedEvent, externalTime]);

  useEffect(() => {
    const onFullscreenChange = () => {
      setFullscreen(document.fullscreenElement === rootRef.current);
      requestAnimationFrame(resizeCanvas);
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, [resizeCanvas]);

  useEffect(() => {
    if (active) requestAnimationFrame(() => {
      resizeCanvas();
      drawOverlay();
    });
  }, [active, drawOverlay, resizeCanvas]);

  const togglePlayback = () => {
    if (transport) { transport.toggle(); return; }
    const video = videoRef.current;
    if (!video || status === "error") return;
    if (video.paused) void video.play();
    else video.pause();
  };

  const seek = (time: number) => {
    if (transport) { transport.seekTime(time); return; }
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = clamp(time, 0, duration || 0);
    publishTime(true);
    drawOverlay();
  };

  const toggleFullscreen = async () => {
    if (!rootRef.current) return;
    if (document.fullscreenElement === rootRef.current) await document.exitFullscreen();
    else await rootRef.current.requestFullscreen();
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement || event.target instanceof HTMLSelectElement) return;
    if (event.key === " ") { event.preventDefault(); togglePlayback(); }
    if (event.key === "ArrowLeft") { event.preventDefault(); seek(currentTime - (event.shiftKey ? 1 : 1 / fps)); }
    if (event.key === "ArrowRight") { event.preventDefault(); seek(currentTime + (event.shiftKey ? 1 : 1 / fps)); }
  };

  if (!run.videoUrl) {
    return (
      <div className="telemetry-empty-state" role="status">
        <strong>No analyzed video selected</strong>
        <span>Select a run with a browser-compatible source video to inspect frame telemetry.</span>
      </div>
    );
  }

  const aspectRatio = sourceSize.width > 0 && sourceSize.height > 0
    ? `${sourceSize.width} / ${sourceSize.height}`
    : undefined;

  return (
    <div
      ref={rootRef}
      className={transport ? "evidence-player evidence-player-controlled" : "evidence-player"}
      onKeyDown={onKeyDown}
      tabIndex={0}
      aria-label="Video evidence player. Space plays or pauses. Arrow keys step frames. Shift and arrow keys seek one second."
    >
      <div ref={stageRef} className="video-stage" style={{ aspectRatio }}>
        <video
          ref={videoRef}
          src={run.videoUrl}
          muted={muted}
          playsInline
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => {
            setPlaying(false);
            publishTime(true);
            drawOverlay();
          }}
          onTimeUpdate={() => publishTime()}
          onLoadedMetadata={hydrateVideoMetadata}
          onLoadedData={() => {
            hydrateVideoMetadata();
            requestAnimationFrame(() => {
              resizeCanvas();
              drawOverlay();
            });
          }}
          onError={() => setStatus("error")}
          onClick={togglePlayback}
          aria-label="Source match video"
        />
        <canvas ref={canvasRef} className="telemetry-canvas" aria-hidden="true" />
        {status === "loading" ? (
          <div className="telemetry-stage-state" role="status">
            <span className="telemetry-loader" aria-hidden="true" />
            <strong>Preparing telemetry view</strong>
            <small>{run.summary.label}</small>
          </div>
        ) : null}
        {status === "error" ? (
          <div className="telemetry-stage-state telemetry-stage-error" role="alert">
            <strong>Unable to load the analyzed video</strong>
            <small>The source may be unavailable or unsupported by this browser.</small>
          </div>
        ) : null}
        {status === "ready" ? (
          <div className="frame-chip" aria-label={`Frame ${frameIndex}, ${currentTime.toFixed(3)} seconds`}>
            <strong>F {frameIndex}</strong>
            <span>{currentTime.toFixed(3)} s</span>
          </div>
        ) : null}
      </div>

      {!transport ? <><div className="timeline-wrap">
        <input
          className="video-scrubber"
          aria-label="Playback position"
          type="range"
          min="0"
          max={duration || 0}
          step={1 / fps}
          value={Math.min(currentTime, duration || 0)}
          onChange={(event) => seek(Number(event.target.value))}
        />
        <div className="semantic-ticks" aria-label="Analysis events">
          {run.events.map((event) => (
            <button
              key={event.event_id}
              type="button"
              className={event.event_id === selectedEventId ? "tick-selected" : ""}
              style={{ left: `${duration > 0 ? (event.timestamp_s / duration) * 100 : 0}%` }}
              onClick={() => {
                onSelectEvent(event.event_id);
                seek(event.timestamp_s);
              }}
              aria-label={`${humanize(event.event_type)} at ${event.timestamp_s.toFixed(2)} seconds`}
            />
          ))}
        </div>
      </div>

      <div className="player-controls">
        <div className="control-cluster">
          <button className="icon-button control-primary" type="button" onClick={togglePlayback} disabled={status !== "ready"} aria-label={playing ? "Pause" : "Play"}>
            {playing ? <Pause size={18} weight="fill" aria-hidden="true" /> : <Play size={18} weight="fill" aria-hidden="true" />}
          </button>
          <button className="icon-button" type="button" onClick={() => seek(0)} disabled={status !== "ready"} aria-label="Restart">
            <ArrowCounterClockwise size={18} aria-hidden="true" />
          </button>
          <button className="icon-button" type="button" onClick={() => seek(currentTime - 1 / fps)} disabled={status !== "ready"} aria-label="Previous frame">
            <CaretLeft size={18} aria-hidden="true" />
          </button>
          <button className="icon-button" type="button" onClick={() => seek(currentTime + 1 / fps)} disabled={status !== "ready"} aria-label="Next frame">
            <CaretRight size={18} aria-hidden="true" />
          </button>
          <span className="time-readout">{formatDuration(currentTime)} / {formatDuration(duration)}</span>
        </div>
        <div className="control-cluster control-options">
          <label className="speed-control">
            <span>Speed</span>
            <select value={speed} onChange={(event) => {
              const next = Number(event.target.value);
              setSpeed(next);
              if (videoRef.current) videoRef.current.playbackRate = next;
            }}>
              {[0.25, 0.5, 1, 1.5, 2].map((value) => <option key={value} value={value}>{value}x</option>)}
            </select>
          </label>
          <button className="icon-button" type="button" onClick={() => setMuted((value) => !value)} aria-label="Mute audio" aria-pressed={muted}>
            {muted ? <SpeakerX size={18} aria-hidden="true" /> : <SpeakerHigh size={18} aria-hidden="true" />}
          </button>
          <button className="icon-button" type="button" onClick={() => void toggleFullscreen()} aria-label={fullscreen ? "Exit fullscreen" : "Enter fullscreen"} aria-pressed={fullscreen}>
            {fullscreen ? <CornersIn size={18} aria-hidden="true" /> : <CornersOut size={18} aria-hidden="true" />}
          </button>
        </div>
      </div>

      </> : null}
      <div className="overlay-controls">
        <span><SlidersHorizontal size={16} aria-hidden="true" /> Overlays</span>
        <ToggleGroup.Root
          type="multiple"
          value={overlays}
          onValueChange={(values) => setOverlays(values as OverlayKey[])}
          aria-label="Video overlay controls"
        >
          {OVERLAY_KEYS.map((key) => (
            <ToggleGroup.Item key={key} value={key} aria-label={`Toggle ${key} overlay`}>
              {humanize(key)}
            </ToggleGroup.Item>
          ))}
        </ToggleGroup.Root>
      </div>
    </div>
  );
}
