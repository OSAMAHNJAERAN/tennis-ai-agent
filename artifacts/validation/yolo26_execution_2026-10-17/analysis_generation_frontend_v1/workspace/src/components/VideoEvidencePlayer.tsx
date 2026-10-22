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
import { createBallTrailIndex } from '../lib/ballTrail';
import { courtOverlayPoints } from '../lib/courtOverlay';
import { createLandingZoneIndex, DEFAULT_LANDING_PRESENTATION, type LandingPresentation } from '../lib/landingZones';
import {
  canvasBitmapSize,
  createCoordinateTransform,
  type CoordinateTransform,
  type Point2D,
} from "../lib/telemetryGeometry";
import { formatDuration, humanize } from "../lib/format";

type OverlayKey = "people" | "players" | "ball" | "trajectory" | "skeleton" | "racket" | "court" | "events";
type VideoStatus = "loading" | "ready" | "error";

type VideoMetadata = {
  width: number;
  height: number;
  duration: number;
};

const OVERLAY_KEYS: OverlayKey[] = ["people", "players", "ball", "trajectory", "skeleton", "racket", "court", "events"];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function decoderSeekTime(time: number) {
  // Decoder timestamps are quantized. A nominal boundary represented just below
  // its timestamp can select the preceding frame; one microsecond crosses it.
  return time > 0 ? time + 0.000001 : time;
}

function isPlayerBox(value: unknown): value is [number, number, number, number] {
  if (!Array.isArray(value) || value.length !== 4 ||
      !value.every(coordinate => typeof coordinate === 'number' && Number.isFinite(coordinate))) return false;
  const [left, top, right, bottom] = value;
  return right > left && bottom > top;
}

export function VideoEvidencePlayer({
  run,
  selectedEventId,
  onSelectEvent,
  active = true,
  onTimeChange,
  onMetadata,
  transport,
  landingPresentation = DEFAULT_LANDING_PRESENTATION,
}: {
  run: AnalysisRun;
  selectedEventId: number | null;
  onSelectEvent: (id: number | null) => void;
  active?: boolean;
  onTimeChange?: (time: number) => void;
  onMetadata?: (metadata: VideoMetadata) => void;
  transport?: { time: number; playing: boolean; rate: number; loop?: boolean; seekRevision?: number; toggle: () => void; seekTime: (time: number) => void; onPlayingChange?: (playing: boolean) => void };
  landingPresentation?: LandingPresentation;
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
  const presentationTimeRef = useRef(0);
  const transportRef = useRef(transport);
  const seekCommandRef = useRef({ revision: transport?.seekRevision, time: transport?.time ?? 0 });
  const pendingSeekTimeRef = useRef<number | null>(transport?.time ?? null);
  const standalonePlayRequestRef = useRef(0);
  useEffect(() => { transportRef.current = transport; }, [transport]);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [status, setStatus] = useState<VideoStatus>(run.videoUrl ? "loading" : "error");
  const [sourceSize, setSourceSize] = useState({ width: 0, height: 0 });
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(run.summary.durationSeconds);
  const [speed, setSpeed] = useState(1);
  const [buffering, setBuffering] = useState(false);
  const [seeking, setSeeking] = useState(false);
  const [localSeekRevision, setLocalSeekRevision] = useState(0);
  const [playbackError, setPlaybackError] = useState('');
  const [overlays, setOverlays] = useState<OverlayKey[]>(["players", "ball", "trajectory", "skeleton", "racket"]);
  // This control draws image boxes, which can survive unavailable court/world
  // tracking. Clip-wide evidence keeps it usable through individual frame gaps.
  const playerBoxesAvailable = useMemo(() => run.detections?.frames.some(frame =>
    isPlayerBox(frame.player_1?.bbox) || isPlayerBox(frame.player_2?.bbox)) ?? false, [run.detections]);
  const ballTrailAtFrame = useMemo(() => createBallTrailIndex(run.trajectories?.ballTrajectory ?? []), [run.trajectories?.ballTrajectory]);
  const landingZonesAtTime = useMemo(() => createLandingZoneIndex(run), [run]);
  const fps = run.summary.fps;
  const frameIndex = currentFrameIndex(currentTime, fps, run.summary.frames);
  const selectedEvent = useMemo(
    () => run.events.find((event) => event.event_id === selectedEventId) ?? null,
    [run.events, selectedEventId],
  );

  const externalTime = transport?.time;
  const externalPlaying = transport?.playing;
  const externalRate = transport?.rate;
  const seekRevision = transport?.seekRevision;
  const invalidateStandalonePlay = useCallback(() => { ++standalonePlayRequestRef.current; }, []);
  useEffect(() => {
    invalidateStandalonePlay();
    if (!active && !transportRef.current) videoRef.current?.pause();
    return invalidateStandalonePlay;
  }, [active, run.videoUrl, invalidateStandalonePlay]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || externalPlaying == null || status !== "ready") return;
    video.playbackRate = externalRate ?? 1;
    video.loop = transportRef.current?.loop ?? false;
    let disposed = false;
    if (externalPlaying && active) void video.play().catch((error: unknown) => {
      if (disposed || (error instanceof DOMException && error.name === 'AbortError')) return;
      setPlaybackError('Playback did not start. Press play to try again.');
      transportRef.current?.onPlayingChange?.(false);
    });
    else video.pause();
    return () => { disposed = true; };
  }, [active, externalPlaying, externalRate, status, transport?.loop]);

  useEffect(() => {
    onTimeChangeRef.current = onTimeChange;
  }, [onTimeChange]);

  const publishTime = useCallback((force = false, presentedTime?: number) => {
    const video = videoRef.current;
    if (!video) return;
    // The playback clock can move ahead of the last decoded image, particularly
    // during buffering or pause. On this path only a frame callback may advance
    // evidence; ordinary media events serve browsers without that callback API.
    if (presentedTime == null && typeof video.requestVideoFrameCallback === 'function') return;
    if (presentedTime == null && video.seeking) return;
    const nextTime = presentedTime ?? video.currentTime;
    presentationTimeRef.current = nextTime;
    // A relative input composes with the pending command until the decoder
    // presents its result, including an acknowledgement of the same frame.
    if (!video.seeking) pendingSeekTimeRef.current = null;
    if (!force && nextTime === lastPublishedTimeRef.current) return;
    lastPublishedTimeRef.current = nextTime;
    setCurrentTime(nextTime);
    onTimeChangeRef.current?.(nextTime);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !transportRef.current) return;
    if (seekCommandRef.current.revision !== seekRevision) {
      seekCommandRef.current = { revision: seekRevision, time: transportRef.current.time };
      pendingSeekTimeRef.current = seekCommandRef.current.time;
    }
    let disposed = false;
    // Presentation observations never seek the decoder. Only transport commands do.
    // Browsers can retain this as the default start position before metadata arrives.
    // A decoder that rejects an early seek receives the same command when it is ready.
    try { video.currentTime = decoderSeekTime(seekCommandRef.current.time); }
    catch {
      if (status === 'ready') queueMicrotask(() => {
        if (!disposed) setPlaybackError('Seeking is unavailable for this video.');
      });
    }
    return () => { disposed = true; };
  }, [seekRevision, status]);

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

    const index = currentFrameIndex(presentationTimeRef.current, fps, run.summary.frames);
    const frame = run.detections?.frames[index];
    const strokeWidth = clamp(mediaRect.width / 560, 1.25, 2);
    const labelFontSize = clamp(mediaRect.width / 72, 10, 13);
    context.lineJoin = "round";
    context.lineCap = "round";

    if (landingPresentation.visible) {
      for (const landing of landingZonesAtTime(presentationTimeRef.current, index, landingPresentation.partition)) {
        const color = landing.status === 'inside' ? '120, 218, 163' : '241, 195, 110';
        for (const zone of landing.zones) {
          context.beginPath();
          zone.polygon_px.forEach((point, corner) => {
            const [x, y] = transform.point(point);
            if (corner) context.lineTo(x, y); else context.moveTo(x, y);
          });
          context.closePath();
          context.fillStyle = `rgba(${color}, 0.22)`;
          context.strokeStyle = `rgba(${color}, 0.8)`;
          context.lineWidth = strokeWidth;
          context.fill(); context.stroke();
        }
        const [x, y] = transform.point(landing.contact_position_px);
        context.beginPath(); context.arc(x, y, 4, 0, 2 * Math.PI);
        context.fillStyle = `rgb(${color})`; context.fill();
        context.font = `600 ${labelFontSize}px Geist Mono, monospace`;
        const text = landing.label;
        const labelWidth = Math.min(context.measureText(text).width + 16, mediaRect.width - 20);
        context.fillStyle = 'rgba(20, 20, 15, 0.92)';
        context.fillRect(mediaRect.x + 10, mediaRect.y + mediaRect.height - labelFontSize - 26, labelWidth, labelFontSize + 16);
        context.fillStyle = `rgb(${color})`; context.textBaseline = 'middle';
        context.fillText(text, mediaRect.x + 18, mediaRect.y + mediaRect.height - labelFontSize / 2 - 18, labelWidth - 16);
      }
    }

    if (overlays.includes("court")) {
      for (const sourcePoint of courtOverlayPoints(run, index)) {
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

    if (overlays.includes('people')) {
      const people = run.imagePeople?.frames[index]?.people;
      for (const person of people ?? []) {
        const box = transform.box(person.bbox);
        context.strokeStyle = '#f4d3a1';
        context.lineWidth = strokeWidth;
        context.setLineDash([5, 4]);
        context.strokeRect(box.x, box.y, box.width, box.height);
        context.setLineDash([]);
        context.font = `600 ${labelFontSize}px Geist Mono, monospace`;
        context.textBaseline = 'top';
        context.fillStyle = '#f4d3a1';
        context.fillText('Person', box.x + 3, box.y + 3);
      }
    }

    if (overlays.includes("players") && frame) {
      [frame.player_1, frame.player_2].forEach((player, playerIndex) => {
        const bounds = player?.bbox;
        if (!isPlayerBox(bounds)) return;
        const box = transform.box(bounds);
        const color = playerIndex === 0 ? "#83b99c" : "#86b7dd";
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

    if (overlays.includes('skeleton') && run.poseTracking) {
      const pose = run.poseTracking.frames[index];
      [pose?.player_1, pose?.player_2].forEach((joints, playerIndex) => {
        if (!joints) return;
        context.strokeStyle = playerIndex === 0 ? '#83b99c' : '#86b7dd';
        context.fillStyle = context.strokeStyle;
        context.lineWidth = strokeWidth;
        for (const [a, b] of run.poseTracking!.boneTopology) {
          const first = joints[a]; const second = joints[b];
          if (!first?.visible || !second?.visible || first.x_px == null || first.y_px == null || second.x_px == null || second.y_px == null) continue;
          const start = transform.point([first.x_px, first.y_px]); const end = transform.point([second.x_px, second.y_px]);
          context.beginPath(); context.moveTo(...start); context.lineTo(...end); context.stroke();
        }
        for (const joint of joints) {
          if (!joint.visible || joint.x_px == null || joint.y_px == null) continue;
          const [x, y] = transform.point([joint.x_px, joint.y_px]);
          context.beginPath(); context.arc(x, y, strokeWidth * 1.4, 0, Math.PI * 2); context.fill();
        }
      });
    }

    if (overlays.includes('racket')) {
      const racketFrame = run.racketTracking?.frames[index];
      [racketFrame?.player_1, racketFrame?.player_2].forEach((racket, playerIndex) => {
        if (racket?.state !== 'DETECTED' || !racket.bbox_xyxy) return;
        const box = transform.box(racket.bbox_xyxy);
        context.strokeStyle = playerIndex === 0 ? '#83b99c' : '#86b7dd'; context.lineWidth = strokeWidth;
        context.setLineDash([4, 3]); context.strokeRect(box.x, box.y, box.width, box.height); context.setLineDash([]);
      });
    }

    if (overlays.includes("trajectory") && run.trajectories?.ballTrajectory.length) {
      const tail = ballTrailAtFrame(index);
      for (let pointIndex = 1; pointIndex < tail.length; pointIndex += 1) {
        const previous = tail[pointIndex - 1];
        const current = tail[pointIndex];
        if (previous?.x_px == null || previous.y_px == null || current?.x_px == null || current.y_px == null || current.frame_index !== previous.frame_index + 1) continue;
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

    context.restore();
  }, [ballTrailAtFrame, fps, overlays, run, landingPresentation, landingZonesAtTime]);

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
    if (!video || !active) return;
    const requestVideoFrame = typeof video.requestVideoFrameCallback === 'function'
      ? video.requestVideoFrameCallback.bind(video) : null;
    if (!requestVideoFrame && !playing) return;
    let running = true;
    const tick = (_timestamp: number, metadata?: VideoFrameCallbackMetadata) => {
      if (!running) return;
      publishTime(false, metadata?.mediaTime);
      if (active) drawOverlay();
      if (requestVideoFrame) {
        frameCallbackRef.current = requestVideoFrame(tick);
      } else {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    // Keep listening while paused: a completed seek or the final decoded frame
    // still needs to update evidence. Subscribing itself is not an observation.
    if (requestVideoFrame) frameCallbackRef.current = requestVideoFrame(tick);
    else tick(performance.now());
    return () => {
      running = false;
      if (frameCallbackRef.current != null && typeof video.cancelVideoFrameCallback === "function") {
        video.cancelVideoFrameCallback(frameCallbackRef.current);
      }
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      frameCallbackRef.current = null;
      rafRef.current = null;
    };
  }, [active, drawOverlay, localSeekRevision, playing, publishTime, seekRevision]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !selectedEvent || externalTime != null) return;
    const limit = Math.min(duration || 0, Math.max(0, (run.summary.frames - 1) / fps));
    const target = clamp(selectedEvent.timestamp_s, 0, limit);
    pendingSeekTimeRef.current = target;
    video.currentTime = Math.min(duration || 0, decoderSeekTime(target));
    setLocalSeekRevision(revision => revision + 1);
    publishTime(true);
    drawOverlay();
  }, [drawOverlay, duration, externalTime, fps, publishTime, run.summary.frames, selectedEvent]);

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
    if (!video || !active || status === "error") return;
    const request = ++standalonePlayRequestRef.current;
    if (video.paused) void video.play().catch((error: unknown) => {
      if (request !== standalonePlayRequestRef.current || (error instanceof DOMException && error.name === 'AbortError')) return;
      setPlaying(false);
      setPlaybackError('Playback did not start. Press play to try again.');
    });
    else video.pause();
  };

  const seek = (time: number) => {
    const lastFrameTime = Math.max(0, (run.summary.frames - 1) / fps);
    const limit = transport ? lastFrameTime : Math.min(duration || 0, lastFrameTime);
    const target = clamp(time, 0, limit);
    // Update synchronously: another key event can arrive before React commits
    // the transport command or the video presents a new image.
    pendingSeekTimeRef.current = target;
    if (transport) { transport.seekTime(target); return; }
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.min(duration || 0, decoderSeekTime(target));
    setLocalSeekRevision(revision => revision + 1);
    publishTime(true);
    drawOverlay();
  };

  const seekRelative = (delta: number) => seek((pendingSeekTimeRef.current ?? presentationTimeRef.current) + delta);

  const toggleFullscreen = async () => {
    if (!rootRef.current) return;
    try {
      if (document.fullscreenElement === rootRef.current) await document.exitFullscreen();
      else await rootRef.current.requestFullscreen();
    } catch { setPlaybackError('Fullscreen is unavailable. The video can still play here.'); }
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement || event.target instanceof HTMLSelectElement) return;
    if (event.key === " ") { event.preventDefault(); togglePlayback(); }
    if (event.key === "ArrowLeft") { event.preventDefault(); seekRelative(-(event.shiftKey ? 1 : 1 / fps)); }
    if (event.key === "ArrowRight") { event.preventDefault(); seekRelative(event.shiftKey ? 1 : 1 / fps); }
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
          onPlay={() => { setPlaying(true); setPlaybackError(''); if (active) transportRef.current?.onPlayingChange?.(true); }}
          onPlaying={() => setBuffering(false)}
          onWaiting={() => setBuffering(true)}
          onStalled={() => {
            if (videoRef.current && videoRef.current.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) setBuffering(true);
          }}
          onEnded={() => { ++standalonePlayRequestRef.current; setPlaying(false); transportRef.current?.onPlayingChange?.(false); publishTime(true); }}
          onPause={() => {
            ++standalonePlayRequestRef.current;
            setPlaying(false);
            if (active) transportRef.current?.onPlayingChange?.(false);
            publishTime(true);
            drawOverlay();
          }}
          onSeeking={() => { setSeeking(true); setPlaybackError(''); }}
          onSeeked={() => { setSeeking(false); publishTime(true); drawOverlay(); }}
          onTimeUpdate={() => { if (typeof videoRef.current?.requestVideoFrameCallback !== 'function') { publishTime(); drawOverlay(); } }}
          onLoadedMetadata={hydrateVideoMetadata}
          onLoadedData={() => {
            hydrateVideoMetadata();
            requestAnimationFrame(() => {
              resizeCanvas();
              drawOverlay();
            });
          }}
          onError={() => { ++standalonePlayRequestRef.current; setStatus("error"); transportRef.current?.onPlayingChange?.(false); }}
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

      {seeking ? <p role="status" className="upload-hint">Seeking video · analysis holds on the presented frame.</p>
        : buffering && playing ? <p role="status" className="upload-hint">Buffering video · analysis holds on the presented frame.</p> : null}
      {playbackError ? <p role="status" className="upload-hint">{playbackError}</p> : null}

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
          <button className="icon-button" type="button" onClick={() => seekRelative(-1 / fps)} disabled={status !== "ready"} aria-label="Previous frame">
            <CaretLeft size={18} aria-hidden="true" />
          </button>
          <button className="icon-button" type="button" onClick={() => seekRelative(1 / fps)} disabled={status !== "ready"} aria-label="Next frame">
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
      {transport ? <div className="player-controls">
        <span className="time-readout">{formatDuration(currentTime)} / {formatDuration(duration)}</span>
        <div className="control-cluster control-options">
          <button className="icon-button control-primary" type="button" onClick={togglePlayback} disabled={status !== 'ready'} aria-label={playing ? 'Pause video' : 'Play video'}>
            {playing ? <Pause size={18} weight="fill" aria-hidden="true" /> : <Play size={18} weight="fill" aria-hidden="true" />}
          </button>
          <button className="icon-button" type="button" onClick={() => void toggleFullscreen()} aria-label={fullscreen ? 'Exit fullscreen' : 'Enter fullscreen'} aria-pressed={fullscreen}>
            {fullscreen ? <CornersIn size={18} aria-hidden="true" /> : <CornersOut size={18} aria-hidden="true" />}
          </button>
        </div>
      </div> : null}
      <div className="overlay-controls">
        <span><SlidersHorizontal size={16} aria-hidden="true" /> Overlays</span>
        <ToggleGroup.Root
          type="multiple"
          value={overlays}
          onValueChange={(values) => setOverlays(values as OverlayKey[])}
          aria-label="Video overlay controls"
        >
          {OVERLAY_KEYS.filter(key => key !== 'events' || run.summary.capabilities.events).map((key) => (
            <ToggleGroup.Item key={key} value={key} disabled={(key === 'players' && !playerBoxesAvailable) || (key === 'people' && !run.imagePeople?.frames.some(frame => frame.people?.length)) || (key === 'skeleton' && (!run.summary.capabilities.pose || !run.poseTracking)) || (key === 'racket' && (!run.summary.capabilities.racket || !run.racketTracking)) || (key === 'court' && !run.court?.isValid)} aria-label={key === 'people' ? 'Toggle detected people overlay' : `Toggle ${key} overlay`}>
              {key === 'people' ? 'Detected people' : humanize(key)}
            </ToggleGroup.Item>
          ))}
        </ToggleGroup.Root>
      </div>
      {overlays.includes('people') ? <p className="upload-hint">Detected people may include spectators, officials and false detections. Active participants are not verified.</p> : null}
      {!run.poseTracking || !run.racketTracking ? <p className="upload-hint">{!run.poseTracking ? 'Skeleton observations unavailable. ' : ''}{!run.racketTracking ? 'Racket observations unavailable.' : ''}</p> : null}
    </div>
  );
}
