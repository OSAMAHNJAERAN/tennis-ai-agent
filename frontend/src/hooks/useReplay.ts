import { readPreference } from "../lib/preferences";
import { useCallback, useEffect, useRef, useState } from "react";

export function useReplay(frames: number, fps: number, initialFrame = 0, clock: 'simulation' | 'media' = 'simulation') {
  const lastFrame = Math.max(0, frames - 1);
  const [frame, setFrame] = useState(Math.min(lastFrame, initialFrame));
  const [mediaFrame, setMediaFrame] = useState(Math.min(lastFrame, initialFrame));
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(() => { const saved = Number(readPreference("astra.replay.rate", "1")); return [0.25, 0.5, 1, 1.5, 2].includes(saved) ? saved : 1; });
  const [loop, setLoop] = useState(false);
  const [seekRevision, setSeekRevision] = useState(0);
  const [requestedFrame, setRequestedFrame] = useState<number | null>(null);
  const requestedFrameRef = useRef<number | null>(null);
  const playhead = useRef(frame);
  const observeFrame = useCallback((next: number) => {
    const safe = Math.max(0, Math.min(lastFrame, Number.isFinite(next) ? next : 0));
    playhead.current = safe;
    setFrame(safe);
    requestedFrameRef.current = null;
    setRequestedFrame(null);
  }, [lastFrame]);
  const seek = useCallback((next: number) => {
    if (clock === 'media') {
      // A requested position is a decoder command, not a presented image.
      const safe = Math.max(0, Math.min(lastFrame, Number.isFinite(next) ? next : 0));
      requestedFrameRef.current = safe;
      setRequestedFrame(safe);
    } else observeFrame(next);
    setSeekRevision(value => value + 1);
  }, [clock, lastFrame, observeFrame]);
  const observeTime = useCallback((seconds: number) => {
    const next = Math.max(0, Math.min(lastFrame, Number.isFinite(seconds) ? Math.round(seconds * fps) : 0));
    setMediaFrame(next);
    observeFrame(next);
  }, [fps, lastFrame, observeFrame]);

  useEffect(() => {
    if (clock === 'media' || !playing || frames < 2) return;
    let raf = 0;
    let previous: number | null = null;
    let published = 0;
    const tick = (time: number) => {
      if (document.hidden) { previous = null; raf = requestAnimationFrame(tick); return; }
      const delta = previous == null ? 0 : Math.min(0.1, (time - previous) / 1000);
      previous = time;
      let next = playhead.current + delta * fps * rate;
      if (next >= lastFrame) {
        next = loop ? 0 : lastFrame;
        if (!loop) { playhead.current = next; setFrame(next); setPlaying(false); return; }
      }
      playhead.current = next;
      if (time - published >= 1000 / Math.min(30, fps)) { setFrame(next); published = time; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [clock, fps, frames, lastFrame, loop, playing, rate]);

  const toggle = useCallback(() => {
    if (frames < 2) return;
    if (!playing && playhead.current >= lastFrame) seek(0);
    setPlaying(value => !value);
  }, [frames, lastFrame, playing, seek]);
  const step = (amount: number) => { setPlaying(false); seek(Math.round(requestedFrameRef.current ?? playhead.current) + amount); };
  // Returning from simulation retains the last video observation until decoding
  // reaches the requested simulation position. Both views then share that frame.
  const displayedFrame = clock === 'media' ? mediaFrame : frame;
  return { frame: displayedFrame, playhead, playing, setPlaying, rate, setRate, loop, setLoop, seek, seekRevision, observeTime, toggle, step, lastFrame, time: displayedFrame / fps, commandTime: (requestedFrame ?? frame) / fps };
}
export type ReplayTransport = ReturnType<typeof useReplay>;
