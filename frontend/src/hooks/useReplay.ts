import { useCallback, useEffect, useRef, useState } from "react";

export function useReplay(frames: number, fps: number, initialFrame = 0) {
  const lastFrame = Math.max(0, frames - 1);
  const [frame, setFrame] = useState(Math.min(lastFrame, initialFrame));
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [loop, setLoop] = useState(false);
  const playhead = useRef(frame);
  const seek = useCallback((next: number) => {
    const safe = Math.max(0, Math.min(lastFrame, Number.isFinite(next) ? next : 0));
    playhead.current = safe;
    setFrame(safe);
  }, [lastFrame]);

  useEffect(() => {
    if (!playing || frames < 2) return;
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
  }, [fps, frames, lastFrame, loop, playing, rate]);

  const toggle = useCallback(() => {
    if (frames < 2) return;
    if (!playing && playhead.current >= lastFrame) seek(0);
    setPlaying(value => !value);
  }, [frames, lastFrame, playing, seek]);
  const step = (amount: number) => { setPlaying(false); seek(Math.round(playhead.current) + amount); };
  return { frame, playhead, playing, setPlaying, rate, setRate, loop, setLoop, seek, toggle, step, lastFrame, time: frame / fps };
}
export type ReplayTransport = ReturnType<typeof useReplay>;
