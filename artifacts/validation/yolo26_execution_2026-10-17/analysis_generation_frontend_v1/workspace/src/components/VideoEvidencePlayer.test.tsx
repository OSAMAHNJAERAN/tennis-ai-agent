import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { snapshot } from "../data/repository";
import { useReplay } from "../hooks/useReplay";
import { VideoEvidencePlayer } from "./VideoEvidencePlayer";

describe("video evidence player", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: vi.fn(function (this: HTMLMediaElement) {
        this.dispatchEvent(new Event("play"));
        return Promise.resolve();
      }),
    });
    Object.defineProperty(HTMLMediaElement.prototype, "pause", {
      configurable: true,
      value: vi.fn(function (this: HTMLMediaElement) {
        this.dispatchEvent(new Event("pause"));
      }),
    });
  });

  it("supports keyboard playback and overlay toggles", async () => {
    const user = userEvent.setup();
    const run = snapshot.details.phase5_scoring_1!;
    render(<VideoEvidencePlayer run={run} selectedEventId={1} onSelectEvent={vi.fn()} />);
    const player = screen.getByLabelText(/Video evidence player/);
    player.focus();
    await user.keyboard(" ");
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
    const ballToggle = screen.getByRole("button", { name: "Toggle ball overlay" });
    expect(ballToggle).toHaveAttribute("data-state", "on");
    await user.click(ballToggle);
    expect(ballToggle).toHaveAttribute("data-state", "off");
  });

  it('keeps fullscreen available when replay transport owns playback', async () => {
    const user = userEvent.setup();
    const requestFullscreen = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(HTMLElement.prototype, 'requestFullscreen', { configurable: true, value: requestFullscreen });
    const run = snapshot.details.phase5_scoring_1!;
    render(<VideoEvidencePlayer run={run} selectedEventId={null} onSelectEvent={vi.fn()}
      transport={{ time: 0, playing: false, rate: 1, toggle: vi.fn(), seekTime: vi.fn() }} />);
    await user.click(screen.getByRole('button', { name: 'Enter fullscreen' }));
    expect(requestFullscreen).toHaveBeenCalledOnce();
    requestFullscreen.mockRejectedValueOnce(new Error('Unavailable'));
    await user.click(screen.getByRole('button', { name: 'Enter fullscreen' }));
    expect(await screen.findByText('Fullscreen is unavailable. The video can still play here.')).toBeInTheDocument();
  });

  it('holds the presented frame through buffering, pause, timeupdate, resume and overlay changes', () => {
    const pending = new Map<number, VideoFrameRequestCallback>();
    let nextId = 0;
    const request = vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
      .mockImplementation(callback => { pending.set(++nextId, callback); return nextId; });
    const cancel = vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback')
      .mockImplementation(id => { pending.delete(id); });
    const fps = 24000 / 1001;
    const base = snapshot.details.phase5_scoring_1!;
    const run = { ...base, events: [], summary: { ...base.summary, fps, frames: 469 } };
    function Replay() {
      const replay = useReplay(run.summary.frames, fps, 0, 'media');
      return <><output aria-label="Shared replay frame">{replay.frame}</output>
        <VideoEvidencePlayer run={run} selectedEventId={null} onSelectEvent={vi.fn()}
          onTimeChange={replay.observeTime}
          transport={{ time: replay.commandTime, playing: replay.playing, rate: replay.rate,
            seekRevision: replay.seekRevision, toggle: replay.toggle,
            seekTime: time => replay.seek(time * fps), onPlayingChange: replay.setPlaying }} />
      </>;
    }
    function present(time: number) {
      const entry = pending.entries().next().value;
      expect(entry).toBeDefined();
      const [id, callback] = entry!;
      pending.delete(id);
      act(() => callback(0, { mediaTime: time } as VideoFrameCallbackMetadata));
    }
    function expectFrame(frame: number, seconds: string) {
      expect(screen.getByLabelText('Shared replay frame')).toHaveTextContent(String(frame));
      expect(screen.getByLabelText(`Frame ${frame}, ${seconds} seconds`)).toBeInTheDocument();
    }
    try {
      render(<Replay />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 19.56 } });
      fireEvent.loadedMetadata(video);
      fireEvent.click(screen.getByRole('button', { name: 'Play video' }));
      video.currentTime = 1.285404;
      present(30 / fps);
      fireEvent.waiting(video);
      expectFrame(30, '1.251');
      expect(screen.getByText('Buffering video · analysis holds on the presented frame.')).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Pause video' }));
      fireEvent.timeUpdate(video);
      expectFrame(30, '1.251');
      expect(screen.queryByText('Buffering video · analysis holds on the presented frame.')).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Play video' }));
      expectFrame(30, '1.251');
      expect(screen.getByText('Buffering video · analysis holds on the presented frame.')).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Toggle ball overlay' }));
      expectFrame(30, '1.251');
      present(31 / fps);
      fireEvent.playing(video);
      expectFrame(31, '1.293');
      expect(screen.queryByText('Buffering video · analysis holds on the presented frame.')).not.toBeInTheDocument();
    } finally {
      cancel.mockRestore();
      request.mockRestore();
    }
  });

  it.each([
    { name: 'rapid frame steps', keys: [{ key: 'ArrowRight', shiftKey: false }, { key: 'ArrowRight', shiftKey: false }],
      initialFrame: 30, targetFrame: 32, nextKey: 'ArrowLeft', nextFrame: 31, batch: false },
    { name: 'opposing one-second seeks', keys: [{ key: 'ArrowRight', shiftKey: true }, { key: 'ArrowLeft', shiftKey: true }],
      initialFrame: 30, targetFrame: 30, nextKey: 'ArrowRight', nextFrame: 31, batch: false },
    { name: 'frame steps before a transport commit', keys: [{ key: 'ArrowRight', shiftKey: false }, { key: 'ArrowRight', shiftKey: false }],
      initialFrame: 30, targetFrame: 32, nextKey: 'ArrowLeft', nextFrame: 31, batch: true },
    { name: 'upper-bound reversal', keys: [{ key: 'ArrowRight', shiftKey: false }, { key: 'ArrowRight', shiftKey: false }, { key: 'ArrowLeft', shiftKey: false }],
      initialFrame: 468, targetFrame: 467, nextKey: 'ArrowLeft', nextFrame: 466, batch: true },
  ])('accumulates pending keyboard $name and resumes from the decoded acknowledgement', ({ keys, initialFrame, targetFrame, nextKey, nextFrame, batch }) => {
    const pending = new Map<number, VideoFrameRequestCallback>();
    let nextId = 0;
    const request = vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
      .mockImplementation(callback => { pending.set(++nextId, callback); return nextId; });
    const cancel = vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback')
      .mockImplementation(id => { pending.delete(id); });
    const fps = 24000 / 1001;
    const base = snapshot.details.phase5_scoring_1!;
    const run = { ...base, events: [], summary: { ...base.summary, fps, frames: 469 } };
    function Replay() {
      const replay = useReplay(run.summary.frames, fps, 0, 'media');
      return <><output aria-label="Shared replay frame">{replay.frame}</output>
        <VideoEvidencePlayer run={run} selectedEventId={null} onSelectEvent={vi.fn()}
          onTimeChange={replay.observeTime}
          transport={{ time: replay.commandTime, playing: replay.playing, rate: replay.rate,
            seekRevision: replay.seekRevision, toggle: replay.toggle,
            seekTime: time => replay.seek(time * fps), onPlayingChange: replay.setPlaying }} />
      </>;
    }
    function present(frame: number) {
      const entry = pending.entries().next().value;
      expect(entry).toBeDefined();
      const [id, callback] = entry!;
      pending.delete(id);
      act(() => callback(0, { mediaTime: frame / fps } as VideoFrameCallbackMetadata));
    }
    function expectFrame(frame: number) {
      expect(screen.getByLabelText('Shared replay frame')).toHaveTextContent(String(frame));
      expect(screen.getByLabelText(`Frame ${frame}, ${(frame / fps).toFixed(3)} seconds`)).toBeInTheDocument();
    }
    try {
      render(<Replay />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      const player = screen.getByLabelText(/Video evidence player/);
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 19.56 } });
      fireEvent.loadedMetadata(video);
      video.currentTime = initialFrame / fps;
      present(initialFrame);
      expect(video.paused).toBe(true);
      expectFrame(initialFrame);
      if (batch) {
        act(() => keys.forEach(key => fireEvent.keyDown(player, key)));
      } else {
        fireEvent.keyDown(player, keys[0]!);
        fireEvent.seeking(video);
        expect(video.currentTime).toBeCloseTo(initialFrame / fps + (keys[0]!.shiftKey ? 1 : 1 / fps), 5);
        expectFrame(initialFrame);
        // Keep the decoder callback queued so the second key must use the command target.
        fireEvent.keyDown(player, keys[1]!);
      }
      expectFrame(initialFrame);
      expect(video.currentTime).toBeCloseTo(targetFrame / fps, 5);
      fireEvent.seeked(video);
      expectFrame(initialFrame);
      present(targetFrame);
      expectFrame(targetFrame);
      fireEvent.keyDown(player, { key: nextKey });
      expect(video.currentTime).toBeCloseTo(nextFrame / fps, 5);
      expectFrame(targetFrame);
      present(nextFrame);
      expectFrame(nextFrame);
    } finally {
      cancel.mockRestore();
      request.mockRestore();
    }
  });

  it('accumulates standalone frame buttons while the decoder has not presented the seek', () => {
    let callback: VideoFrameRequestCallback | undefined;
    const request = vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
      .mockImplementation(next => { callback = next; return 1; });
    const cancel = vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback').mockImplementation(() => undefined);
    const fps = 24000 / 1001;
    const base = snapshot.details.phase5_scoring_1!;
    const run = { ...base, events: [], summary: { ...base.summary, fps, frames: 469 } };
    function present(frame: number) {
      expect(callback).toBeDefined();
      act(() => callback!(0, { mediaTime: frame / fps } as VideoFrameCallbackMetadata));
    }
    try {
      render(<VideoEvidencePlayer run={run} selectedEventId={null} onSelectEvent={vi.fn()} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 19.56 } });
      fireEvent.loadedMetadata(video);
      video.currentTime = 30 / fps;
      present(30);
      fireEvent.click(screen.getByRole('button', { name: 'Next frame' }));
      fireEvent.click(screen.getByRole('button', { name: 'Next frame' }));
      expect(video.currentTime).toBeCloseTo(32 / fps, 5);
      expect(screen.getByLabelText('Frame 30, 1.251 seconds')).toBeInTheDocument();
      fireEvent.seeked(video);
      expect(screen.getByLabelText('Frame 30, 1.251 seconds')).toBeInTheDocument();
      present(32);
      fireEvent.click(screen.getByRole('button', { name: 'Previous frame' }));
      expect(video.currentTime).toBeCloseTo(31 / fps, 5);
      expect(screen.getByLabelText('Frame 32, 1.335 seconds')).toBeInTheDocument();
      present(31);
      expect(screen.getByLabelText('Frame 31, 1.293 seconds')).toBeInTheDocument();
    } finally {
      cancel.mockRestore();
      request.mockRestore();
    }
  });

  it.each(['keys', 'buttons'] as const)('standalonePendingRecovery reverses from the final source frame after repeated %s', (input) => {
    const pending = new Map<number, VideoFrameRequestCallback>();
    let nextId = 0;
    const request = vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
      .mockImplementation(callback => { pending.set(++nextId, callback); return nextId; });
    const cancel = vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback')
      .mockImplementation(id => { pending.delete(id); });
    const fps = 30;
    const frames = 469;
    const base = snapshot.details.phase5_scoring_1!;
    const run = { ...base, events: [], summary: { ...base.summary, fps, frames } };
    function present(frame: number) {
      const entry = pending.entries().next().value;
      expect(entry).toBeDefined();
      const [id, callback] = entry!;
      pending.delete(id);
      act(() => callback(0, { mediaTime: frame / fps } as VideoFrameCallbackMetadata));
    }
    try {
      render(<VideoEvidencePlayer run={run} selectedEventId={null} onSelectEvent={vi.fn()} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      const player = screen.getByLabelText(/Video evidence player/);
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: frames / fps } });
      fireEvent.loadedMetadata(video);
      video.currentTime = (frames - 1) / fps;
      present(frames - 1);
      act(() => {
        if (input === 'keys') {
          fireEvent.keyDown(player, { key: 'ArrowRight' });
          fireEvent.keyDown(player, { key: 'ArrowRight' });
          fireEvent.keyDown(player, { key: 'ArrowLeft' });
        } else {
          fireEvent.click(screen.getByRole('button', { name: 'Next frame' }));
          fireEvent.click(screen.getByRole('button', { name: 'Next frame' }));
          fireEvent.click(screen.getByRole('button', { name: 'Previous frame' }));
        }
      });
      expect(video.currentTime).toBeCloseTo((frames - 2) / fps, 5);
      expect(screen.getByLabelText(`Frame ${frames - 1}, ${((frames - 1) / fps).toFixed(3)} seconds`)).toBeInTheDocument();
      fireEvent.seeked(video);
      present(frames - 2);
      expect(screen.getByLabelText(`Frame ${frames - 2}, ${((frames - 2) / fps).toFixed(3)} seconds`)).toBeInTheDocument();
    } finally {
      cancel.mockRestore();
      request.mockRestore();
    }
  });

  it('standalonePendingRecovery ignores a pre-command callback delivered after seeked and keeps the pending target', () => {
    const pending = new Map<number, VideoFrameRequestCallback>();
    let nextId = 0;
    const request = vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
      .mockImplementation(callback => { pending.set(++nextId, callback); return nextId; });
    const cancel = vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback')
      .mockImplementation(id => { pending.delete(id); });
    const onTimeChange = vi.fn();
    const fps = 24000 / 1001;
    const base = snapshot.details.phase5_scoring_1!;
    const run = { ...base, events: [], summary: { ...base.summary, fps, frames: 469 } };
    function present(frame: number) {
      const entry = pending.entries().next().value;
      expect(entry).toBeDefined();
      const [id, callback] = entry!;
      pending.delete(id);
      act(() => callback(0, { mediaTime: frame / fps } as VideoFrameCallbackMetadata));
    }
    try {
      render(<VideoEvidencePlayer run={run} selectedEventId={null} onSelectEvent={vi.fn()} onTimeChange={onTimeChange} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 469 / fps } });
      fireEvent.loadedMetadata(video);
      video.currentTime = 30 / fps;
      present(30);
      const obsolete = pending.entries().next().value!;
      // Retain a task already queued by the browser even if cancellation removes its registration.
      pending.delete(obsolete[0]);
      fireEvent.change(screen.getByLabelText('Playback position'), { target: { value: String(90 / fps) } });
      expect(video.currentTime).toBeCloseTo(90 / fps, 5);
      fireEvent.seeked(video);
      act(() => obsolete[1](0, { mediaTime: 31 / fps } as VideoFrameCallbackMetadata));
      expect(screen.getByLabelText('Frame 30, 1.251 seconds')).toBeInTheDocument();
      expect(onTimeChange).toHaveBeenLastCalledWith(30 / fps);
      expect(cancel).toHaveBeenCalledWith(obsolete[0]);
      fireEvent.keyDown(screen.getByLabelText(/Video evidence player/), { key: 'ArrowRight' });
      expect(video.currentTime).toBeCloseTo(91 / fps, 5);
      fireEvent.seeked(video);
      present(91);
      expect(screen.getByLabelText('Frame 91, 3.795 seconds')).toBeInTheDocument();
    } finally {
      cancel.mockRestore();
      request.mockRestore();
    }
  });

  it('observes paused decoded seeks and ignores a callback queued before a seek command', () => {
    const pending = new Map<number, VideoFrameRequestCallback>();
    let nextId = 0;
    const request = vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
      .mockImplementation(callback => { pending.set(++nextId, callback); return nextId; });
    const cancel = vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback')
      .mockImplementation(id => { pending.delete(id); });
    const onTimeChange = vi.fn();
    const fps = 24000 / 1001;
    const base = snapshot.details.phase5_scoring_1!;
    const run = { ...base, events: [], summary: { ...base.summary, fps, frames: 469 } };
    const transport = { time: 30 / fps, playing: false, rate: 1, seekRevision: 0,
      toggle: vi.fn(), seekTime: vi.fn() };
    function present(time: number) {
      const entry = pending.entries().next().value;
      expect(entry).toBeDefined();
      const [id, callback] = entry!;
      pending.delete(id);
      act(() => callback(0, { mediaTime: time } as VideoFrameCallbackMetadata));
    }
    try {
      const { rerender } = render(<VideoEvidencePlayer run={run} selectedEventId={null}
        onSelectEvent={vi.fn()} onTimeChange={onTimeChange} transport={transport} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 19.56 } });
      fireEvent.loadedMetadata(video);
      present(30 / fps);
      const obsolete = pending.entries().next().value!;
      rerender(<VideoEvidencePlayer run={run} selectedEventId={null}
        onSelectEvent={vi.fn()} onTimeChange={onTimeChange}
        transport={{ ...transport, time: 90 / fps, seekRevision: 1 }} />);
      expect(video.currentTime).toBeCloseTo(90 / fps);
      fireEvent.timeUpdate(video);
      fireEvent.seeked(video);
      act(() => obsolete[1](0, { mediaTime: 31 / fps } as VideoFrameCallbackMetadata));
      expect(screen.getByLabelText('Frame 30, 1.251 seconds')).toBeInTheDocument();
      expect(onTimeChange).toHaveBeenLastCalledWith(30 / fps);
      expect(cancel).toHaveBeenCalledWith(obsolete[0]);
      present(90 / fps);
      expect(screen.getByLabelText('Frame 90, 3.754 seconds')).toBeInTheDocument();
      expect(onTimeChange).toHaveBeenLastCalledWith(90 / fps);
    } finally {
      cancel.mockRestore();
      request.mockRestore();
    }
  });

  it('waits for the final decoded frame when ended arrives before its callback', () => {
    const pending = new Map<number, VideoFrameRequestCallback>();
    let nextId = 0;
    const request = vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
      .mockImplementation(callback => { pending.set(++nextId, callback); return nextId; });
    const cancel = vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback')
      .mockImplementation(id => { pending.delete(id); });
    const fps = 24000 / 1001;
    const base = snapshot.details.phase5_scoring_1!;
    const run = { ...base, events: [], summary: { ...base.summary, fps, frames: 469 } };
    function present(time: number) {
      const [id, callback] = pending.entries().next().value!;
      pending.delete(id);
      act(() => callback(0, { mediaTime: time } as VideoFrameCallbackMetadata));
    }
    try {
      render(<VideoEvidencePlayer run={run} selectedEventId={null} onSelectEvent={vi.fn()} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 469 / fps } });
      fireEvent.loadedMetadata(video);
      fireEvent.play(video);
      present(467 / fps);
      video.currentTime = 469 / fps;
      fireEvent.ended(video);
      fireEvent.timeUpdate(video);
      expect(screen.getByRole('button', { name: /^Play$/ })).toBeInTheDocument();
      expect(screen.getByLabelText('Frame 467, 19.478 seconds')).toBeInTheDocument();
      present(468 / fps);
      expect(screen.getByLabelText('Frame 468, 19.519 seconds')).toBeInTheDocument();
    } finally {
      cancel.mockRestore();
      request.mockRestore();
    }
  });

  it('uses media events as the fallback clock when video-frame callbacks are unavailable', () => {
    const onTimeChange = vi.fn();
    render(<VideoEvidencePlayer run={snapshot.details.phase5_scoring_1!} selectedEventId={null}
      onSelectEvent={vi.fn()} onTimeChange={onTimeChange} />);
    const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
    Object.defineProperty(video, 'requestVideoFrameCallback', { configurable: true, value: undefined });
    video.currentTime = 1.285404;
    fireEvent.timeUpdate(video);
    expect(onTimeChange).toHaveBeenLastCalledWith(1.285404);
    video.currentTime = 2;
    fireEvent.pause(video);
    expect(onTimeChange).toHaveBeenLastCalledWith(2);
    video.currentTime = 3;
    fireEvent.seeked(video);
    expect(onTimeChange).toHaveBeenLastCalledWith(3);
  });

  it('ignores a superseded play rejection after the playback rate changes', async () => {
    const rejected: ((reason: unknown) => void)[] = [];
    const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(() =>
      new Promise<void>((_resolve, reject) => { rejected.push(reject); }));
    const onPlayingChange = vi.fn();
    const run = snapshot.details.phase5_scoring_1!;
    const transport = { time: 0, playing: true, rate: 1, toggle: vi.fn(), seekTime: vi.fn(), onPlayingChange };
    try {
      const { rerender } = render(<VideoEvidencePlayer run={run} selectedEventId={null}
        onSelectEvent={vi.fn()} transport={transport} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 20 } });
      fireEvent.loadedMetadata(video);
      rerender(<VideoEvidencePlayer run={run} selectedEventId={null}
        onSelectEvent={vi.fn()} transport={{ ...transport, rate: 2 }} />);
      expect(rejected).toHaveLength(2);
      await act(async () => { rejected[0]!(new Error('Superseded start')); });
      expect(screen.queryByText('Playback did not start. Press play to try again.')).not.toBeInTheDocument();
      expect(onPlayingChange).not.toHaveBeenCalledWith(false);
    } finally { play.mockRestore(); }
  });

  it.each([
    { name: 'AbortError', feedback: false },
    { name: 'NotAllowedError', feedback: true },
  ])('handles a current play rejection named $name with appropriate feedback', async ({ name, feedback }) => {
    let rejectPlay: (reason: unknown) => void = () => undefined;
    const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(() =>
      new Promise<void>((_resolve, reject) => { rejectPlay = reject; }));
    const onPlayingChange = vi.fn();
    try {
      render(<VideoEvidencePlayer run={snapshot.details.phase5_scoring_1!} selectedEventId={null}
        onSelectEvent={vi.fn()} transport={{ time: 0, playing: true, rate: 1,
          toggle: vi.fn(), seekTime: vi.fn(), onPlayingChange }} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 20 } });
      fireEvent.loadedMetadata(video);
      await act(async () => { rejectPlay(new DOMException('Browser play request', name)); });
      if (feedback) {
        expect(screen.getByText('Playback did not start. Press play to try again.')).toBeInTheDocument();
        expect(onPlayingChange).toHaveBeenCalledWith(false);
      } else {
        expect(screen.queryByText('Playback did not start. Press play to try again.')).not.toBeInTheDocument();
        expect(onPlayingChange).not.toHaveBeenCalledWith(false);
      }
    } finally { play.mockRestore(); }
  });

  describe('standalone play rejection lifecycle', () => {
    afterEach(() => vi.restoreAllMocks());

    function prepare({ startsPlaying = false } = {}) {
      let paused = true;
      const pendingFrames = new Map<number, VideoFrameRequestCallback>();
      let frameId = 0;
      const requests: Array<{ resolve: () => void; reject: (reason: unknown) => void }> = [];
      vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
        .mockImplementation(callback => { pendingFrames.set(++frameId, callback); return frameId; });
      vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback')
        .mockImplementation(id => { pendingFrames.delete(id); });
      vi.spyOn(HTMLMediaElement.prototype, 'paused', 'get').mockImplementation(() => paused);
      vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) {
        if (startsPlaying) { paused = false; this.dispatchEvent(new Event('play')); }
        return new Promise<void>((resolve, reject) => requests.push({ resolve, reject }));
      });
      vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) {
        paused = true;
        this.dispatchEvent(new Event('pause'));
      });
      const run = snapshot.details.phase5_scoring_1!;
      const onTimeChange = vi.fn();
      const onSelectEvent = vi.fn();
      const props = { run, selectedEventId: null, onSelectEvent, onTimeChange };
      const rendered = render(<VideoEvidencePlayer {...props} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 20 } });
      fireEvent.loadedMetadata(video);
      const entry = pendingFrames.entries().next().value;
      expect(entry).toBeDefined();
      const [id, callback] = entry!;
      pendingFrames.delete(id);
      video.currentTime = 1;
      act(() => callback(0, { mediaTime: 1 } as VideoFrameCallbackMetadata));
      return { ...rendered, props, video, requests, onTimeChange,
        frameLabel: `Frame ${Math.round(run.summary.fps)}, 1.000 seconds` };
    }

    it.each([
      { name: 'NotAllowedError', error: new DOMException('Blocked by browser', 'NotAllowedError') },
      { name: 'generic error', error: new Error('Playback device unavailable') },
    ])('keeps the presented frame and offers a retry for a current $name', async ({ error }) => {
      const { video, requests, onTimeChange, frameLabel } = prepare();
      fireEvent.click(screen.getByRole('button', { name: 'Play' }));
      video.currentTime = 4;
      await act(async () => requests[0]!.reject(error));
      expect(screen.getByText('Playback did not start. Press play to try again.')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Play' })).toBeEnabled();
      expect(screen.getByLabelText(frameLabel)).toBeInTheDocument();
      expect(onTimeChange).toHaveBeenCalledTimes(1);
      expect(onTimeChange).toHaveBeenLastCalledWith(1);

      fireEvent.click(screen.getByRole('button', { name: 'Play' }));
      await act(async () => { fireEvent.play(video); requests[1]!.resolve(); });
      expect(screen.queryByText('Playback did not start. Press play to try again.')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Pause' })).toBeEnabled();
      expect(screen.getByLabelText(frameLabel)).toBeInTheDocument();
    });

    it('handles a cancelled AbortError without retry feedback or advancing evidence', async () => {
      const { requests, onTimeChange, frameLabel } = prepare();
      fireEvent.click(screen.getByRole('button', { name: 'Play' }));
      await act(async () => requests[0]!.reject(new DOMException('Start cancelled', 'AbortError')));
      expect(screen.queryByText('Playback did not start. Press play to try again.')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Play' })).toBeEnabled();
      expect(screen.getByLabelText(frameLabel)).toBeInTheDocument();
      expect(onTimeChange).toHaveBeenCalledTimes(1);
    });

    it('ignores an earlier request rejection after a newer successful start', async () => {
      const { requests, frameLabel } = prepare({ startsPlaying: true });
      fireEvent.click(screen.getByRole('button', { name: 'Play' }));
      fireEvent.click(screen.getByRole('button', { name: 'Pause' }));
      fireEvent.click(screen.getByRole('button', { name: 'Play' }));
      expect(requests).toHaveLength(2);
      await act(async () => requests[1]!.resolve());
      await act(async () => requests[0]!.reject(new DOMException('Old start blocked', 'NotAllowedError')));
      expect(screen.queryByText('Playback did not start. Press play to try again.')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Pause' })).toBeEnabled();
      expect(screen.getByLabelText(frameLabel)).toBeInTheDocument();
    });

    it.each(['pause', 'inactive', 'source change', 'unmount'] as const)('ignores a request rejection after %s invalidates its playback attempt', async invalidation => {
        const { requests, props, rerender, unmount } = prepare({ startsPlaying: true });
        fireEvent.click(screen.getByRole('button', { name: 'Play' }));
        if (invalidation === 'pause') fireEvent.click(screen.getByRole('button', { name: 'Pause' }));
        if (invalidation === 'inactive') rerender(<VideoEvidencePlayer {...props} active={false} />);
        if (invalidation === 'source change') rerender(<VideoEvidencePlayer {...props}
          run={{ ...props.run, videoUrl: '/different-source.mp4' }} />);
        if (invalidation === 'unmount') unmount();
        await act(async () => requests[0]!.reject(new DOMException('Old start blocked', 'NotAllowedError')));
        expect(screen.queryByText('Playback did not start. Press play to try again.')).not.toBeInTheDocument();
      });
  });

  it('offers generic detected people separately from player roles and leaves it off initially', async () => {
    const user = userEvent.setup();
    const run = snapshot.details.phase5_scoring_1!;
    render(<VideoEvidencePlayer run={{ ...run, court: null, playerTracking: null, imagePeople: {
      coordinateSystem: 'normalized_video_pixels', identityScope: 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID',
      participantSemantics: 'May include spectators.', accuracyValidated: false, missingSemantics: 'Null means unavailable.',
      frames: [{ frame_index: 0, timestamp_seconds: 0, people: [{ bbox: [1, 2, 10, 20], confidence: .8,
        sourceTrackId: null, courtRole: null, participantStatus: 'UNKNOWN' }] }],
    } }} selectedEventId={null} onSelectEvent={vi.fn()} />);
    const peopleToggle = screen.getByRole('button', { name: 'Toggle detected people overlay' });
    expect(peopleToggle).toBeEnabled();
    expect(peopleToggle).toHaveTextContent('Detected people');
    expect(peopleToggle).toHaveAttribute('data-state', 'off');
    expect(screen.getByRole('button', { name: 'Toggle players overlay' })).toHaveAttribute('data-state', 'on');
    await user.click(peopleToggle);
    expect(peopleToggle).toHaveAttribute('data-state', 'on');
    expect(screen.getByText(/Detected people may include spectators/)).toBeInTheDocument();
  });

  it('withholds the generic people overlay when no raw observations were exported', () => {
    render(<VideoEvidencePlayer run={snapshot.details.phase5_scoring_1!} selectedEventId={null} onSelectEvent={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Toggle detected people overlay' })).toBeDisabled();
  });

  it.each([false, true])('withholds player overlays without exported boxes even when the players capability is %s', async players => {
    const user = userEvent.setup();
    const run = snapshot.details.phase5_scoring_1!;
    render(<VideoEvidencePlayer run={{ ...run,
      summary: { ...run.summary, capabilities: { ...run.summary.capabilities, players } },
      detections: { metadata: null, frames: [{ frame_index: 0, player_1: null, player_2: null }] },
    }} selectedEventId={null} onSelectEvent={vi.fn()} />);
    const toggle = screen.getByRole('button', { name: 'Toggle players overlay' });
    expect(toggle).toBeDisabled();
    // Unavailability does not overwrite the user's default overlay preference.
    expect(toggle).toHaveAttribute('data-state', 'on');
    await user.click(toggle);
    expect(toggle).toHaveAttribute('data-state', 'on');
  });

  it('withholds player overlays when the entire detection section is unavailable', () => {
    const run = snapshot.details.phase5_scoring_1!;
    render(<VideoEvidencePlayer run={{ ...run, detections: null }} selectedEventId={null} onSelectEvent={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Toggle players overlay' })).toBeDisabled();
  });

  it('keeps sparse image boxes usable without court or player world tracking and preserves preferences', async () => {
    const user = userEvent.setup();
    const base = snapshot.details.phase5_scoring_1!;
    const run = { ...base, court: null, playerTracking: null,
      summary: { ...base.summary, capabilities: { ...base.summary.capabilities, players: false, court: false } },
      detections: { metadata: null, frames: [
        { frame_index: 0, player_1: null, player_2: null },
        { frame_index: 1, player_1: null, player_2: { bbox: [1, 2, 10, 20], confidence: .8 } },
        { frame_index: 2, player_1: null, player_2: null },
      ] },
    };
    const unchanged = structuredClone(run);
    const { rerender } = render(<VideoEvidencePlayer run={run} selectedEventId={null} onSelectEvent={vi.fn()} />);
    const toggle = screen.getByRole('button', { name: 'Toggle players overlay' });
    // The initial frame is missing, but a later actual player-2 box is usable.
    expect(toggle).toBeEnabled();
    expect(toggle).toHaveAttribute('data-state', 'on');
    const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
    Object.defineProperty(video, 'requestVideoFrameCallback', { configurable: true, value: undefined });
    video.currentTime = 2 / run.summary.fps;
    fireEvent.timeUpdate(video);
    expect(toggle).toBeEnabled();
    await user.click(toggle);
    expect(toggle).toHaveAttribute('data-state', 'off');
    rerender(<VideoEvidencePlayer run={{ ...run, detections: null }} selectedEventId={null} onSelectEvent={vi.fn()} />);
    expect(toggle).toBeDisabled();
    rerender(<VideoEvidencePlayer run={run} selectedEventId={null} onSelectEvent={vi.fn()} />);
    expect(toggle).toBeEnabled();
    expect(toggle).toHaveAttribute('data-state', 'off');
    await user.click(toggle);
    expect(toggle).toHaveAttribute('data-state', 'on');
    expect(run).toEqual(unchanged);
  });

  it('offers UNKNOWN people without presenting them as exported players', () => {
    const run = snapshot.details.phase5_scoring_1!;
    render(<VideoEvidencePlayer run={{ ...run,
      detections: { metadata: null, frames: [{ frame_index: 0, player_1: null, player_2: null }] },
      imagePeople: {
        coordinateSystem: 'normalized_video_pixels', identityScope: 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID',
        participantSemantics: 'May include spectators.', accuracyValidated: false, missingSemantics: 'Null means unavailable.',
        frames: [{ frame_index: 0, timestamp_seconds: 0, people: [{ bbox: [1, 2, 10, 20], confidence: .8,
          sourceTrackId: null, courtRole: null, participantStatus: 'UNKNOWN' }] }],
      },
    }} selectedEventId={null} onSelectEvent={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Toggle detected people overlay' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Toggle players overlay' })).toBeDisabled();
  });

  it.each([
    [1, 2, 1, 20], [10, 2, 1, 20], [1, 2, Number.POSITIVE_INFINITY, 20], [1, 2, Number.NaN, 20], [1, 2, 10],
  ])('withholds player overlays when only malformed box coordinates %j were exported', (...bbox) => {
    const run = snapshot.details.phase5_scoring_1!;
    render(<VideoEvidencePlayer run={{ ...run,
      summary: { ...run.summary, capabilities: { ...run.summary.capabilities, players: true } },
      detections: { metadata: null, frames: [{ frame_index: 0, player_1: { bbox }, player_2: null }] },
    }} selectedEventId={null} onSelectEvent={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Toggle players overlay' })).toBeDisabled();
  });

  it.each([
    { bbox: [1, 2, 10, 20], expected: [6, 12, 54, 108] },
    { bbox: [-5, -4, 10, 20], expected: [-30, -24, 90, 144] },
  ])('skips malformed boxes and draws sparse evidence %j within the media clip', ({ bbox, expected }) => {
    const context = {
      setTransform: vi.fn(), clearRect: vi.fn(), save: vi.fn(), beginPath: vi.fn(), rect: vi.fn(),
      clip: vi.fn(), strokeRect: vi.fn(), fillRect: vi.fn(), fillText: vi.fn(), restore: vi.fn(),
    };
    const canvas = vi.spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockReturnValue(context as unknown as CanvasRenderingContext2D);
    const bounds = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockReturnValue({ width: 600, height: 600, x: 0, y: 0, left: 0, top: 0, right: 600, bottom: 600, toJSON: () => ({}) });
    const animation = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => { callback(0); return 1; });
    try {
      const base = snapshot.details.phase5_scoring_1!;
      render(<VideoEvidencePlayer run={{ ...base, court: null, poseTracking: null, racketTracking: null,
        trajectories: null, imagePeople: null, events: [],
        detections: { metadata: null, frames: [
          { frame_index: 0, player_1: { bbox: [1, 2, 1, 20] }, player_2: null },
          { frame_index: 1, player_1: null, player_2: { bbox } },
        ] },
      }} selectedEventId={null} onSelectEvent={vi.fn()} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperty(video, 'requestVideoFrameCallback', { configurable: true, value: undefined });
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 1 } });
      fireEvent.loadedMetadata(video);
      expect(screen.getByRole('button', { name: 'Toggle players overlay' })).toBeEnabled();
      expect(context.strokeRect).not.toHaveBeenCalled();
      video.currentTime = 1 / base.summary.fps;
      fireEvent.timeUpdate(video);
      expect(context.rect).toHaveBeenCalledWith(0, 0, 600, 600);
      expect(context.clip).toHaveBeenCalled();
      expect(context.strokeRect).toHaveBeenCalledWith(...expected);
    } finally {
      animation.mockRestore();
      bounds.mockRestore();
      canvas.mockRestore();
    }
  });
});

