import { act, fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { snapshot } from '../../data/repository';
import { TelemetryDashboard } from './TelemetryDashboard';

describe('decoded-frame replay integration', () => {
  it('holds shared evidence during seeks and commands the decoder on view and source changes', () => {
    const pending = new Map<number, VideoFrameRequestCallback>();
    let nextId = 0;
    const request = vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
      .mockImplementation(callback => { pending.set(++nextId, callback); return nextId; });
    const cancel = vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback')
      .mockImplementation(id => { pending.delete(id); });
    const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) {
      this.dispatchEvent(new Event('play'));
      return Promise.resolve();
    });
    const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) {
      this.dispatchEvent(new Event('pause'));
    });
    const fps = 24000 / 1001;
    const base = snapshot.details.phase5_scoring_1!;
    const run = { ...base, events: [], annotatedVideoUrl: '/test/annotated.mp4',
      summary: { ...base.summary, fps, frames: 469 } };
    function ready() {
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: 469 / fps } });
      fireEvent.loadedMetadata(video);
      return video;
    }
    function present(frame: number) {
      const entry = pending.entries().next().value;
      expect(entry).toBeDefined();
      const [id, callback] = entry!;
      pending.delete(id);
      act(() => callback(0, { mediaTime: frame / fps } as VideoFrameCallbackMetadata));
    }
    try {
      render(<TelemetryDashboard run={run} selectedEventId={null} onSelectEvent={vi.fn()} />);
      const video = ready();
      present(30);
      const slider = screen.getByLabelText('Replay frame');
      expect(slider).toHaveValue('30');
      fireEvent.change(slider, { target: { value: '468' } });
      fireEvent.seeking(video);
      expect(video.currentTime).toBeGreaterThan(468 / fps);
      expect(video.currentTime).toBeLessThan(469 / fps);
      expect(slider).toHaveValue('30');
      expect(screen.getByLabelText('Frame 30, 1.251 seconds')).toBeInTheDocument();
      expect(screen.getByText('Seeking video · analysis holds on the presented frame.')).toBeInTheDocument();
      fireEvent.seeked(video);
      present(468);
      expect(slider).toHaveValue('468');
      expect(screen.queryByText('Seeking video · analysis holds on the presented frame.')).not.toBeInTheDocument();

      fireEvent.click(screen.getByRole('tab', { name: '2D Court Model' }));
      fireEvent.change(slider, { target: { value: '180' } });
      expect(slider).toHaveValue('180');
      fireEvent.click(screen.getByRole('tab', { name: 'Video Feed' }));
      expect(video.currentTime).toBeCloseTo(180 / fps, 5);
      expect(slider).toHaveValue('468');
      present(180);
      expect(slider).toHaveValue('180');

      fireEvent.click(screen.getByRole('button', { name: 'AI annotated' }));
      const annotated = ready();
      expect(annotated).not.toBe(video);
      expect(annotated).toHaveAttribute('src', '/test/annotated.mp4');
      expect(annotated.currentTime).toBeCloseTo(180 / fps, 5);
      present(180);
      expect(slider).toHaveValue('180');
      expect(screen.getByLabelText(/^Frame 180,/)).toBeInTheDocument();
    } finally {
      pause.mockRestore();
      play.mockRestore();
      cancel.mockRestore();
      request.mockRestore();
    }
  });
});
