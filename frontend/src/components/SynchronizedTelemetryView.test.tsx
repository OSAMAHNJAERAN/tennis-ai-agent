import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { snapshot } from "../data/repository";
import { SynchronizedTelemetryView } from "./SynchronizedTelemetryView";

describe("synchronized telemetry view", () => {
  it("preserves the video element and playback state host while switching tabs", async () => {
    const user = userEvent.setup();
    const run = snapshot.details.phase5_scoring_1!;
    render(<SynchronizedTelemetryView run={run} selectedEventId={1} onSelectEvent={vi.fn()} />);

    const originalVideo = screen.getByLabelText("Source match video");
    const courtTab = screen.getByRole("tab", { name: "2D Court Model" });
    await user.click(courtTab);

    expect(courtTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("Source match video")).toBe(originalVideo);
    expect(screen.getByRole("tabpanel", { name: "2D Court Model" })).not.toHaveAttribute("hidden");
  });

  it("supports arrow-key tab navigation", async () => {
    const user = userEvent.setup();
    const run = snapshot.details.phase5_scoring_1!;
    render(<SynchronizedTelemetryView run={run} selectedEventId={null} onSelectEvent={vi.fn()} />);

    const videoTab = screen.getByRole("tab", { name: /Video Feed/ });
    await user.click(videoTab);
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "2D Court Model" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: /3D Court/ })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Spatial Heatmap" })).toHaveAttribute("aria-selected", "true");
  });

  it("seeks the same source video while stepping frames in an analytical view", async () => {
    const user = userEvent.setup();
    const run = snapshot.details.phase5_scoring_1!;
    render(<SynchronizedTelemetryView run={run} selectedEventId={null} onSelectEvent={vi.fn()} />);
    await user.click(screen.getByRole("tab", { name: "2D Court Model" }));
    fireEvent.change(screen.getByRole("slider", { name: "Replay frame" }), { target: { value: "60" } });
    expect((screen.getByLabelText("Source match video") as HTMLVideoElement).currentTime).toBeCloseTo(2);
    await user.click(screen.getByRole("button", { name: "Next replay frame" }));
    expect(screen.getByRole("slider", { name: "Replay frame" })).toHaveValue("61");
    expect((screen.getByLabelText("Source match video") as HTMLVideoElement).currentTime).toBeCloseTo(61 / 30);
  });

  it("resets the playhead and analysis interval when switching runs", () => {
    const run = snapshot.details.phase5_scoring_1!;
    const props = { selectedEventId: null, onSelectEvent: vi.fn() };
    const { rerender } = render(<SynchronizedTelemetryView run={run} {...props} />);
    fireEvent.change(screen.getByRole("slider", { name: "Replay frame" }), { target: { value: "120" } });
    fireEvent.change(screen.getByLabelText("Analysis start frame"), { target: { value: "30" } });
    rerender(<SynchronizedTelemetryView run={{ ...run, summary: { ...run.summary, id: "another-run", frames: 20 } }} {...props} />);
    expect(screen.getByRole("slider", { name: "Replay frame" })).toHaveValue("0");
    expect(screen.getByLabelText("Analysis start frame")).toHaveValue(0);
    expect(screen.getByLabelText("Analysis end frame")).toHaveValue(19);
  });
  it('seeks reviewed signals missing from physical events using the same video clock', () => {
    const run = snapshot.details.phase5_scoring_1!; const props = { selectedEventId: 999999, onSelectEvent: vi.fn() };
    const pending = new Map<number, VideoFrameRequestCallback>(); let nextId = 0;
    const request = vi.spyOn(HTMLVideoElement.prototype, 'requestVideoFrameCallback')
      .mockImplementation(callback => { pending.set(++nextId, callback); return nextId; });
    const cancel = vi.spyOn(HTMLVideoElement.prototype, 'cancelVideoFrameCallback')
      .mockImplementation(id => { pending.delete(id); });
    const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    function present(frame: number) {
      const entry = pending.entries().next().value; expect(entry).toBeDefined(); const [id, callback] = entry!;
      pending.delete(id); act(() => callback(0, { mediaTime: frame / run.summary.fps } as VideoFrameCallbackMetadata));
    }
    try {
      const { rerender } = render(<SynchronizedTelemetryView run={run} {...props} />);
      const video = screen.getByLabelText('Source match video') as HTMLVideoElement;
      Object.defineProperties(video, { videoWidth: { value: 100 }, videoHeight: { value: 100 }, duration: { value: run.summary.durationSeconds } });
      fireEvent.loadedMetadata(video); present(0);
      const slider = screen.getByRole('slider', { name: 'Replay frame' });
      rerender(<SynchronizedTelemetryView run={run} {...props} focusFrame={{ frame: 45 }} />);
      expect(video.currentTime).toBeCloseTo(45 / run.summary.fps, 5); expect(slider).toHaveValue('0');
      present(45); expect(slider).toHaveValue('45');
      fireEvent.change(slider, { target: { value: '90' } }); present(90); expect(slider).toHaveValue('90');
      rerender(<SynchronizedTelemetryView run={run} {...props} focusFrame={{ frame: 45 }} />);
      expect(screen.getByLabelText('Source match video')).toBe(video); expect(video.currentTime).toBeCloseTo(45 / run.summary.fps, 5);
      expect(slider).toHaveValue('90'); present(45); expect(slider).toHaveValue('45');
    } finally { pause.mockRestore(); cancel.mockRestore(); request.mockRestore(); }
  });
});
