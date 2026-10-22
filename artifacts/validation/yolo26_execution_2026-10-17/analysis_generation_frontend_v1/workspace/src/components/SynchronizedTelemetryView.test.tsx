import { fireEvent, render, screen } from "@testing-library/react";
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
});
