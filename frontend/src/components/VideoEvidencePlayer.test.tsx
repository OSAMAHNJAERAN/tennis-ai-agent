import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { snapshot } from "../data/repository";
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
});

