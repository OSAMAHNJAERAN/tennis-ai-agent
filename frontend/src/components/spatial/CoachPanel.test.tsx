import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { snapshot } from "../../data/repository";
import { normalizeTelemetry } from "../../lib/tennisTelemetry";
import { CoachPanel } from "./CoachPanel";

const run = snapshot.details.phase5_scoring_1!;
const telemetry = normalizeTelemetry(run);
afterEach(() => vi.unstubAllGlobals());

describe("Astra coach connection", () => {
  it("keeps observations local until the user requests coaching, then sends only the selected evidence", async () => {
    const request = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ text: "Review recovery after contact.", model: "configured-model" }) });
    vi.stubGlobal("fetch", request);
    const user = userEvent.setup();
    render(<CoachPanel run={run} telemetry={telemetry} range={[30, 89]} />);
    expect(request).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Generate coaching" }));
    expect(await screen.findByText("Review recovery after contact.")).toBeInTheDocument();
    const body = JSON.parse(request.mock.calls[0]![1].body);
    expect(body.evidence.range).toEqual({ startFrame: 30, endFrame: 89, durationSeconds: 2 });
    expect(body.evidence.events.every((event: { frame: number }) => event.frame >= 30 && event.frame <= 89)).toBe(true);
    expect(Object.keys(body.evidence).sort()).toEqual(["events", "fps", "geometryValid", "limitations", "players", "range", "runId"]);
    expect(request.mock.calls[0]![1].headers).toEqual({ "Content-Type": "application/json" });
  });

  it("explains an unconfigured service without inventing a model response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503 }));
    const user = userEvent.setup();
    render(<CoachPanel run={run} telemetry={telemetry} range={[0, 213]} />);
    await user.click(screen.getByRole("button", { name: "Generate coaching" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Astra GPT is not connected");
    expect(screen.getByText("Local telemetry observation")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate coaching" })).toBeEnabled();
  });

  it("aborts a pending provider request when the selected evidence is removed", async () => {
    const request = vi.fn().mockImplementation((_url: string, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    }));
    vi.stubGlobal("fetch", request);
    const user = userEvent.setup();
    const { unmount } = render(<CoachPanel run={run} telemetry={telemetry} range={[0, 213]} />);
    await user.click(screen.getByRole("button", { name: "Generate coaching" }));
    await waitFor(() => expect(request).toHaveBeenCalledOnce());
    const signal = request.mock.calls[0]![1].signal as AbortSignal;
    unmount();
    expect(signal.aborted).toBe(true);
  });
});
