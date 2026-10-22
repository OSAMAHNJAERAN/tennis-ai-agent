import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AnalysisProvider } from "../app/AnalysisContext";
import Events from "./Events";

function renderEvents(runId = "phase5_scoring_1") {
  return render(<MemoryRouter initialEntries={[`/events?run=${runId}`]}><AnalysisProvider><Events /></AnalysisProvider></MemoryRouter>);
}

describe("events view", () => {
  it("filters a TanStack table and renders mobile records", async () => {
    const user = userEvent.setup();
    renderEvents();
    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(document.querySelectorAll(".mobile-event-card")).toHaveLength(6);
    await user.selectOptions(screen.getByLabelText("Event type"), "BOUNCE");
    expect(screen.getByText("3 of 6")).toBeInTheDocument();
    expect(document.querySelectorAll(".mobile-event-card")).toHaveLength(3);
  });

  it("opens a keyboard-accessible evidence drawer", async () => {
    const user = userEvent.setup();
    renderEvents();
    const openButtons = await screen.findAllByRole("button", { name: /Open event/ });
    await user.click(openButtons[0]!);
    expect(screen.getByRole("dialog")).toHaveTextContent("Recorded evidence");
    expect(screen.getByRole("button", { name: "Replay at this frame" })).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders an explicit empty state for older generations", async () => {
    renderEvents("baseline_run_1");
    expect(await screen.findByText("No semantic events")).toBeInTheDocument();
  });
});

