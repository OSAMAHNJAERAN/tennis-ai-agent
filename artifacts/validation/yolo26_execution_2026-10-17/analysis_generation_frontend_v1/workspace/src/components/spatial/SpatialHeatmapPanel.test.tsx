import axe from "axe-core";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { snapshot } from "../../data/repository";
import { SpatialHeatmapPanel } from "./SpatialHeatmapPanel";

const run = snapshot.details.phase5_scoring_1!;
describe("spatial density", () => {
  it("counts only the selected player's frames and updates zone explanations", async () => {
    const user = userEvent.setup();
    render(<SpatialHeatmapPanel run={run} range={[0, 29]} />);
    expect(screen.getByText(/60 player-frame samples/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "P1" }));
    expect(screen.getByText(/30 player-frame samples/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Left side/ }));
    expect(screen.getByText(/This does not identify forehand or backhand/)).toBeInTheDocument();
  });

  it("offers an explicit empty state for invalid court calibration", () => {
    render(<SpatialHeatmapPanel run={{ ...run, court: run.court ? { ...run.court, isValid: false } : null }} />);
    expect(screen.getByRole("status")).toHaveTextContent("Spatial heatmap unavailable");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("has accessible density and zone controls", async () => {
    const { container } = render(<SpatialHeatmapPanel run={run} />);
    const result = await axe.run(container, { rules: { "color-contrast": { enabled: false }, region: { enabled: false } } });
    expect(result.violations).toEqual([]);
  });
});
