import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { LineCall } from "../data/contracts";
import { ReviewWorkflow } from "./ReviewWorkflow";

const call = (decision: LineCall["decision"]): LineCall => ({
  event_id: 1,
  bounce_frame: 20,
  decision,
  decision_context: "RALLY",
  nearest_line: "BASELINE",
  bounce_position_px: [100, 100],
  bounce_position_m: [5, 22],
  tracker_state: "DETECTED",
  center_signed_distance_cm: 1,
  contact_patch_model: "EMPIRICAL_PATCH",
  contact_patch_radius_cm: 1.25,
  ball_edge_margin_cm: 0,
  position_uncertainty_cm: 1.5,
  spatial_tier: "NEAR",
  confidence: 0.7,
  reason: "Inside uncertainty band.",
  refinement_method: "PIECEWISE_IMPACT_INTERSECTION",
});

describe("review workflow", () => {
  it("shows the current no-review state", () => {
    render(<ReviewWorkflow calls={[call("IN"), call("OUT")]} />);
    expect(screen.getByText("No human review needed")).toBeInTheDocument();
  });

  it("announces review-required decisions", () => {
    render(<ReviewWorkflow calls={[call("REVIEW_REQUIRED")]} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Human review needed");
  });
});

