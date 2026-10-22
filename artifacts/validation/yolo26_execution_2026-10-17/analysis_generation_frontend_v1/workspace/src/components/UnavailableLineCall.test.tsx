import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LineCallSchema, type AnalysisRun, type LineCall } from "../data/contracts";
import { CourtDiagram } from "./CourtDiagram";
import { LineGeometryZoom } from "./LineGeometryZoom";
import { ReviewWorkflow } from "./ReviewWorkflow";

const unavailable: LineCall = {
  event_id: 9, bounce_frame: 30, decision: "REVIEW_REQUIRED", decision_context: "SERVE",
  nearest_line: "NONE", bounce_position_px: null, bounce_position_m: null, tracker_state: "UNKNOWN",
  center_signed_distance_cm: null, contact_patch_model: "NOT_EVALUATED", contact_patch_radius_cm: null,
  ball_edge_margin_cm: null, position_uncertainty_cm: null, spatial_tier: "UNAVAILABLE", confidence: 0,
  reason: "Current service context unavailable.", refinement_method: "NOT_PERFORMED", line_strip_info: null,
  reason_code: "SERVICE_TARGET_UNAVAILABLE",
};
const measured: LineCall = {
  ...unavailable, event_id: 10, decision: "IN", decision_context: "RALLY", nearest_line: "LEFT_SIDELINE",
  bounce_position_px: [200, 300], bounce_position_m: [3, 6], tracker_state: "DETECTED",
  center_signed_distance_cm: 100, contact_patch_model: "EMPIRICAL_PATCH", contact_patch_radius_cm: 1.25,
  ball_edge_margin_cm: 101.25, position_uncertainty_cm: 1.5, spatial_tier: "MID", confidence: .9,
  reason: "Explicit synthetic measured fixture.", refinement_method: "SYNTHETIC_ONLY", reason_code: null,
};

describe("unavailable line-call evidence", () => {
  it("accepts explicit null geometry and legacy measured records without reason codes", () => {
    expect(LineCallSchema.parse(unavailable).bounce_position_m).toBeNull();
    const legacy = { ...measured };
    delete legacy.reason_code;
    expect(LineCallSchema.parse(legacy).ball_edge_margin_cm).toBe(101.25);
  });
  it("does not draw an invented contact patch or signed margin", () => {
    const { container } = render(<LineGeometryZoom call={unavailable} />);
    expect(screen.getByRole("img")).toHaveAccessibleName("Line-call geometry unavailable");
    expect(container.querySelector(".contact-patch")).toBeNull();
    expect(container.textContent).not.toContain("0.0 CM");
  });
  it("preserves the measured geometry display", () => {
    const { container } = render(<LineGeometryZoom call={measured} />);
    expect(container.querySelector(".contact-patch")).not.toBeNull();
    expect(screen.getByText("101.3 CM SIGNED MARGIN")).toBeInTheDocument();
  });
  it("plots only known court coordinates", () => {
    const run = { lineCalls: [unavailable, measured], events: [], trajectories: null } as unknown as AnalysisRun;
    const { container } = render(<CourtDiagram run={run} />);
    expect(container.querySelectorAll(".line-call-marker")).toHaveLength(1);
    expect(container.innerHTML).not.toMatch(/NaN|undefined/);
  });
  it("shows a neutral empty state without claiming all calls resolved", () => {
    render(<ReviewWorkflow calls={[]} />);
    expect(screen.getByRole("status")).toHaveTextContent("No line calls available");
    expect(screen.queryByText("No human review needed")).not.toBeInTheDocument();
  });
  it.each(["SERVICE_TARGET_UNAVAILABLE", "SERVICE_CONTEXT_UNAVAILABLE"])("explains %s without an uncertainty-band claim", (reason_code) => {
    render(<ReviewWorkflow calls={[{ ...unavailable, reason_code }]} />);
    expect(screen.getByRole("alert")).toHaveTextContent("verified service context");
    expect(screen.getByRole("alert")).not.toHaveTextContent("intersects the uncertainty band");
  });
  it("distinguishes missing measurements from measured uncertainty in a mixed review", () => {
    render(<ReviewWorkflow calls={[unavailable, { ...measured, decision: "REVIEW_REQUIRED" }]} />);
    expect(screen.getByRole("alert")).toHaveTextContent("1 decision needs verified service context");
    expect(screen.getByRole("alert")).toHaveTextContent("1 decision intersects the uncertainty band");
  });
});
