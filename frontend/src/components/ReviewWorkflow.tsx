import { CheckCircle, WarningDiamond } from "@phosphor-icons/react";
import type { LineCall } from "../data/contracts";
import { isReviewRequired } from "./DecisionBadge";

export function ReviewWorkflow({ calls }: { calls: LineCall[] }) {
  if (!calls.length) {
    return (
      <div className="review-workflow" role="status">
        <div><strong>No line calls available</strong><p>Line-call evidence is unavailable for this analysis.</p></div>
      </div>
    );
  }
  const hasMeasurements = (call: LineCall) => call.contact_patch_model !== "NOT_EVALUATED"
    && call.ball_edge_margin_cm != null && call.position_uncertainty_cm != null && call.contact_patch_radius_cm != null;
  const reviewCalls = calls.filter((call) => isReviewRequired(call.decision) || !hasMeasurements(call));
  if (!reviewCalls.length) {
    return (
      <div className="review-workflow review-clear" role="status">
        <CheckCircle size={23} weight="duotone" aria-hidden="true" />
        <div><strong>No human review needed</strong><p>Every current line call resolved outside the configured uncertainty band.</p></div>
      </div>
    );
  }
  const unmeasuredCalls = reviewCalls.filter((call) => !hasMeasurements(call));
  const missingServiceContext = unmeasuredCalls.filter((call) =>
    call.reason_code === "SERVICE_TARGET_UNAVAILABLE" || call.reason_code === "SERVICE_CONTEXT_UNAVAILABLE").length;
  const otherUnavailable = unmeasuredCalls.length - missingServiceContext;
  const measuredReviews = reviewCalls.length - unmeasuredCalls.length;
  const details = [
    missingServiceContext ? `${missingServiceContext} decision${missingServiceContext === 1 ? " needs" : "s need"} verified service context before geometry can be evaluated.` : "",
    otherUnavailable ? `${otherUnavailable} decision${otherUnavailable === 1 ? " has" : "s have"} unavailable contact measurements.` : "",
    measuredReviews ? `${measuredReviews} decision${measuredReviews === 1 ? " intersects" : "s intersect"} the uncertainty band.` : "",
    unmeasuredCalls.length ? "Review the source evidence before acceptance." : "Inspect the source frame before acceptance.",
  ].filter(Boolean).join(" ");
  return (
    <div className="review-workflow review-needed" role="alert">
      <WarningDiamond size={23} weight="fill" aria-hidden="true" />
      <div><strong>Human review needed</strong><p>{details}</p></div>
    </div>
  );
}

