import { CheckCircle, WarningDiamond } from "@phosphor-icons/react";
import type { LineCall } from "../data/contracts";
import { isReviewRequired } from "./DecisionBadge";

export function ReviewWorkflow({ calls }: { calls: LineCall[] }) {
  const reviewCalls = calls.filter((call) => isReviewRequired(call.decision));
  if (!reviewCalls.length) {
    return (
      <div className="review-workflow review-clear" role="status">
        <CheckCircle size={23} weight="duotone" aria-hidden="true" />
        <div><strong>No human review needed</strong><p>Every current line call resolved outside the configured uncertainty band.</p></div>
      </div>
    );
  }
  return (
    <div className="review-workflow review-needed" role="alert">
      <WarningDiamond size={23} weight="fill" aria-hidden="true" />
      <div><strong>Human review needed</strong><p>{reviewCalls.length} decision{reviewCalls.length === 1 ? "" : "s"} intersects the uncertainty band. Inspect the source frame before acceptance.</p></div>
    </div>
  );
}

