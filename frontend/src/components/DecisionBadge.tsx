import { CheckCircle, Question, XCircle } from "@phosphor-icons/react";
import type { LineDecision } from "../data/contracts";
import { cn } from "../lib/cn";
import { humanize } from "../lib/format";

export function isReviewRequired(decision: LineDecision) {
  return decision === "REVIEW_REQUIRED" || decision === "UNKNOWN";
}

export function DecisionBadge({ decision }: { decision: LineDecision }) {
  const review = isReviewRequired(decision);
  const positive = decision === "IN" || decision === "SERVE_IN";
  const Icon = review ? Question : positive ? CheckCircle : XCircle;
  return (
    <span className={cn("decision-badge", review ? "decision-review" : positive ? "decision-in" : "decision-out")}>
      <Icon size={15} weight="fill" aria-hidden="true" />
      {review ? "Human review needed" : humanize(decision)}
    </span>
  );
}

