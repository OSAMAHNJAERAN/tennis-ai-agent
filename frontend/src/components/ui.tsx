import type { HTMLAttributes, ReactNode } from "react";
import { WarningCircle } from "@phosphor-icons/react";
import { cn } from "../lib/cn";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("card", className)} {...props} />;
}

export function UtilityCard({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("utility-card", className)} {...props} />;
}

export function SectionHeading({
  eyebrow,
  title,
  detail,
  action,
}: {
  eyebrow?: string;
  title: string;
  detail?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h2>{title}</h2>
        {detail ? <p className="section-detail">{detail}</p> : null}
      </div>
      {action ? <div className="section-action">{action}</div> : null}
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action ? <div className="page-action">{action}</div> : null}
    </header>
  );
}

export function Metric({
  label,
  value,
  meta,
  primary = false,
}: {
  label: string;
  value: ReactNode;
  meta?: ReactNode;
  primary?: boolean;
}) {
  return (
    <div className={cn("metric", primary && "metric-primary")}>
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      {meta ? <span className="metric-meta">{meta}</span> : null}
    </div>
  );
}

export function StatusBadge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "signal" }) {
  return <span className={cn("status-badge", tone === "signal" && "status-signal")}>{children}</span>;
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state" role="status">
      <WarningCircle size={24} weight="duotone" aria-hidden="true" />
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}

export function SkeletonPage() {
  return (
    <div className="skeleton-page" aria-label="Loading analysis" role="status">
      <span className="skeleton skeleton-title" />
      <span className="skeleton skeleton-subtitle" />
      <div className="skeleton-grid">
        <span className="skeleton skeleton-card" />
        <span className="skeleton skeleton-card" />
        <span className="skeleton skeleton-card" />
      </div>
    </div>
  );
}

export function DataError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="error-state" role="alert">
      <WarningCircle size={32} weight="fill" aria-hidden="true" />
      <div>
        <h2>Analysis data could not be loaded</h2>
        <p>{message}</p>
      </div>
      <button className="button button-primary" type="button" onClick={onRetry}>
        Retry
      </button>
    </div>
  );
}

