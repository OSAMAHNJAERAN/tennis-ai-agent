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

export function MetricCard({
  icon: Icon,
  label,
  value,
  subvalue,
  subValue,
  trend,
  trendPositive = true,
  className,
  highlight = false,
  variant,
  progress,
  progressColor = "lime",
}: {
  icon?: (props: { size?: number; weight?: "regular" | "fill" | "duotone" | "bold"; className?: string }) => ReactNode;
  label: string;
  value: ReactNode;
  subvalue?: ReactNode;
  subValue?: ReactNode;
  trend?: string;
  trendPositive?: boolean;
  className?: string;
  highlight?: boolean;
  variant?: "highlight" | "default";
  progress?: number;
  progressColor?: "lime" | "ink" | "charcoal" | "ash";
}) {
  const displaySub = subvalue ?? subValue;
  const isHighlighted = highlight || variant === "highlight";

  return (
    <div className={cn("metric-card", isHighlighted && "metric-card-highlight", className)}>
      <div className="metric-card-head">
        <div className="metric-card-icon-title">
          {Icon ? <Icon size={16} weight="duotone" className="metric-card-icon" /> : null}
          <span className="metric-card-label">{label}</span>
        </div>
        {trend ? (
          <span className={cn("trend-badge", trendPositive ? "trend-positive" : "trend-negative")}>
            {trend}
          </span>
        ) : null}
      </div>
      <div className="metric-card-body">
        <strong className="metric-card-val">{value}</strong>
        {displaySub ? <span className="metric-card-sub">{displaySub}</span> : null}
      </div>
      {typeof progress === "number" && (
        <ProgressTrack value={progress} color={progressColor} className="mt-1" />
      )}
    </div>
  );
}

export function StatRow({
  label,
  value,
  meta,
  highlight = false,
}: {
  label: ReactNode;
  value: ReactNode;
  meta?: ReactNode;
  highlight?: boolean;
}) {
  return (
    <div className={cn("stat-row", highlight && "stat-row-highlight")}>
      <span className="stat-row-label">{label}</span>
      <div className="stat-row-val-group">
        <strong className="stat-row-value">{value}</strong>
        {meta ? <small className="stat-row-meta">{meta}</small> : null}
      </div>
    </div>
  );
}

export function ProgressTrack({
  value,
  max = 100,
  className,
  color = "lime",
}: {
  value: number;
  max?: number;
  className?: string;
  color?: "lime" | "ink" | "charcoal" | "ash";
}) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className={cn("progress-track", className)}>
      <div
        className={cn("progress-bar", `progress-${color}`)}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}
