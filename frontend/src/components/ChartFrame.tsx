import type { ReactNode } from "react";

export function ChartFrame({ label, detail, children }: { label: string; detail: string; children: ReactNode }) {
  return (
    <figure className="chart-frame" aria-label={`${label}. ${detail}`}>
      <figcaption><strong>{label}</strong><span>{detail}</span></figcaption>
      <div className="chart-area" aria-hidden="true">{children}</div>
    </figure>
  );
}

