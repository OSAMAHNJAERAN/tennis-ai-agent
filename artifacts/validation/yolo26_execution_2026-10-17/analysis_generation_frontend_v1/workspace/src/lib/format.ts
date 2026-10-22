export const numberFormatter = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });

export function formatNumber(value: number | null | undefined, suffix = "") {
  return value == null || !Number.isFinite(value) ? "Unavailable" : `${numberFormatter.format(value)}${suffix}`;
}

export function formatPercent(value: number | null | undefined) {
  return formatNumber(value, "%");
}

export function formatConfidence(value: number | null | undefined) {
  return value == null ? "Unavailable" : `${Math.round(value * 100)}%`;
}

export function formatDuration(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return "Unavailable";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

export function formatFrameTime(frame: number, fps: number) {
  return `${frame} / ${formatDuration(frame / fps)}`;
}

export function formatDateTime(value: string) {
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function humanize(value: string) {
  return value.replaceAll("_", " ").toLowerCase().replace(/^./, (letter) => letter.toUpperCase());
}

