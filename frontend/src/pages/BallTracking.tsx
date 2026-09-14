import { Flask, Info, Path, Pulse, TennisBall } from "@phosphor-icons/react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useAnalysis } from "../app/AnalysisContext";
import { ChartFrame } from "../components/ChartFrame";
import { Card, EmptyState, Metric, PageHeader, SectionHeading, StatusBadge } from "../components/ui";
import { deriveCoverage, trackingDistribution } from "../lib/analytics";
import { formatNumber, formatPercent, humanize } from "../lib/format";

function getConfigString(config: Record<string, unknown> | null, section: string, key: string) {
  const group = config?.[section];
  if (!group || typeof group !== "object") return null;
  const value = (group as Record<string, unknown>)[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : null;
}

export default function BallTracking() {
  const { run } = useAnalysis();
  if (!run) return null;
  const trajectory = run.trajectories?.ballTrajectory ?? [];
  if (!trajectory.length) return <><PageHeader eyebrow="Temporal estimation" title="Ball tracking" description="Frame-level tracking evidence and scientific speed estimates." /><EmptyState title="Ball trajectory unavailable" detail="This generation contains no normalized ball trajectory points." /></>;
  const distribution = trackingDistribution(trajectory).filter((item) => item.count > 0);
  const coverage = deriveCoverage(trajectory);
  const timeline = trajectory.map((point) => ({ frame: point.frame_index, confidence: point.confidence == null ? null : point.confidence * 100, speed: point.speed_kmh, state: point.state }));
  const overview = run.ballMetrics?.speed_overview;

  return (
    <div className="page ball-page">
      <PageHeader eyebrow="Temporal estimation" title="Ball tracking" description="Tracker-state provenance, confidence, court trajectory, and experimental two-dimensional speed estimates remain separated from match facts." action={<StatusBadge tone="signal">{trajectory.length} trajectory points</StatusBadge>} />
      <section className="ball-kpi-grid">
        <Card><Metric primary label="Valid tracking coverage" value={formatPercent(coverage)} meta="Observed or reconstructed frames" /></Card>
        <Card><Metric label="Mean estimated speed" value={formatNumber(overview?.average_speed_kmh, " km/h")} meta={overview ? "2D court-plane estimate" : "Unavailable in this run"} /></Card>
        <Card><Metric label="Maximum estimate" value={formatNumber(overview?.maximum_speed_kmh, " km/h")} meta={overview ? `${overview.segments_count} flight segments` : "Unavailable in this run"} /></Card>
        <Card><Metric label="Model" value="YOLO11s" meta={getConfigString(run.runConfig, "ball_detection", "model") ?? "Path unavailable"} /></Card>
      </section>
      <div className="ball-chart-grid">
        <Card className="panel-padding chart-card-wide">
          <SectionHeading eyebrow="Frame telemetry" title="Confidence and speed timeline" detail="Null speed values stay discontinuous rather than being interpolated for presentation." />
          <ChartFrame label="Ball telemetry by frame" detail="Confidence percentage and estimated kilometers per hour.">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeline} margin={{ top: 10, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid stroke="var(--color-gridline)" vertical={false} />
                <XAxis dataKey="frame" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 10 }} width={42} />
                <Tooltip contentStyle={{ border: "1px solid var(--color-gridline)", borderRadius: 8, fontSize: 11 }} />
                <Line type="linear" dataKey="confidence" name="Confidence %" stroke="var(--color-ink)" dot={false} strokeWidth={1.5} connectNulls={false} />
                <Line type="linear" dataKey="speed" name="Speed km/h" stroke="var(--color-ember-orange)" dot={false} strokeWidth={2.5} connectNulls={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartFrame>
        </Card>
        <Card className="panel-padding">
          <SectionHeading eyebrow="Provenance" title="Tracker states" detail="Scientific state labels are preserved." />
          <ChartFrame label="Tracking state distribution" detail={distribution.map((item) => `${item.state} ${item.count}`).join(", ")}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distribution} layout="vertical" margin={{ top: 4, right: 4, left: 10, bottom: 4 }}>
                <CartesianGrid stroke="var(--color-gridline)" horizontal={false} />
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="state" width={92} tick={{ fontSize: 9 }} />
                <Tooltip cursor={{ fill: "var(--surface-vellum)" }} />
                <Bar dataKey="count" fill="var(--color-ink)" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartFrame>
        </Card>
      </div>

      <section className="flight-section">
        <SectionHeading eyebrow="Kinematic segments" title="Flight inspection" detail="Segment boundaries come from detected contacts and bounces." />
        {run.ballMetrics ? <div className="flight-list">{run.ballMetrics.flight_segments.map((segment) => (
          <Card className="flight-card" key={segment.segment_id}>
            <div className="flight-number">{String(segment.segment_id).padStart(2, "0")}</div>
            <div><span>{segment.start_event ? humanize(segment.start_event) : "Clip start"}</span><strong>F{segment.start_frame} → F{segment.end_frame}</strong><small>{segment.end_event ? humanize(segment.end_event) : "Clip end"}</small></div>
            <div><span>Mean</span><strong>{formatNumber(segment.mean_speed_kmh, " km/h")}</strong></div>
            <div><span>Distance</span><strong>{formatNumber(segment.total_2d_distance_m, " m")}</strong></div>
            <StatusBadge>{segment.confidence_tier}</StatusBadge>
          </Card>
        ))}</div> : <EmptyState title="Speed segments unavailable" detail="Ball speed metrics were introduced in Phase 5. Tracking states and coordinates remain available above." />}
      </section>

      <Card className="scientific-note">
        <div className="scientific-icon"><Flask size={22} weight="duotone" /></div>
        <div><span>Scientific scope</span><h2>{run.ballMetrics?.scientific_status ? humanize(run.ballMetrics.scientific_status) : "Speed metrics unavailable"}</h2><p>{overview?.scientific_disclaimer ?? "This run predates the 2D speed estimation artifact. No speed values have been fabricated."}</p></div>
        <div className="method-pills"><span><Path size={14} /> Court ground plane</span><span><Pulse size={14} /> Native frame time</span><span><Info size={14} /> Monocular source</span><span><TennisBall size={14} /> Experimental</span></div>
      </Card>
    </div>
  );
}

