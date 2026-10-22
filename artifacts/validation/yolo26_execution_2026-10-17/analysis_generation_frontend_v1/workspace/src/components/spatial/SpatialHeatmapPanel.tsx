import { lazy, Suspense, useId, useMemo, useRef, useState } from "react";
import type { AnalysisRun } from "../../data/contracts";
import { COURT, FIELD, densityGrid, normalizeTelemetry, samplePoints, spatialStats, type FrameRange, type PlayerFilter } from "../../lib/tennisTelemetry";
import { CourtLines2D } from "./TacticalCourt";

const TennisCourt3D = lazy(() => import("./TennisCourt3D"));
const FILTERS: Array<{ value: PlayerFilter; label: string }> = [{ value: "combined", label: "Combined" }, { value: "player1", label: "P1" }, { value: "player2", label: "P2" }];

export function SpatialHeatmapPanel({ run, range = [0, Math.max(0, run.summary.frames - 1)], reducedMotion = false }: { run: AnalysisRun; range?: FrameRange; reducedMotion?: boolean }) {
  const titleId = useId(); const blurId = useId();
  const [filter, setFilter] = useState<PlayerFilter>("combined");
  const [dimension, setDimension] = useState<"2d" | "3d">("2d");
  const [zone, setZone] = useState("behindBaseline");
  const playhead = useRef(0);
  const telemetry = useMemo(() => normalizeTelemetry(run), [run]);
  const [start, end] = range;
  const stableRange = useMemo(() => [start, end] as const, [end, start]);
  const density = useMemo(() => {
    const one = densityGrid(samplePoints(telemetry, "player1", stableRange));
    const two = densityGrid(samplePoints(telemetry, "player2", stableRange));
    return { one, two, peak: Math.max(1, ...one.map(p => p.value), ...two.map(p => p.value)) };
  }, [stableRange, telemetry]);
  const points = useMemo(() => samplePoints(telemetry, filter, stableRange), [filter, stableRange, telemetry]);
  const stats = useMemo(() => spatialStats(points), [points]);
  const zones = [
    { key: "behindBaseline", label: "Behind baseline", value: stats.behindBaseline, note: "Tracked positions beyond either baseline. Depth can buy time, but gives the opponent more court to work with." },
    { key: "backcourt", label: "Back court", value: stats.backcourt, note: "Positions behind the service lines, including baseline run-off. Review the video before interpreting this as defensive play." },
    { key: "frontcourt", label: "Front court", value: stats.frontcourt, note: "Positions between the two service lines. Look for opportunities to move forward after a short ball." },
    { key: "center", label: "Center corridor", value: stats.center, note: "Positions in the middle third of singles-court width. Recovery targets depend on the opponent's shot angle." },
    { key: "left", label: "Left side", value: stats.left, note: "Positions left of court center in the camera's canonical orientation. This does not identify forehand or backhand." },
  ];
  const selected = zones.find(value => value.key === zone)!;
  if (!run.court?.isValid || !samplePoints(telemetry, "combined", stableRange).length) return <div className="spatial-empty" role="status"><h3>Spatial heatmap unavailable</h3><p>This range has no valid mapped player positions. Choose another analysis or a wider frame range.</p></div>;
  return <div className="astra-heatmap">
    <div className="heatmap-topline"><div><strong>Player position density</strong><span>{start === 0 && end === run.summary.frames - 1 ? "Entire analysis" : `Frames ${start}–${end}`} · {stats.total.toLocaleString()} player-frame samples</span></div><div className="astra-segment" role="group" aria-label="Heatmap player filter">{FILTERS.map(item => <button key={item.value} type="button" aria-pressed={item.value === filter} onClick={() => setFilter(item.value)}>{item.label}</button>)}</div></div>
    <div className="heatmap-content">
      <div className="heatmap-visual">
        <div className="heatmap-dimension astra-segment" role="group" aria-label="Density dimension"><button type="button" aria-pressed={dimension === "2d"} onClick={() => setDimension("2d")}>2D density</button><button type="button" aria-pressed={dimension === "3d"} onClick={() => setDimension("3d")}>3D density</button></div>
        {dimension === "3d" ? <Suspense fallback={<div className="spatial-loading">Preparing spatial density…</div>}><TennisCourt3D run={run} telemetry={telemetry} playhead={playhead} trails={false} arc={false} reducedMotion={reducedMotion} density={{ filter, range: stableRange }} onFallback={() => setDimension("2d")} /></Suspense> : <svg className="astra-density-court" viewBox={`${FIELD.minX} ${FIELD.minY} ${FIELD.maxX - FIELD.minX} ${FIELD.maxY - FIELD.minY}`} role="img" aria-labelledby={titleId}>
          <title id={titleId}>Normalized player position density including space behind the baseline</title>
          <defs><filter id={blurId} x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="0.28" /></filter></defs>
          <rect x={FIELD.minX} y={FIELD.minY} width={FIELD.maxX - FIELD.minX} height={FIELD.maxY - FIELD.minY} fill="#172b2d" rx=".5" /><rect width={COURT.width} height={COURT.length} fill="#263f3c" />
          <g filter={`url(#${blurId})`} key={filter} className="density-fade">{[density.one, density.two].map((grid, index) => filter !== "combined" && filter !== `player${index + 1}` ? null : <g key={index} fill={index === 0 ? "#c6f66b" : "#7cd6ec"}>{grid.filter(p => p.value / density.peak > .015).map((p, i) => <rect key={i} x={p.x - p.width / 2} y={p.y - p.depth / 2} width={p.width + .03} height={p.depth + .03} opacity={Math.min(.95, p.value / density.peak)} />)}</g>)}</g>
          <CourtLines2D />
          {zone === "behindBaseline" ? <g fill="none" stroke="#e8f0df" strokeWidth=".045" strokeDasharray=".2 .15"><rect x="0" y={FIELD.minY + .25} width={COURT.width} height={-FIELD.minY - .5} /><rect x="0" y={COURT.length + .25} width={COURT.width} height={FIELD.maxY - COURT.length - .5} /></g> : null}
          {zone === "center" ? <rect x={COURT.width / 2 - COURT.singles / 6} y={FIELD.minY} width={COURT.singles / 3} height={FIELD.maxY - FIELD.minY} fill="#fff" opacity=".06" /> : null}
          <text x={COURT.width / 2} y={-3.8} textAnchor="middle" fill="#d3ded8" fontSize=".55">FAR BASELINE</text><text x={COURT.width / 2} y={COURT.length + 4} textAnchor="middle" fill="#d3ded8" fontSize=".55">NEAR BASELINE</text>
        </svg>}
        <div className="density-legend"><span className="player-one">● P1</span><span className="player-two">◆ P2</span><span>Low</span><i /><span>High</span></div>
      </div>
      <aside className="heatmap-zones"><span className="astra-eyebrow">Position breakdown</span><h3>Where the play lives.</h3><div className="zone-list" aria-label="Court zone insights">{zones.map(item => <button key={item.key} type="button" aria-pressed={zone === item.key} onClick={() => setZone(item.key)}><span>{item.label}</span><strong>{stats.total ? item.value.toFixed(0) : "—"}<small>%</small></strong><i style={{ "--zone-fill": `${item.value}%` } as React.CSSProperties} /></button>)}</div><div className="zone-insight" aria-live="polite"><strong>{selected.label}</strong><p>{stats.total ? selected.note : "No samples for this player in the selected range."}</p></div><small>Zones can overlap. Density uses the same peak scale for both players; percentages describe selected player-frame samples.</small></aside>
    </div>
  </div>;
}
