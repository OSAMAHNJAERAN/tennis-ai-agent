import { Footprints, Gauge, Target, UserFocus } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";
import { OccupancyMap } from "../components/OccupancyMap";
import { Card, EmptyState, Metric, PageHeader, SectionHeading, StatusBadge } from "../components/ui";
import { eventParticipation } from "../lib/analytics";
import { formatNumber, formatPercent } from "../lib/format";

export default function Players() {
  const { run } = useAnalysis();
  if (!run) return null;
  if (!run.playerMetrics || !run.trajectories) return <><PageHeader eyebrow="Movement intelligence" title="Players" description="Player movement requires both trajectory and metrics artifacts." /><EmptyState title="Player analytics unavailable" detail="Select a run that contains player_metrics.json and trajectories.json." /></>;
  const players = [run.playerMetrics.player_1, run.playerMetrics.player_2];
  const trails = [run.trajectories.player1CourtPositions, run.trajectories.player2CourtPositions];
  const playerOne = run.playerMetrics.player_1;
  const playerTwo = run.playerMetrics.player_2;

  return (
    <div className="page players-page">
      <PageHeader eyebrow="Movement intelligence" title="Player comparison" description="Distances, average speeds, occupancy, and event participation derived only from tracked player court positions and semantic events." />
      <div className="player-hero-grid">
        {players.map((player, index) => (
          <Card className={`player-hero-card player-${index + 1}`} key={index}>
            <div className="player-card-number">0{index + 1}</div>
            <div className="player-card-heading"><span><UserFocus size={18} weight="duotone" /> Tracked athlete</span><h2>Player {index + 1}</h2></div>
            <div className="player-metric-grid">
              <Metric primary label="Distance" value={formatNumber(player.total_distance_m, " m")} meta="Court-plane path length" />
              <Metric label="Average speed" value={formatNumber(player.avg_speed_kmh, " km/h")} />
              <Metric label="Coverage" value={formatPercent(player.coverage_pct)} />
              <Metric label="Event participation" value={eventParticipation(run, index + 1)} meta="Contacts attributed" />
            </div>
          </Card>
        ))}
      </div>
      <section className="occupancy-section">
        <SectionHeading eyebrow="Spatial behavior" title="Court occupancy" detail="Grayscale density preserves the single lime selection signal elsewhere in the interface." />
        <div className="occupancy-grid">
          {trails.map((trail, index) => <Card className="occupancy-card" key={index}><div><StatusBadge>Player {index + 1}</StatusBadge><span>{trail.filter(Boolean).length} tracked frames</span></div><OccupancyMap points={trail} label={`Player ${index + 1}`} /></Card>)}
          <Card className="movement-notes panel-padding">
            <h3>Movement record</h3>
            <div className="movement-note"><Footprints size={19} /><span><strong>{formatNumber(playerTwo.total_distance_m - playerOne.total_distance_m, " m")}</strong><small>Player 2 distance advantage</small></span></div>
            <div className="movement-note"><Gauge size={19} /><span><strong>{formatNumber(playerTwo.avg_speed_kmh - playerOne.avg_speed_kmh, " km/h")}</strong><small>Average speed difference</small></span></div>
            <div className="movement-note"><Target size={19} /><span><strong>{formatPercent((playerOne.coverage_pct + playerTwo.coverage_pct) / 2)}</strong><small>Mean detection coverage</small></span></div>
            <p>Positions outside the canonical boundary are clamped for display. Source coordinates remain unchanged in the snapshot.</p>
          </Card>
        </div>
      </section>
    </div>
  );
}
