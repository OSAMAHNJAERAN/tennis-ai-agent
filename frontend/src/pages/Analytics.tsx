import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
        Sparkle,
        DownloadSimple,
} from "@phosphor-icons/react";
import { MetricCard } from "../components/ui";

const MATCH_DATA = {
  p1: { name: "Carlos Alcaraz", seed: 1, rank: 1, country: "ESP" },
  p2: { name: "Jannik Sinner", seed: 2, rank: 2, country: "ITA" },
  score: "6-4, 3-6, 7-6 (5), 6-4",
  duration: "3h 48m",
  tournament: "US Open 2026 - Championship Match",
  date: "September 8, 2026",
};

const STATS_COMPARISON = [
  { metric: "1st Serve Percentage", p1: "68%", p2: "62%", winner: "p1" },
  { metric: "1st Serve Points Won", p1: "79% (64/81)", p2: "73% (58/79)", winner: "p1" },
  { metric: "2nd Serve Points Won", p1: "56% (24/43)", p2: "51% (21/41)", winner: "p1" },
  { metric: "Aces / Double Faults", p1: "14 / 2", p2: "11 / 4", winner: "p1" },
  { metric: "Break Points Converted", p1: "4/9 (44%)", p2: "2/7 (29%)", winner: "p1" },
  { metric: "Net Points Won", p1: "26/34 (76%)", p2: "18/27 (67%)", winner: "p1" },
  { metric: "Total Winners", p1: "52", p2: "44", winner: "p1" },
  { metric: "Unforced Errors", p1: "28", p2: "34", winner: "p1" },
  { metric: "Max Serve Velocity", p1: "138 mph", p2: "135 mph", winner: "p1" },
  { metric: "Avg Rally Length", p1: "5.4 shots", p2: "5.4 shots", winner: "tie" },
];

export const Analytics: React.FC = () => {
  const [activeSet, setActiveSet] = useState<string>("all");

  return (
    <div className="page">
      {/* Header */}
      <header className="page-header">
        <div>
          <div className="eyebrow">Tactical Performance Telemetry</div>
          <h1>Advanced Match Analytics</h1>
          <p>
            Comprehensive post-match and live aggregate breakdown across stroke distributions, momentum curves, serve efficiency, and strategic metrics.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <select
            value={activeSet}
            onChange={(e) => setActiveSet(e.target.value)}
            className="select-control"
            aria-label="Select match set"
          >
            <option value="all">Full Match Aggregate</option>
            <option value="set1">Set 1 (6-4)</option>
            <option value="set2">Set 2 (3-6)</option>
            <option value="set3">Set 3 (7-6)</option>
            <option value="set4">Set 4 (6-4)</option>
          </select>

          <button type="button" className="button button-quiet">
            <DownloadSimple size={14} weight="bold" />
            <span>Export CSV</span>
          </button>
        </div>
      </header>

      {/* Matchup Header Banner */}
      <div className="card analytics-matchup-bar">
        <div className="matchup-athlete">
          <div className="athlete-avatar-circle">CA</div>
          <div>
            <strong className="block text-sm">Carlos Alcaraz</strong>
            <small className="text-muted">ESP • Seed #1 • Winner</small>
          </div>
        </div>

        <div className="text-center">
          <div className="matchup-vs-badge">{MATCH_DATA.score}</div>
          <small className="text-muted block mt-1">{MATCH_DATA.tournament} • {MATCH_DATA.duration}</small>
        </div>

        <div className="matchup-athlete">
          <div className="text-right">
            <strong className="block text-sm">Jannik Sinner</strong>
            <small className="text-muted">ITA • Seed #2 • Runner-up</small>
          </div>
          <div className="athlete-avatar-circle athlete-secondary">JS</div>
        </div>
      </div>

      {/* Top 6 Analytical KPI Cards */}
      <div className="analytics-kpi-grid">
        <MetricCard
          label="1st Serve Win %"
          value="79%"
          subvalue="+6% vs Sinner"
          variant="highlight"
          progress={79}
          progressColor="lime"
        />
        <MetricCard
          label="Break Pts Saved"
          value="71%"
          subvalue="5 of 7 saved"
          progress={71}
          progressColor="ink"
        />
        <MetricCard
          label="Net Approach %"
          value="76%"
          subvalue="26 / 34 points"
          progress={76}
          progressColor="lime"
        />
        <MetricCard
          label="Winners / UFE"
          value="1.86"
          subvalue="52 W / 28 UFE"
          progress={65}
          progressColor="charcoal"
        />
        <MetricCard
          label="Rally 9+ Shots"
          value="64%"
          subvalue="23 of 36 won"
          progress={64}
          progressColor="lime"
        />
        <MetricCard
          label="Max Serve Speed"
          value="138 mph"
          subvalue="Avg: 124 mph"
          progress={88}
          progressColor="ash"
        />
      </div>

      {/* Main Analysis Layout */}
      <div className="analytics-main-layout">
        {/* Left Column: Charts and Visual Breakdown */}
        <div className="analytics-charts-column">
          {/* Chart 1: Points Won by Game Bar Chart */}
          <div className="card panel-padding">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Scoring Distribution</div>
                <h2>Points Won by Game Sequence</h2>
              </div>
              <div className="flex gap-4 text-xs">
                <span className="flex items-center gap-1.5">
                  <span className="dot-lime"></span> Alcaraz
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="dot-dark"></span> Sinner
                </span>
              </div>
            </div>

            <div className="chart-container-inner">
              <svg viewBox="0 0 600 160" className="w-full h-40">
                {/* Horizontal grid lines */}
                <line x1="40" y1="20" x2="580" y2="20" stroke="#d2d2c8" strokeWidth="1" strokeDasharray="3,3" />
                <line x1="40" y1="60" x2="580" y2="60" stroke="#d2d2c8" strokeWidth="1" strokeDasharray="3,3" />
                <line x1="40" y1="100" x2="580" y2="100" stroke="#d2d2c8" strokeWidth="1" strokeDasharray="3,3" />
                <line x1="40" y1="140" x2="580" y2="140" stroke="#14140f" strokeWidth="1" />

                {/* Bars for 12 games */}
                {[
                  { g: "G1", p1: 6, p2: 2 },
                  { g: "G2", p1: 4, p2: 5 },
                  { g: "G3", p1: 7, p2: 3 },
                  { g: "G4", p1: 5, p2: 4 },
                  { g: "G5", p1: 6, p2: 1 },
                  { g: "G6", p1: 3, p2: 6 },
                  { g: "G7", p1: 8, p2: 6 },
                  { g: "G8", p1: 5, p2: 3 },
                  { g: "G9", p1: 7, p2: 5 },
                  { g: "G10", p1: 6, p2: 2 },
                  { g: "G11", p1: 5, p2: 4 },
                  { g: "G12", p1: 7, p2: 3 },
                ].map((item, idx) => {
                  const x = 55 + idx * 44;
                  const h1 = item.p1 * 12;
                  const h2 = item.p2 * 12;
                  return (
                    <g key={idx}>
                      <rect x={x} y={140 - h1} width="12" height={h1} rx="3" fill="#beff50" stroke="#14140f" strokeWidth="1" />
                      <rect x={x + 14} y={140 - h2} width="12" height={h2} rx="3" fill="#14140f" />
                      <text x={x + 13} y={154} fontSize="10" textAnchor="middle" fill="#6e6e64" fontFamily="monospace">
                        {item.g}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>

          {/* Chart 2: Cumulative Momentum Timeline */}
          <div className="card panel-padding">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Match Dynamic Flow</div>
                <h2>Cumulative Rally Momentum Index</h2>
              </div>
              <span className="status-badge status-signal">Alcaraz +18.4 net momentum</span>
            </div>

            <div className="chart-container-inner">
              <svg viewBox="0 0 600 130" className="w-full h-32">
                <line x1="40" y1="65" x2="580" y2="65" stroke="#d2d2c8" strokeWidth="1.5" />
                <text x="15" y="40" fontSize="9" fill="#166534" fontWeight="bold">+MOM</text>
                <text x="15" y="95" fontSize="9" fill="#991b1b" fontWeight="bold">-MOM</text>

                {/* Momentum Area Path */}
                <path
                  d="M 50 65 Q 90 25, 140 45 T 220 85 T 310 30 T 400 70 T 490 35 T 570 20 L 570 65 Z"
                  fill="#beff50"
                  fillOpacity="0.3"
                />
                {/* Momentum Line */}
                <path
                  d="M 50 65 Q 90 25, 140 45 T 220 85 T 310 30 T 400 70 T 490 35 T 570 20"
                  fill="none"
                  stroke="#14140f"
                  strokeWidth="2.5"
                />
                {/* Set Break Markers */}
                <line x1="180" y1="15" x2="180" y2="115" stroke="#6e6e64" strokeDasharray="3,3" />
                <text x="185" y="25" fontSize="9" fill="#6e6e64">Set 1</text>
                <line x1="330" y1="15" x2="330" y2="115" stroke="#6e6e64" strokeDasharray="3,3" />
                <text x="335" y="25" fontSize="9" fill="#6e6e64">Set 2</text>
                <line x1="460" y1="15" x2="460" y2="115" stroke="#6e6e64" strokeDasharray="3,3" />
                <text x="465" y="25" fontSize="9" fill="#6e6e64">Set 3</text>
              </svg>
            </div>
          </div>

          {/* Tri-split Row: Shot Type Donut + Serve Speed Curve + Placement Court */}
          <div className="analytics-tri-row">
            {/* Shot Type Breakdown Donut */}
            <div className="card panel-padding">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">Distribution</div>
                  <h3>Shot Types</h3>
                </div>
              </div>

              <div className="donut-chart-container flex justify-center py-2">
                <svg viewBox="0 0 100 100" className="w-28 h-28">
                  <circle cx="50" cy="50" r="38" fill="none" stroke="#beff50" strokeWidth="16" strokeDasharray="130 240" strokeDashoffset="0" />
                  <circle cx="50" cy="50" r="38" fill="none" stroke="#14140f" strokeWidth="16" strokeDasharray="70 240" strokeDashoffset="-130" />
                  <circle cx="50" cy="50" r="38" fill="none" stroke="#6e6e64" strokeWidth="16" strokeDasharray="40 240" strokeDashoffset="-200" />
                </svg>
              </div>

              <div className="shot-breakdown-legend">
                <div className="legend-row">
                  <span className="flex items-center gap-1.5"><span className="legend-dot bg-lime-primary"></span> Forehands</span>
                  <strong>54%</strong>
                </div>
                <div className="legend-row">
                  <span className="flex items-center gap-1.5"><span className="legend-dot bg-ink"></span> Backhands</span>
                  <strong>29%</strong>
                </div>
                <div className="legend-row">
                  <span className="flex items-center gap-1.5"><span className="legend-dot bg-ash"></span> Volleys / Smashes</span>
                  <strong>17%</strong>
                </div>
              </div>
            </div>

            {/* Serve Speed Distribution Curve */}
            <div className="card panel-padding">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">Velocity Curve</div>
                  <h3>Serve Speeds</h3>
                </div>
              </div>

              <div className="py-2">
                <svg viewBox="0 0 140 80" className="w-full h-24">
                  <path
                    d="M 10 70 Q 40 68, 60 30 T 90 15 T 120 65 T 135 70"
                    fill="none"
                    stroke="#14140f"
                    strokeWidth="2"
                  />
                  <path
                    d="M 10 70 Q 40 68, 60 30 T 90 15 T 120 65 T 135 70 L 135 70 L 10 70 Z"
                    fill="#beff50"
                    fillOpacity="0.4"
                  />
                  <line x1="85" y1="10" x2="85" y2="70" stroke="#14140f" strokeDasharray="2,2" />
                  <text x="88" y="24" fontSize="9" fontWeight="bold">128 mph (Peak)</text>
                </svg>
              </div>

              <div className="speed-meta-box">
                <div><span className="text-muted">1st Avg:</span> <strong>126 mph</strong></div>
                <div><span className="text-muted">2nd Avg:</span> <strong>102 mph</strong></div>
              </div>
            </div>

            {/* Serve Placement Diagram */}
            <div className="card panel-padding">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">Spatial Zones</div>
                  <h3>Serve Placement</h3>
                </div>
              </div>

              <div className="serve-placement-mini-court">
                <svg viewBox="0 0 120 70" className="w-full h-20">
                  <rect x="5" y="5" width="110" height="60" fill="#f5f5eb" stroke="#14140f" strokeWidth="1.5" />
                  <line x1="60" y1="5" x2="60" y2="65" stroke="#14140f" strokeWidth="1" />
                  <line x1="5" y1="35" x2="115" y2="35" stroke="#14140f" strokeWidth="1" strokeDasharray="2,2" />
                  {/* Heat circles */}
                  <circle cx="30" cy="15" r="10" fill="#beff50" fillOpacity="0.7" />
                  <text x="30" y="18" fontSize="8" textAnchor="middle" fontWeight="bold">42% T</text>
                  <circle cx="30" cy="55" r="8" fill="#beff50" fillOpacity="0.4" />
                  <text x="30" y="58" fontSize="8" textAnchor="middle">24% W</text>
                  <circle cx="90" cy="15" r="7" fill="#beff50" fillOpacity="0.4" />
                  <circle cx="90" cy="55" r="11" fill="#beff50" fillOpacity="0.8" />
                  <text x="90" y="58" fontSize="8" textAnchor="middle" fontWeight="bold">48% W</text>
                </svg>
              </div>

              <div className="serve-placement-legend text-xs text-muted mt-2">
                <span>Deuce Court: 42% T / 34% Body / 24% Wide</span>
                <span>Ad Court: 48% Wide / 22% Body / 30% T</span>
              </div>
            </div>
          </div>

          {/* Key Stats Comparison Table */}
          <div className="card panel-padding">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Head-to-Head Telemetry</div>
                <h2>Detailed Match Statistics</h2>
              </div>
            </div>

            <div className="comparison-table-wrapper">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th className="text-left font-bold text-ink">C. Alcaraz</th>
                    <th className="text-center font-bold">Metric</th>
                    <th className="text-right font-bold text-ink">J. Sinner</th>
                  </tr>
                </thead>
                <tbody>
                  {STATS_COMPARISON.map((row, idx) => (
                    <tr key={idx}>
                      <td className={`text-left font-mono ${row.winner === "p1" ? "font-bold text-positive" : ""}`}>
                        {row.p1}
                      </td>
                      <td className="text-center text-xs text-muted font-medium">
                        {row.metric}
                      </td>
                      <td className={`text-right font-mono ${row.winner === "p2" ? "font-bold text-positive" : ""}`}>
                        {row.p2}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Column: AI Match Synthesis Report */}
        <div className="analytics-sidebar-column">
          <div className="card panel-padding ai-report-card">
            <div className="ai-report-header">
              <div className="ai-badge">
                <Sparkle size={16} weight="fill" className="text-lime-primary" />
                <strong>AI Match Intelligence Report</strong>
              </div>
              <span className="status-badge">Model v2.4</span>
            </div>

            <div>
              <span className="eyebrow">Executive Summary</span>
              <p className="text-xs leading-relaxed text-muted mt-1">
                Alcaraz controlled the baseline exchange velocity by targeting Sinner’s deep backhand corner in +5 shot rallies, neutralizing Sinner’s down-the-line transition.
              </p>
            </div>

            <div>
              <span className="eyebrow">Tactical Takeaways</span>
              <ul className="ai-takeaways-list">
                <li>
                  <span className="bullet-indicator"></span>
                  <div>
                    <strong>Heavy Forehand Cross-Court:</strong> Generated 18 forced errors with average spin of 3,240 RPM.
                  </div>
                </li>
                <li>
                  <span className="bullet-indicator"></span>
                  <div>
                    <strong>Drop Shot Disruption:</strong> 9 of 12 drop shots resulted in won points, forcing Sinner forward from deep baseline positions.
                  </div>
                </li>
                <li>
                  <span className="bullet-indicator"></span>
                  <div>
                    <strong>First Serve Plus One:</strong> Won 82% of points when first serve landed wide on the Ad-court.
                  </div>
                </li>
              </ul>
            </div>

            <div>
              <span className="eyebrow">Efficiency Ratings</span>
              <div className="efficiency-gauges">
                <div className="efficiency-gauge-box">
                  <div className="gauge-ring gauge-lime">92</div>
                  <span className="text-xs text-muted font-medium">Alcaraz Score</span>
                </div>
                <div className="efficiency-gauge-box">
                  <div className="gauge-ring gauge-charcoal">84</div>
                  <span className="text-xs text-muted font-medium">Sinner Score</span>
                </div>
              </div>
            </div>

            <div>
              <span className="eyebrow">Set Progression</span>
              <div className="set-scoreline-rows">
                <div className="set-row">
                  <span>Set 1</span>
                  <strong>6 <span className="vs-sep">-</span> 4</strong>
                  <span className="text-xs text-muted">Alcaraz +1 Break</span>
                </div>
                <div className="set-row">
                  <span>Set 2</span>
                  <strong>3 <span className="vs-sep">-</span> 6</strong>
                  <span className="text-xs text-muted">Sinner +1 Break</span>
                </div>
                <div className="set-row">
                  <span>Set 3</span>
                  <strong>7 <span className="vs-sep">-</span> 6 (5)</strong>
                  <span className="text-xs text-muted">Tiebreak Decider</span>
                </div>
                <div className="set-row">
                  <span>Set 4</span>
                  <strong>6 <span className="vs-sep">-</span> 4</strong>
                  <span className="text-xs text-muted">Championship Point</span>
                </div>
              </div>
            </div>

            <div className="pt-2">
              <Link to="/agent" className="button button-primary w-full">
                <Sparkle size={14} weight="fill" />
                <span>Ask AI Agent About Match</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
