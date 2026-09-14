import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
          Sparkle,
  ArrowUpRight,
    } from "@phosphor-icons/react";

interface PlayerProfile {
  id: string;
  name: string;
  rank: number;
  country: string;
  age: number;
  height: string;
  plays: string;
  titles: number;
  points: string;
  form: string[];
  winRate: {
    hard: number;
    clay: number;
    grass: number;
  };
  strengths: string[];
  weaknesses: string[];
}

const PLAYERS: PlayerProfile[] = [
  {
    id: "alcaraz",
    name: "Carlos Alcaraz",
    rank: 1,
    country: "ESP",
    age: 23,
    height: "183 cm",
    plays: "Right-handed (Two-handed backhand)",
    titles: 18,
    points: "9,840",
    form: ["W", "W", "W", "L", "W"],
    winRate: { hard: 82, clay: 88, grass: 85 },
    strengths: ["Explosive Forehand Spin (>3,200 RPM)", "Drop Shot Disruption", "Court Coverage Speed"],
    weaknesses: ["Occasional Unforced Errors on 2nd Return", "Over-aggressive on Break Points"],
  },
  {
    id: "sinner",
    name: "Jannik Sinner",
    rank: 2,
    country: "ITA",
    age: 25,
    height: "191 cm",
    plays: "Right-handed (Two-handed backhand)",
    titles: 16,
    points: "9,410",
    form: ["W", "W", "W", "W", "L"],
    winRate: { hard: 86, clay: 78, grass: 80 },
    strengths: ["Heavy Flat Acceleration", "2nd Serve Return Depth", "Clutch Baseline Consistency"],
    weaknesses: ["Vertical Movement against Drop Shots", "High Backhand Elevation on Heavy Spin"],
  },
  {
    id: "djokovic",
    name: "Novak Djokovic",
    rank: 3,
    country: "SRB",
    age: 39,
    height: "188 cm",
    plays: "Right-handed (Two-handed backhand)",
    titles: 99,
    points: "6,920",
    form: ["W", "L", "W", "W", "W"],
    winRate: { hard: 85, clay: 80, grass: 86 },
    strengths: ["Precision Return Depth", "Elite Tactical Defense", "Mental Composure in Tiebreaks"],
    weaknesses: ["Recovery Time in 5-set Marathons", "Overhead Smash Consistency"],
  },
  {
    id: "zverev",
    name: "Alexander Zverev",
    rank: 4,
    country: "GER",
    age: 29,
    height: "198 cm",
    plays: "Right-handed (Two-handed backhand)",
    titles: 23,
    points: "6,480",
    form: ["W", "W", "L", "W", "L"],
    winRate: { hard: 79, clay: 82, grass: 72 },
    strengths: ["First Serve Velocity (>135 mph)", "Backhand Cross-Court Wall", "Net Reach"],
    weaknesses: ["Second Serve Double Faults under Pressure", "Forehand Spin Depth"],
  },
];

export const Players: React.FC = () => {
  const [selectedPlayer1, setSelectedPlayer1] = useState<PlayerProfile>(PLAYERS[0]!);
  const [selectedPlayer2] = useState<PlayerProfile>(PLAYERS[1]!);

  return (
    <div className="page">
      {/* Header */}
      <header className="page-header">
        <div>
          <div className="eyebrow">Athlete Intelligence & H2H Comparison</div>
          <h1>Player Profiles & Head-to-Head</h1>
          <p>
            Detailed athletic bios, career progression, multi-surface win percentages, spatial baseline heatmaps, and head-to-head match records.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link to="/scouting" className="button button-quiet">
            <span>Scouting Report</span>
            <ArrowUpRight size={14} weight="bold" />
          </Link>
          <Link to="/agent" className="button button-primary">
            <Sparkle size={14} weight="fill" />
            <span>Compare with AI Copilot</span>
          </Link>
        </div>
      </header>

      {/* Main Players Layout */}
      <div className="players-layout-grid">
        {/* Left Column: Interactive Rankings List */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">ATP Top Rankings</div>
              <h2>Tour Roster</h2>
            </div>
          </div>

          <div className="rankings-interactive-list">
            {PLAYERS.map((p) => {
              const isP1 = selectedPlayer1.id === p.id;
                            return (
                <div
                  key={p.id}
                  className={`ranking-item-btn ${isP1 ? "ranking-item-selected" : ""}`}
                  onClick={() => setSelectedPlayer1(p)}
                  role="button"
                  tabIndex={0}
                >
                  <span className="rank-badge">#{p.rank}</span>
                  <div className="rank-player-details">
                    <div className="rank-name-row">
                      <strong>{p.name}</strong>
                      <small>{p.country} • {p.titles} Titles</small>
                    </div>
                    <div className="rank-form-strip">
                      {p.form.map((res, i) => (
                        <span key={i} className={`form-dot ${res === "W" ? "form-w" : "form-l"}`}>
                          {res}
                        </span>
                      ))}
                    </div>
                  </div>
                  <strong className="rank-points-text">{p.points} pts</strong>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Head-to-Head Detailed Comparison View */}
        <div className="players-detail-column">
          {/* Athlete Comparison Hero Card */}
          <div className="card panel-padding">
            <div className="h2h-matchup-row">
              {/* Player 1 Card */}
              <div className="athlete-card-side">
                <div className="athlete-lg-avatar">
                  {selectedPlayer1.name.split(" ").map((n) => n[0]).join("")}
                </div>
                <div className="athlete-lg-info">
                  <span className="eyebrow">ATP #{selectedPlayer1.rank} • {selectedPlayer1.country}</span>
                  <h2>{selectedPlayer1.name}</h2>
                  <p>{selectedPlayer1.age} yrs • {selectedPlayer1.height} • {selectedPlayer1.plays}</p>
                </div>
              </div>

              {/* H2H Center Badge */}
              <div className="h2h-center-pill">
                <span className="eyebrow">H2H Record</span>
                <strong>6 - 4</strong>
                <small className="text-muted">10 Encounters</small>
              </div>

              {/* Player 2 Card */}
              <div className="athlete-card-side flex-row-reverse text-right">
                <div className="athlete-lg-avatar avatar-dark">
                  {selectedPlayer2.name.split(" ").map((n) => n[0]).join("")}
                </div>
                <div className="athlete-lg-info">
                  <span className="eyebrow">ATP #{selectedPlayer2.rank} • {selectedPlayer2.country}</span>
                  <h2>{selectedPlayer2.name}</h2>
                  <p>{selectedPlayer2.age} yrs • {selectedPlayer2.height} • {selectedPlayer2.plays}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Key Metrics Comparison Grid */}
          <div className="h2h-metrics-grid">
            <div className="card panel-padding h2h-stat-box">
              <span className="h2h-stat-title">Career Titles</span>
              <div className="h2h-stat-bar-group">
                <strong className="font-mono text-positive">{selectedPlayer1.titles}</strong>
                <div className="progress-track flex-1">
                  <div className="progress-bar progress-lime" style={{ width: "60%" }}></div>
                </div>
                <strong className="font-mono">{selectedPlayer2.titles}</strong>
              </div>
            </div>

            <div className="card panel-padding h2h-stat-box">
              <span className="h2h-stat-title">1st Serve Win %</span>
              <div className="h2h-stat-bar-group">
                <strong className="font-mono text-positive">79%</strong>
                <div className="progress-track flex-1">
                  <div className="progress-bar progress-lime" style={{ width: "52%" }}></div>
                </div>
                <strong className="font-mono">75%</strong>
              </div>
            </div>

            <div className="card panel-padding h2h-stat-box">
              <span className="h2h-stat-title">Break Point Save %</span>
              <div className="h2h-stat-bar-group">
                <strong className="font-mono text-positive">68%</strong>
                <div className="progress-track flex-1">
                  <div className="progress-bar progress-lime" style={{ width: "54%" }}></div>
                </div>
                <strong className="font-mono">63%</strong>
              </div>
            </div>

            <div className="card panel-padding h2h-stat-box">
              <span className="h2h-stat-title">Net Approach Win %</span>
              <div className="h2h-stat-bar-group">
                <strong className="font-mono text-positive">76%</strong>
                <div className="progress-track flex-1">
                  <div className="progress-bar progress-lime" style={{ width: "58%" }}></div>
                </div>
                <strong className="font-mono">68%</strong>
              </div>
            </div>
          </div>

          {/* Mid Split: Surface Win Rates & Strengths/Weaknesses */}
          <div className="players-mid-split">
            {/* Surface Performance Breakdown */}
            <div className="card panel-padding">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">Surface Specialization</div>
                  <h2>Win Rate by Surface</h2>
                </div>
              </div>

              <div className="surface-stats-stack">
                <div className="surface-row">
                  <div className="surface-name-box">
                    <strong>Hard Court</strong>
                    <span className="text-xs text-muted block">DecoTurf / Laykold</span>
                  </div>
                  <div className="surface-bars">
                    <div className="bar-athlete">
                      <span className="w-12 text-muted">{selectedPlayer1.name.split(" ")[1]}:</span>
                      <div className="progress-track flex-1">
                        <div className="progress-bar progress-lime" style={{ width: `${selectedPlayer1.winRate.hard}%` }}></div>
                      </div>
                      <strong className="font-mono">{selectedPlayer1.winRate.hard}%</strong>
                    </div>
                    <div className="bar-athlete">
                      <span className="w-12 text-muted">{selectedPlayer2.name.split(" ")[1]}:</span>
                      <div className="progress-track flex-1">
                        <div className="progress-bar progress-ink" style={{ width: `${selectedPlayer2.winRate.hard}%` }}></div>
                      </div>
                      <strong className="font-mono">{selectedPlayer2.winRate.hard}%</strong>
                    </div>
                  </div>
                </div>

                <div className="surface-row">
                  <div className="surface-name-box">
                    <strong>Clay Court</strong>
                    <span className="text-xs text-muted block">Red Clay</span>
                  </div>
                  <div className="surface-bars">
                    <div className="bar-athlete">
                      <span className="w-12 text-muted">{selectedPlayer1.name.split(" ")[1]}:</span>
                      <div className="progress-track flex-1">
                        <div className="progress-bar progress-lime" style={{ width: `${selectedPlayer1.winRate.clay}%` }}></div>
                      </div>
                      <strong className="font-mono">{selectedPlayer1.winRate.clay}%</strong>
                    </div>
                    <div className="bar-athlete">
                      <span className="w-12 text-muted">{selectedPlayer2.name.split(" ")[1]}:</span>
                      <div className="progress-track flex-1">
                        <div className="progress-bar progress-ink" style={{ width: `${selectedPlayer2.winRate.clay}%` }}></div>
                      </div>
                      <strong className="font-mono">{selectedPlayer2.winRate.clay}%</strong>
                    </div>
                  </div>
                </div>

                <div className="surface-row">
                  <div className="surface-name-box">
                    <strong>Grass Court</strong>
                    <span className="text-xs text-muted block">Rye Grass</span>
                  </div>
                  <div className="surface-bars">
                    <div className="bar-athlete">
                      <span className="w-12 text-muted">{selectedPlayer1.name.split(" ")[1]}:</span>
                      <div className="progress-track flex-1">
                        <div className="progress-bar progress-lime" style={{ width: `${selectedPlayer1.winRate.grass}%` }}></div>
                      </div>
                      <strong className="font-mono">{selectedPlayer1.winRate.grass}%</strong>
                    </div>
                    <div className="bar-athlete">
                      <span className="w-12 text-muted">{selectedPlayer2.name.split(" ")[1]}:</span>
                      <div className="progress-track flex-1">
                        <div className="progress-bar progress-ink" style={{ width: `${selectedPlayer2.winRate.grass}%` }}></div>
                      </div>
                      <strong className="font-mono">{selectedPlayer2.winRate.grass}%</strong>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Strengths and Weaknesses Comparison */}
            <div className="card panel-padding">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">Scouting Profile</div>
                  <h2>Tactical Matrix</h2>
                </div>
              </div>

              <div className="sw-comparison-grid">
                <div className="sw-athlete-col">
                  <span className="sw-athlete-tag">{selectedPlayer1.name}</span>
                  <div className="sw-list">
                    {selectedPlayer1.strengths.map((s, i) => (
                      <div key={i} className="sw-item positive">
                        <span>+ {s}</span>
                      </div>
                    ))}
                    {selectedPlayer1.weaknesses.map((w, i) => (
                      <div key={i} className="sw-item negative">
                        <span>- {w}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="sw-athlete-col">
                  <span className="sw-athlete-tag tag-dark">{selectedPlayer2.name}</span>
                  <div className="sw-list">
                    {selectedPlayer2.strengths.map((s, i) => (
                      <div key={i} className="sw-item positive">
                        <span>+ {s}</span>
                      </div>
                    ))}
                    {selectedPlayer2.weaknesses.map((w, i) => (
                      <div key={i} className="sw-item negative">
                        <span>- {w}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Spatial Occupancy Comparison Maps */}
          <div className="card panel-padding">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Court Space Utilization</div>
                <h2>Baseline Movement & Position Density</h2>
              </div>
            </div>

            <div className="occupancy-dual-grid">
              <div className="occupancy-box">
                <div className="occupancy-header">
                  <strong>{selectedPlayer1.name}</strong>
                  <span className="text-xs text-muted">Avg Depth: 0.8m behind baseline</span>
                </div>
                <svg viewBox="0 0 200 110" className="w-full h-24">
                  <rect x="10" y="10" width="180" height="90" fill="#f5f5eb" stroke="#14140f" strokeWidth="1.5" />
                  <line x1="100" y1="10" x2="100" y2="100" stroke="#14140f" strokeWidth="1" />
                  <circle cx="150" cy="30" r="18" fill="#beff50" fillOpacity="0.7" />
                  <circle cx="140" cy="80" r="16" fill="#beff50" fillOpacity="0.5" />
                </svg>
              </div>

              <div className="occupancy-box">
                <div className="occupancy-header">
                  <strong>{selectedPlayer2.name}</strong>
                  <span className="text-xs text-muted">Avg Depth: 0.3m behind baseline</span>
                </div>
                <svg viewBox="0 0 200 110" className="w-full h-24">
                  <rect x="10" y="10" width="180" height="90" fill="#f5f5eb" stroke="#14140f" strokeWidth="1.5" />
                  <line x1="100" y1="10" x2="100" y2="100" stroke="#14140f" strokeWidth="1" />
                  <circle cx="145" cy="55" r="22" fill="#14140f" fillOpacity="0.4" />
                  <circle cx="65" cy="55" r="12" fill="#14140f" fillOpacity="0.3" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Players;
