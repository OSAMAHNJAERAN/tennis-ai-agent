import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Trophy,
      CloudSun,
      ArrowUpRight,
      } from "@phosphor-icons/react";
import { MetricCard } from "../components/ui";

const SCHEDULE_DAYS = [
  { day: "Mon", date: "Sep 2", active: false, event: "Round 1 & 2", matches: 16 },
  { day: "Tue", date: "Sep 3", active: false, event: "Round 3", matches: 12 },
  { day: "Wed", date: "Sep 4", active: false, event: "Round of 16", matches: 8 },
  { day: "Thu", date: "Sep 5", active: false, event: "Quarter-Finals", matches: 4 },
  { day: "Fri", date: "Sep 6", active: false, event: "Semi-Finals", matches: 2 },
  { day: "Sun", date: "Sep 8", active: true, event: "Championship Final", matches: 1 },
];

const FIXTURES_TODAY = [
  {
    court: "Arthur Ashe Stadium",
    time: "16:00 EST",
    p1: "Carlos Alcaraz (1)",
    p2: "Jannik Sinner (2)",
    round: "Men's Singles Final",
    status: "Live in Progress",
  },
  {
    court: "Louis Armstrong Stadium",
    time: "13:00 EST",
    p1: "Iga Swiatek (1)",
    p2: "Aryna Sabalenka (2)",
    round: "Women's Singles Final",
    status: "Final: 6-3, 2-6, 6-4",
  },
];

const RACE_TO_TURIN = [
  { rank: 1, name: "Carlos Alcaraz", points: "9,840", qualified: true },
  { rank: 2, name: "Jannik Sinner", points: "9,410", qualified: true },
  { rank: 3, name: "Novak Djokovic", points: "6,920", qualified: true },
  { rank: 4, name: "Alexander Zverev", points: "6,480", qualified: true },
  { rank: 5, name: "Daniil Medvedev", points: "5,890", qualified: false },
];

export const Tournaments: React.FC = () => {
  const [selectedDay, setSelectedDay] = useState<string>("Sun");

  return (
    <div className="page">
      {/* Header */}
      <header className="page-header">
        <div>
          <div className="eyebrow">Circuit Operations & Draws</div>
          <h1>Tournament Circuit Hub</h1>
          <p>
            Grand Slam & Masters 1000 brackets, daily court schedules, race standings, venue weather telemetry, and historical results.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <select className="select-control" aria-label="Select tournament">
            <option>US Open 2026 (Grand Slam - Hard)</option>
            <option>Wimbledon 2026 (Grand Slam - Grass)</option>
            <option>Roland Garros 2026 (Grand Slam - Clay)</option>
            <option>Australian Open 2026 (Grand Slam - Hard)</option>
            <option>ATP Finals Turin 2026 (Hard - Indoor)</option>
          </select>

          <Link to="/match-center" className="button button-primary">
            <span>Active Match</span>
            <ArrowUpRight size={14} weight="bold" />
          </Link>
        </div>
      </header>

      {/* Top 4 Tournament KPIs */}
      <div className="tournaments-kpi-grid">
        <MetricCard
          label="Tournament Category"
          value="Grand Slam"
          subvalue="2,000 ATP Points"
          variant="highlight"
          progress={100}
          progressColor="lime"
        />
        <MetricCard
          label="Total Prize Purse"
          value="$75,000,000"
          subvalue="Winner: $3,600,000"
          progress={85}
          progressColor="ink"
        />
        <MetricCard
          label="Singles Draw Size"
          value="128 Players"
          subvalue="7 Rounds of Play"
          progress={92}
          progressColor="charcoal"
        />
        <MetricCard
          label="Court Surface & Speed"
          value="Laykold Hard"
          subvalue="CPR Index: 42.4 (Medium-Fast)"
          progress={68}
          progressColor="ash"
        />
      </div>

      {/* Schedule Timeline Bar */}
      <div className="card panel-padding">
        <div className="section-heading">
          <div>
            <div className="eyebrow">Tournament Timeline</div>
            <h2>Match Schedule & Rounds</h2>
          </div>
          <span className="status-badge status-signal">Finals Weekend</span>
        </div>

        <div className="schedule-timeline-grid">
          {SCHEDULE_DAYS.map((d) => (
            <div
              key={d.day}
              className={`schedule-day-box ${d.day === selectedDay ? "schedule-day-active" : ""}`}
              onClick={() => setSelectedDay(d.day)}
              role="button"
              tabIndex={0}
            >
              <div className="day-header">
                <span className="day-name">{d.day}</span>
                <strong className="day-date">{d.date}</strong>
              </div>
              <strong className="event-name">{d.event}</strong>
              <span className="event-cat">{d.matches} match{d.matches > 1 ? "es" : ""}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Main Tournament Grid: Draw Bracket Tree & Today's Fixtures */}
      <div className="tournaments-main-grid">
        {/* Left: Tournament Draw Bracket Tree */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Interactive Bracket</div>
              <h2>Championship Draw Tree (Men's Singles)</h2>
            </div>
          </div>

          <div className="bracket-tree-wrapper">
            {/* Quarter-Finals */}
            <div className="bracket-column">
              <div className="bracket-col-header">Quarter-Finals</div>
              <div className="bracket-match-node">
                <div className="node-player winner">
                  <strong>C. Alcaraz (1)</strong>
                  <span>6 6 6</span>
                </div>
                <div className="node-player">
                  <span>J. Lehecka (20)</span>
                  <span>4 2 3</span>
                </div>
              </div>

              <div className="bracket-match-node">
                <div className="node-player winner">
                  <strong>D. Medvedev (5)</strong>
                  <span>7 6 6</span>
                </div>
                <div className="node-player">
                  <span>A. de Minaur (10)</span>
                  <span>6 3 4</span>
                </div>
              </div>

              <div className="bracket-match-node">
                <div className="node-player winner">
                  <strong>A. Zverev (4)</strong>
                  <span>6 7 6</span>
                </div>
                <div className="node-player">
                  <span>T. Fritz (12)</span>
                  <span>4 6 3</span>
                </div>
              </div>

              <div className="bracket-match-node">
                <div className="node-player winner">
                  <strong>J. Sinner (2)</strong>
                  <span>6 6 7</span>
                </div>
                <div className="node-player">
                  <span>H. Rune (7)</span>
                  <span>2 4 6</span>
                </div>
              </div>
            </div>

            {/* Semi-Finals */}
            <div className="bracket-column">
              <div className="bracket-col-header">Semi-Finals</div>
              <div className="bracket-match-node bracket-node-spanning">
                <div className="node-player winner">
                  <strong>C. Alcaraz (1)</strong>
                  <span>6 7 6</span>
                </div>
                <div className="node-player">
                  <span>D. Medvedev (5)</span>
                  <span>4 5 3</span>
                </div>
              </div>

              <div className="bracket-match-node bracket-node-spanning">
                <div className="node-player winner">
                  <strong>J. Sinner (2)</strong>
                  <span>7 6 6</span>
                </div>
                <div className="node-player">
                  <span>A. Zverev (4)</span>
                  <span>6 3 4</span>
                </div>
              </div>
            </div>

            {/* Championship Final */}
            <div className="bracket-column">
              <div className="bracket-col-header">Championship Final</div>
              <div className="bracket-match-node bracket-node-final">
                <div className="node-player">
                  <strong>C. Alcaraz (1)</strong>
                  <span className="font-bold">6 3 7 6</span>
                </div>
                <div className="node-player">
                  <strong>J. Sinner (2)</strong>
                  <span className="font-bold">4 6 6 4</span>
                </div>
                <div className="bracket-final-tag">
                  <Trophy size={14} weight="fill" />
                  <span>Champion: C. Alcaraz</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Daily Matches & Tournament Insights */}
        <div className="tournaments-side-stack">
          {/* Matches Today */}
          <div className="card panel-padding">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Order of Play</div>
                <h2>Featured Fixtures</h2>
              </div>
            </div>

            <div className="fixtures-list">
              {FIXTURES_TODAY.map((f, idx) => (
                <div key={idx} className="fixture-item">
                  <div className="fixture-court-meta">
                    <strong>{f.court}</strong>
                    <span>{f.time} • {f.round}</span>
                  </div>
                  <div className="fixture-athletes">
                    <div className="fixture-athlete-row">
                      <span>{f.p1}</span>
                      <small className="text-muted">vs</small>
                      <span>{f.p2}</span>
                    </div>
                  </div>
                  <span className="status-badge status-signal text-xs">{f.status}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Tournament Insights Card */}
          <div className="card panel-padding">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Tournament Telemetry</div>
                <h2>Circuit Insights</h2>
              </div>
            </div>

            <div className="top-performer-box">
              <div className="performer-avatar">CA</div>
              <div className="performer-info">
                <strong>Tournament Ace Leader</strong>
                <small className="text-muted">Carlos Alcaraz • 78 Aces in 7 Matches</small>
              </div>
              <div className="performer-stat">
                <strong className="font-mono">78 Aces</strong>
              </div>
            </div>

            <div className="break-conversion-rates">
              <div className="rate-row">
                <span className="text-muted">Tournament Avg Serve:</span>
                <strong>118 mph</strong>
                <div className="progress-track flex-1">
                  <div className="progress-bar progress-ink" style={{ width: "72%" }}></div>
                </div>
              </div>
              <div className="rate-row">
                <span className="text-muted">Avg Match Length:</span>
                <strong>2h 44m</strong>
                <div className="progress-track flex-1">
                  <div className="progress-bar progress-lime" style={{ width: "65%" }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Row: Race to Turin & Venue Weather Telemetry */}
      <div className="tournaments-bottom-row">
        {/* Standings / Race to Turin */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">ATP Standings</div>
              <h2>Race to Turin 2026</h2>
            </div>
            <span className="status-badge">Live Points</span>
          </div>

          <div className="standings-table-wrapper">
            <table className="standings-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Player</th>
                  <th className="text-right">Points</th>
                  <th className="text-right">Status</th>
                </tr>
              </thead>
              <tbody>
                {RACE_TO_TURIN.map((p) => (
                  <tr key={p.rank}>
                    <td className="font-bold">{p.rank}</td>
                    <td>
                      <div className="standing-player-cell">
                        <strong>{p.name}</strong>
                      </div>
                    </td>
                    <td className="text-right font-mono">{p.points}</td>
                    <td className="text-right">
                      {p.qualified ? (
                        <span className="badge-win">Qualified</span>
                      ) : (
                        <span className="text-xs text-muted">Contending</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Featured Venues */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Court Facilities</div>
              <h2>Venues & Capacity</h2>
            </div>
          </div>

          <div className="venues-grid">
            <div className="venue-item">
              <strong>Arthur Ashe Stadium</strong>
              <span className="text-xs text-muted block">Capacity: 23,771 • Retractable Roof • Laykold Hard</span>
            </div>
            <div className="venue-item">
              <strong>Louis Armstrong Stadium</strong>
              <span className="text-xs text-muted block">Capacity: 14,053 • Retractable Roof • Laykold Hard</span>
            </div>
            <div className="venue-item">
              <strong>Grandstand Stadium</strong>
              <span className="text-xs text-muted block">Capacity: 8,125 • Outdoor • Laykold Hard</span>
            </div>
          </div>
        </div>

        {/* Weather & Court Conditions */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Atmospheric Telemetry</div>
              <h2>Weather & Court Speed</h2>
            </div>
          </div>

          <div className="weather-main-stat">
            <CloudSun size={28} weight="duotone" className="text-lime-primary" />
            <div>
              <strong className="text-2xl font-bold">24°C / 75°F</strong>
              <span className="text-xs text-muted block">Sunny • Wind 8 km/h NNW • Humidity 46%</span>
            </div>
          </div>

          <div className="weather-sub-grid">
            <div><span className="text-muted">Air Pressure:</span> <strong>1014 hPa</strong></div>
            <div><span className="text-muted">Ball Bounce Index:</span> <strong>High (+4%)</strong></div>
          </div>

          <div className="court-speed-bar">
            <div className="flex justify-between text-xs">
              <span className="text-muted">Court Pace Rating (CPR)</span>
              <strong className="font-mono font-bold">42.4 (Medium-Fast)</strong>
            </div>
            <div className="progress-track">
              <div className="progress-bar progress-lime" style={{ width: "70%" }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Tournaments;
