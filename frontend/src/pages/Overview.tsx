import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
    ArrowUpRight,
  Sparkle,
  Television,
          ShieldCheck,
    CaretRight,
} from "@phosphor-icons/react";
import { MetricCard } from "../components/ui";
import { IntelligenceCarousel } from "../components/IntelligenceCarousel";

export const Overview: React.FC = () => {
  const [activeTourTab, setActiveTourTab] = useState<"ATP" | "WTA">("ATP");

  return (
    <div className="page">
      {/* 1. Header with Eyebrow, Main Title, and Actions */}
      <header className="page-header">
        <div>
          <div className="eyebrow">Executive Intelligence & Real-Time Circuit Hub</div>
          <h1>Tennis AI Operations Command Center</h1>
          <p>
            Autonomous multi-camera tracking, frame-accurate computer-vision analytics, player performance scouting, and strategic AI agent dialogues.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="status-badge status-signal">
            <span className="live-dot-green"></span>
            <span>Vision Core Online (120 FPS)</span>
          </div>
          <Link to="/match-center" className="button button-primary">
            <Television size={14} weight="bold" />
            <span>Launch Match Center</span>
          </Link>
        </div>
      </header>

      {/* 2. Top 4 High-Density KPI Cards */}
      <div className="overview-kpi-row">
        <MetricCard
          label="Tracked Matches (2026)"
          value="1,428"
          subvalue="+18 this week"
          trend="+12.4%"
          trendPositive={true}
          progress={78}
          progressColor="lime"
          variant="highlight"
        />
        <MetricCard
          label="Active Tracking Accuracy"
          value="98.7%"
          subvalue="Hawk-Eye certified homography"
          trend="+0.3%"
          trendPositive={true}
          progress={98}
          progressColor="ink"
        />
        <MetricCard
          label="Processed Rally Frames"
          value="24.8M"
          subvalue="120 FPS latency: 42ms"
          progress={85}
          progressColor="charcoal"
        />
        <MetricCard
          label="AI Scouting Directives"
          value="8,920"
          subvalue="142 player profiles mapped"
          progress={64}
          progressColor="ash"
        />
      </div>

      {/* 3. Hero Command Center Grid (Match Broadcast + Court Snapshot + Circuit Standings) */}
      <div className="overview-hero-layout">
        {/* Matchup Hero Card */}
        <div className="card panel-padding intelligence-hero-card">
          <div className="intelligence-head">
            <div className="intelligence-badge">
              <span className="live-pill">LIVE NOW</span>
              <span className="tour-badge-text">US Open 2026 • Championship Final</span>
            </div>
            <div className="confidence-chip">
              <Sparkle size={13} weight="fill" className="text-lime-primary" />
              <span>AI Win Probability: 74% Alcaraz</span>
            </div>
          </div>

          <div className="matchup-scoreboard-stage">
            <div className="athlete-profile-cell">
              <div className="athlete-avatar-box">CA</div>
              <div className="athlete-name-group">
                <strong>Carlos Alcaraz</strong>
                <small>ESP • ATP #1 • Serving</small>
              </div>
            </div>

            <div className="match-score-board">
              <div className="score-row-p1">
                <span className="score-set-num">6</span>
                <span className="score-set-num">3</span>
                <span className="score-set-num score-set-active">5</span>
              </div>
              <div className="score-row-p2">
                <span className="score-set-num">4</span>
                <span className="score-set-num">6</span>
                <span className="score-set-num score-set-active">4</span>
              </div>
              <div className="score-game-state">Set 3 • 40-30</div>
            </div>

            <div className="athlete-profile-cell flex-row-reverse text-right">
              <div className="athlete-avatar-box avatar-dark">JS</div>
              <div className="athlete-name-group">
                <strong>Jannik Sinner</strong>
                <small>ITA • ATP #2 • Receiving</small>
              </div>
            </div>
          </div>

          <div className="win-prob-section">
            <div className="win-prob-labels">
              <span className="font-semibold text-xs">Alcaraz (74%)</span>
              <span className="font-semibold text-xs">Sinner (26%)</span>
            </div>
            <div className="progress-track">
              <div className="progress-bar progress-lime" style={{ width: "74%" }}></div>
            </div>
          </div>

          <div className="intelligence-footer">
            <span className="live-meta-note">
              <ShieldCheck size={14} weight="fill" className="text-positive" />
              <span>Court 1 Arthur Ashe • 120 FPS Optical Ingestion</span>
            </span>
            <Link to="/match-center" className="button button-primary">
              <span>Open Match Operations</span>
              <ArrowUpRight size={14} weight="bold" />
            </Link>
          </div>
        </div>

        {/* Live Court Snapshot */}
        <div className="card panel-padding snapshot-hero-card">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Real-Time Spatial Model</div>
              <h2>Court Telemetry Snapshot</h2>
            </div>
            <span className="status-badge">Point 42</span>
          </div>

          <div className="snapshot-court-wrapper">
            <svg viewBox="0 0 200 120" className="w-full h-auto">
              <rect x="10" y="10" width="180" height="100" fill="#f5f5eb" stroke="#14140f" strokeWidth="1.5" />
              <line x1="100" y1="10" x2="100" y2="110" stroke="#14140f" strokeWidth="1" strokeDasharray="2,2" />
              <rect x="40" y="25" width="120" height="70" fill="none" stroke="#14140f" strokeWidth="1" />
              <line x1="40" y1="60" x2="160" y2="60" stroke="#14140f" strokeWidth="1" />
              {/* Ball Trajectory Vector */}
              <path d="M 30 35 Q 90 70, 165 40" fill="none" stroke="#14140f" strokeWidth="2" strokeDasharray="3,3" />
              <circle cx="165" cy="40" r="4" fill="#beff50" stroke="#14140f" strokeWidth="1.5" />
              {/* Player Nodes */}
              <circle cx="30" cy="35" r="5" fill="#beff50" stroke="#14140f" strokeWidth="1" />
              <circle cx="170" cy="75" r="5" fill="#14140f" />
            </svg>
          </div>

          <div className="rally-insight-banner">
            <div className="insight-icon-box">
              <Sparkle size={14} weight="fill" />
            </div>
            <div className="insight-text">
              <strong>Forehand Cross-Court Angle: 42°</strong>
              <p>Alcaraz generated 3,420 RPM topspin landing 0.4m from ad sideline.</p>
            </div>
          </div>
        </div>

        {/* ATP / WTA Top Players Widget */}
        <div className="card panel-padding top-players-card">
          <div className="section-heading">
            <div>
              <div className="eyebrow">World Standings</div>
              <h2>Top Athletes</h2>
            </div>
            <div className="tour-tabs-mini">
              <button
                type="button"
                className={`tab-btn-mini ${activeTourTab === "ATP" ? "tab-active" : ""}`}
                onClick={() => setActiveTourTab("ATP")}
              >
                ATP
              </button>
              <button
                type="button"
                className={`tab-btn-mini ${activeTourTab === "WTA" ? "tab-active" : ""}`}
                onClick={() => setActiveTourTab("WTA")}
              >
                WTA
              </button>
            </div>
          </div>

          <div className="players-rank-list">
            {[
              { rank: 1, name: "Carlos Alcaraz", country: "ESP", points: "9,840", active: true },
              { rank: 2, name: "Jannik Sinner", country: "ITA", points: "9,410", active: false },
              { rank: 3, name: "Novak Djokovic", country: "SRB", points: "6,920", active: false },
              { rank: 4, name: "Alexander Zverev", country: "GER", points: "6,480", active: false },
            ].map((p) => (
              <div key={p.rank} className="player-rank-row">
                <span className="rank-idx">0{p.rank}</span>
                <div className="player-avatar-mini">{p.name.split(" ").map((n) => n[0]).join("")}</div>
                <div className="player-info-mini">
                  <strong>{p.name}</strong>
                  <small>{p.country} • {p.points} pts</small>
                </div>
                <Link to="/players" className="icon-button" aria-label={`View ${p.name} profile`}>
                  <CaretRight size={14} weight="bold" />
                </Link>
              </div>
            ))}
          </div>

          <Link to="/players" className="text-link-block">
            <span>View Full 100-Player Roster</span>
            <ArrowUpRight size={14} weight="bold" />
          </Link>
        </div>
      </div>

      {/* 4. Mid-Grid: Upcoming Matches, Insights Feed & Tournament Focus */}
      <div className="overview-mid-grid">
        {/* Upcoming Circuit Matches */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Upcoming Schedule</div>
              <h2>Today's Fixtures</h2>
            </div>
            <Link to="/tournaments" className="text-link">
              <span>View All</span>
              <CaretRight size={12} weight="bold" />
            </Link>
          </div>

          <div className="upcoming-matches-list">
            <div className="upcoming-item">
              <div className="upcoming-meta">
                <span>Arthur Ashe Stadium • 18:30 EST</span>
                <span className="status-badge">Scheduled</span>
              </div>
              <div className="upcoming-players">
                <strong>Iga Swiatek (1)</strong>
                <span className="vs-tag">vs</span>
                <strong>Aryna Sabalenka (2)</strong>
              </div>
            </div>

            <div className="upcoming-item">
              <div className="upcoming-meta">
                <span>Louis Armstrong • 20:00 EST</span>
                <span className="status-badge">Warmup</span>
              </div>
              <div className="upcoming-players">
                <strong>Daniil Medvedev (5)</strong>
                <span className="vs-tag">vs</span>
                <strong>Taylor Fritz (12)</strong>
              </div>
            </div>
          </div>
        </div>

        {/* AI Tactical Insights Feed */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">AI Feed</div>
              <h2>Real-Time Insights</h2>
            </div>
          </div>

          <div className="insights-feed-list">
            <div className="insight-feed-item">
              <div className="insight-icon-circle">
                <Sparkle size={13} weight="fill" />
              </div>
              <div className="insight-feed-content">
                <strong>Alcaraz 2nd Serve Kick Exploit</strong>
                <p>Kicking out wide to Sinner’s backhand on Ad-court yielded 82% unreturned ball rate.</p>
                <small className="text-muted">2 mins ago • Match Center Model</small>
              </div>
            </div>

            <div className="insight-feed-item">
              <div className="insight-icon-circle">
                <Sparkle size={13} weight="fill" />
              </div>
              <div className="insight-feed-content">
                <strong>Sinner Deep Return Recovery</strong>
                <p>Stepping 1.2m inside baseline on 2nd serves increased winner probability by 28%.</p>
                <small className="text-muted">8 mins ago • Scouting Engine</small>
              </div>
            </div>
          </div>
        </div>

        {/* Tournament Circuit Focus */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Circuit Focus</div>
              <h2>US Open 2026</h2>
            </div>
            <span className="status-badge status-signal">Hard Court</span>
          </div>

          <div className="tournament-focus-list">
            <div className="focus-item">
              <span className="text-muted text-xs">Total Prize Purse:</span>
              <strong className="text-sm block">$75,000,000 USD</strong>
            </div>
            <div className="focus-item">
              <span className="text-muted text-xs">Surface Velocity:</span>
              <strong className="text-sm block">Laykold Fast Hard (CPR: 42.4)</strong>
            </div>
            <div className="focus-item">
              <span className="text-muted text-xs">Draw Progression:</span>
              <div className="progress-track mt-1">
                <div className="progress-bar progress-lime" style={{ width: "95%" }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 5. Bottom Grid: Performance Trends Chart & System Health */}
      <div className="overview-bottom-grid">
        <div className="card panel-padding trends-chart-card">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Historical Trends</div>
              <h2>Rally Length & Serve Efficiency Variance</h2>
            </div>
            <div className="trends-filter-pills">
              <span className="pill-static">Last 10 Matches</span>
            </div>
          </div>

          <div className="chart-container-inner">
            <svg viewBox="0 0 500 120" className="w-full h-28">
              <line x1="30" y1="20" x2="480" y2="20" stroke="#d2d2c8" strokeDasharray="3,3" />
              <line x1="30" y1="60" x2="480" y2="60" stroke="#d2d2c8" strokeDasharray="3,3" />
              <line x1="30" y1="100" x2="480" y2="100" stroke="#14140f" />
              {/* Trend 1: 1st serve win rate */}
              <path
                d="M 40 80 Q 120 40, 200 55 T 320 30 T 440 25"
                fill="none"
                stroke="#14140f"
                strokeWidth="2.5"
              />
              {/* Trend 2: Rally win rate */}
              <path
                d="M 40 90 Q 120 70, 200 65 T 320 45 T 440 35"
                fill="none"
                stroke="#beff50"
                strokeWidth="2.5"
              />
            </svg>
          </div>

          <div className="trends-footer">
            <div className="trends-legend-row">
              <span className="flex items-center gap-1.5"><span className="dot-dark"></span> 1st Serve Win Rate (79%)</span>
              <span className="flex items-center gap-1.5"><span className="dot-lime"></span> Rally Win Rate &gt; 5 Shots (68%)</span>
            </div>
            <Link to="/analytics" className="text-link">
              <span>View Full Analytics</span>
              <ArrowUpRight size={14} weight="bold" />
            </Link>
          </div>
        </div>

        <div className="card panel-padding system-status-card">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Infrastructure</div>
              <h2>Computer Vision Core</h2>
            </div>
          </div>

          <div className="system-health-lines">
            <div className="health-line">
              <span>Ball Trajectory Tracking</span>
              <strong className="text-positive font-mono">120 FPS</strong>
            </div>
            <div className="health-line">
              <span>Player Keypoint Pose</span>
              <strong className="text-positive font-mono">17 Joints</strong>
            </div>
            <div className="health-line">
              <span>Court Homography Matrix</span>
              <strong className="text-positive font-mono">Calibrated</strong>
            </div>
          </div>

          <Link to="/system" className="button button-quiet w-full">
            <span>Inspect System Telemetry</span>
            <ArrowUpRight size={14} weight="bold" />
          </Link>
        </div>
      </div>

      {/* 6. Embedded 7-Slide Reference Carousel */}
      <IntelligenceCarousel />
    </div>
  );
};

export default Overview;
