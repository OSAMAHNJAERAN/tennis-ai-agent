import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
      Play,
  CheckCircle,
    Sparkle,
  ArrowUpRight,
  ShieldCheck,
    FloppyDisk,
} from "@phosphor-icons/react";

const OPPONENT = {
  name: "Jannik Sinner",
  rank: 2,
  country: "ITA",
  playStyle: "Aggressive Baseline / Heavy Strike",
  h2h: "Alcaraz leads 6-4",
  nextMatch: "US Open 2026 - Final (Sunday 16:00 EST)",
};

const TACTICAL_RECOMMENDATIONS = [
  {
    title: "Heavy Topspin to High Backhand",
    description:
      "Sinner struggles to generate flat acceleration above shoulder height on clay/hard courts. Attack with heavy cross-court forehands (>3,200 RPM).",
    impact: "High Impact",
  },
  {
    title: "Short Angle Drop Shots on Deep Baseline Recovery",
    description:
      "When Sinner recovers deep behind the baseline on backhand defense, utilize forehand drop shots to expose vertical court sprint time.",
    impact: "High Impact",
  },
  {
    title: "Body Serves in Ad Court on Pressure Points",
    description:
      "Sinner anticipates wide kicker on 30-40 and Ad-out. Jamming the hip restricts his explosive two-handed extension.",
    impact: "Medium Impact",
  },
];

const TAGGED_CLIPS = [
  {
    id: "clip-1",
    title: "Backhand forced error under high ball",
    set: "Set 1, Game 5 (15-40)",
    duration: "0:24",
    type: "Weakness Evidence",
  },
  {
    id: "clip-2",
    title: "Drop shot recovery deficit",
    set: "Set 2, Game 3 (30-30)",
    duration: "0:18",
    type: "Tactical Exploit",
  },
  {
    id: "clip-3",
    title: "Second serve return attack down the line",
    set: "Set 2, Game 7 (Deuce)",
    duration: "0:31",
    type: "Opponent Threat",
  },
  {
    id: "clip-4",
    title: "Forehand inside-out winner from deep ad-court",
    set: "Set 3, Game 2 (40-15)",
    duration: "0:20",
    type: "Opponent Threat",
  },
];

export const Scouting: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [notes, setNotes] = useState<string>(
    "Game Plan Notes: Focus on first-serve depth to T in Deuce court. Attack Sinner's 2nd serve early by stepping 1m inside baseline. Keep backhand exchanges cross-court until opening appears."
  );

  return (
    <div className="page">
      {/* Header */}
      <header className="page-header">
        <div>
          <div className="eyebrow">Tactical Preparation Hub</div>
          <h1>Opponent Scouting Intelligence</h1>
          <p>
            Synthesized AI scouting report, serve direction tendencies, spatial weak-zone maps, and tagged video evidence.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link to="/match-center" className="button button-quiet">
            <span>Live Match Center</span>
            <ArrowUpRight size={14} weight="bold" />
          </Link>
          <Link to="/agent" className="button button-primary">
            <Sparkle size={14} weight="fill" />
            <span>Ask Scouting Agent</span>
          </Link>
        </div>
      </header>

      {/* Opponent Card Banner */}
      <div className="card panel-padding scouting-matchup-top">
        <div className="scouting-player-box">
          <div className="avatar-circle avatar-dark">JS</div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold">Jannik Sinner</h2>
              <span className="status-badge">ATP #2 • ITA</span>
            </div>
            <p className="text-xs text-muted mt-0.5">{OPPONENT.playStyle} • {OPPONENT.nextMatch}</p>
          </div>
        </div>

        <div className="h2h-pill-row">
          <span className="text-xs text-muted font-medium">Head to Head:</span>
          <span className="status-badge status-signal font-mono font-bold">6 - 4</span>
          <span className="badge-win">Alcaraz Won Last 3</span>
        </div>
      </div>

      {/* Top 4 Scouting Tendency Cards */}
      <div className="scouting-tendencies-grid">
        {/* Tendency 1: Serve Direction Breakdown */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Serve Distribution</div>
              <h3>1st Serve Target</h3>
            </div>
          </div>

          <div className="tendency-stat-center">
            <div className="gauge-circle-lime">
              <strong>48%</strong>
              <small>Wide Target</small>
            </div>
            <div className="serve-split-list">
              <div className="split-row"><span>Wide Slice:</span> <strong>48%</strong></div>
              <div className="split-row"><span>T Flat:</span> <strong>34%</strong></div>
              <div className="split-row"><span>Body:</span> <strong>18%</strong></div>
            </div>
          </div>

          <div className="tendency-footer-stats">
            <span className="text-muted">Avg Speed: <strong>128 mph</strong></span>
            <span className="text-muted">Ace Rate: <strong>12.4%</strong></span>
          </div>
        </div>

        {/* Tendency 2: Return Tendencies */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Return Footprint</div>
              <h3>2nd Serve Attack</h3>
            </div>
          </div>

          <div className="return-court-diagram">
            <div className="return-stat-big">
              <strong className="text-positive">+1.2m</strong>
              <small className="text-muted">Inside Baseline</small>
            </div>
            <div className="mini-court-heatmap-box">
              <svg viewBox="0 0 100 60" className="w-full h-12">
                <rect x="5" y="5" width="90" height="50" fill="#f5f5eb" stroke="#14140f" strokeWidth="1" />
                <line x1="50" y1="5" x2="50" y2="55" stroke="#14140f" strokeWidth="1" />
                <circle cx="35" cy="45" r="10" fill="#beff50" fillOpacity="0.7" />
                <circle cx="75" cy="42" r="8" fill="#beff50" fillOpacity="0.5" />
              </svg>
            </div>
          </div>

          <div className="tendency-footer-stats">
            <span className="text-muted">Return Win %: <strong>38%</strong></span>
            <span className="text-muted">Aggressive: <strong>68%</strong></span>
          </div>
        </div>

        {/* Tendency 3: Pressure Points Conversion */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Clutch Performance</div>
              <h3>Break Point Tendency</h3>
            </div>
          </div>

          <div className="pressure-metrics-list">
            <div className="pressure-row">
              <span className="text-muted">BP Saved Rate:</span>
              <strong className="font-mono">61%</strong>
            </div>
            <div className="pressure-row">
              <span className="text-muted">Prefers Forehand Down-Line:</span>
              <strong className="font-mono">72%</strong>
            </div>
          </div>

          <div className="pressure-alert-callout">
            <ShieldCheck size={14} weight="fill" className="text-positive flex-shrink-0" />
            <span>Target backhand cross-court on 30-40</span>
          </div>
        </div>

        {/* Tendency 4: Baseline Rally Vectors */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Rally Pattern</div>
              <h3>Shot Direction Vector</h3>
            </div>
          </div>

          <div className="vector-quad-grid">
            <div className="vector-box">
              <strong>64%</strong>
              <small>FH Cross-Court</small>
            </div>
            <div className="vector-box">
              <strong>36%</strong>
              <small>FH Down Line</small>
            </div>
            <div className="vector-box">
              <strong>58%</strong>
              <small>BH Cross-Court</small>
            </div>
            <div className="vector-box">
              <strong>42%</strong>
              <small>BH Down Line</small>
            </div>
          </div>
        </div>
      </div>

      {/* Mid Grid: Weak Zone Heatmap & Tagged Video Clips */}
      <div className="scouting-mid-grid">
        {/* Left: Weak Zone Spatial Analysis */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Vulnerability Mapping</div>
              <h2>High-Vulnerability Spatial Zone</h2>
            </div>
            <span className="status-badge status-signal">Exploit Probability 84%</span>
          </div>

          <div className="weak-zone-content">
            <div className="court-heatmap-visual">
              <svg viewBox="0 0 300 180" className="w-full h-auto">
                <rect x="15" y="15" width="270" height="150" fill="#f5f5eb" stroke="#14140f" strokeWidth="2" />
                <line x1="150" y1="15" x2="150" y2="165" stroke="#14140f" strokeWidth="1.5" strokeDasharray="4,4" />
                <rect x="60" y="35" width="180" height="110" fill="none" stroke="#14140f" strokeWidth="1" />
                <line x1="60" y1="90" x2="240" y2="90" stroke="#14140f" strokeWidth="1" />

                {/* Red/Amber Weak Zone Marker */}
                <ellipse cx="230" cy="45" rx="35" ry="22" fill="#fee2e2" stroke="#ef4444" strokeWidth="2" strokeDasharray="3,3" />
                <text x="230" y="48" fontSize="10" fontWeight="bold" textAnchor="middle" fill="#991b1b">
                  HIGH BACKHAND ZONE
                </text>

                {/* Drop Shot Exploitation Zone */}
                <ellipse cx="85" cy="80" rx="20" ry="15" fill="#beff50" fillOpacity="0.5" stroke="#14140f" strokeWidth="1" />
                <text x="85" y="83" fontSize="8" fontWeight="bold" textAnchor="middle">
                  DROP EXPLOIT
                </text>
              </svg>
            </div>

            <div className="weakness-callout-box">
              <span className="eyebrow">Tactical Finding</span>
              <h3>Ad-Court High Elevation</h3>
              <p className="text-xs text-muted leading-relaxed">
                When balls bounce over 1.4m on Sinner’s backhand side, his unforced error rate rises from 11% to 34%. He generates 18% less topspin RPM from this coordinate.
              </p>
              <div className="compact-facts mt-3 pt-3 border-t border-hairline">
                <div className="flex justify-between">
                  <span className="text-muted">Topspin Vulnerability:</span>
                  <strong>8.4 / 10</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Lateral Movement Deficit:</span>
                  <strong>+0.32s recovery</strong>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Tagged Scouting Clips Browser */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Video Library</div>
              <h2>Tagged Scouting Clips</h2>
            </div>
          </div>

          <div className="clips-filter-pills">
            {["all", "Weakness Evidence", "Tactical Exploit", "Opponent Threat"].map((cat) => (
              <button
                key={cat}
                type="button"
                className={`pill-btn ${selectedCategory === cat ? "pill-active" : ""}`}
                onClick={() => setSelectedCategory(cat)}
              >
                {cat === "all" ? "All Clips" : cat}
              </button>
            ))}
          </div>

          <div className="clips-scroll-list">
            {TAGGED_CLIPS.filter(
              (c) => selectedCategory === "all" || c.type === selectedCategory
            ).map((clip) => (
              <div key={clip.id} className="clip-row-item">
                <div className="clip-thumb-box">
                  <Play size={14} weight="fill" />
                  <span className="clip-time-tag">{clip.duration}</span>
                </div>
                <div className="clip-info">
                  <strong>{clip.title}</strong>
                  <small>{clip.set}</small>
                  <p className="text-xs text-muted">{clip.type}</p>
                </div>
                <Link to="/match-center" className="button button-quiet text-xs">
                  <span>Analyze</span>
                </Link>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Grid: Tactical Recommendations & Coach Notes */}
      <div className="scouting-bottom-grid">
        {/* Tactical Playbook List */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Coaching Playbook</div>
              <h2>Key Tactical Directives</h2>
            </div>
          </div>

          <div className="tactical-recs-list">
            {TACTICAL_RECOMMENDATIONS.map((rec, idx) => (
              <div key={idx} className="rec-item">
                <span
                  className={`rec-impact-badge ${rec.impact === "High Impact" ? "impact-high" : "impact-medium"}`}
                >
                  {rec.impact}
                </span>
                <div className="rec-text">
                  <strong>{rec.title}</strong>
                  <p>{rec.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Team Collaboration Notes */}
        <div className="card panel-padding">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Team Collaboration</div>
              <h2>Coaching Staff Notes</h2>
            </div>
            <span className="status-badge">Auto-Saved</span>
          </div>

          <textarea
            className="scouting-notes-textarea"
            rows={5}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Type confidential match preparation notes..."
          />

          <div className="notes-footer-actions">
            <div className="notes-sync-status">
              <CheckCircle size={14} weight="fill" className="text-positive" />
              <span>Synced with Coaching Tablet</span>
            </div>
            <button type="button" className="button button-primary">
              <FloppyDisk size={14} weight="bold" />
              <span>Save Directives</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Scouting;
