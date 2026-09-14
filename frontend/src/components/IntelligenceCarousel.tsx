import React, { useState, useEffect, useCallback, useRef } from "react";
import { Link } from "react-router-dom";
import {
  CaretLeft,
  CaretRight,
  ArrowsOut,
  X,
  ArrowUpRight,
    SquaresFour,
} from "@phosphor-icons/react";

export interface IntelligenceSlide {
  id: string;
  title: string;
  subtitle: string;
  badge: string;
  category: string;
  route: string;
  routeLabel: string;
  imagePath: string;
  description: string;
  tags: string[];
}

export const INTELLIGENCE_SLIDES: IntelligenceSlide[] = [
  {
    id: "overview",
    title: "01. Executive Command Center",
    subtitle: "Real-time circuit telemetry, KPI metrics & live match broadcast",
    badge: "Ref 01 - Overview",
    category: "System Pulse",
    route: "/",
    routeLabel: "Go to Overview",
    imagePath: "/references/ref-01-overview.png",
    description:
      "High-density command center orchestrating real-time score tracking, live AI win probabilities, tournament updates, active rally breakdown, and circuit standings.",
    tags: ["Live Telemetry", "Win Probability", "KPI Cards", "Match Center"],
  },
  {
    id: "match-center",
    title: "02. Tactical Match Center",
    subtitle: "Multi-modal video tracking, 2D court simulation & shot evidence",
    badge: "Ref 02 - Match Center",
    category: "Live Match",
    route: "/match-center",
    routeLabel: "Go to Match Center",
    imagePath: "/references/ref-02-match-center.png",
    description:
      "Live match operations room providing frame-synchronized video playback, real-time spatial court reconstruction, continuous heatmaps, point timelines, and automated referee challenge telemetry.",
    tags: ["Video Analysis", "Court 2D", "Spatial Heatmap", "Challenge AI"],
  },
  {
    id: "analytics",
    title: "03. Advanced Match Analytics",
    subtitle: "Rally momentum timelines, serve speed curves & distribution maps",
    badge: "Ref 03 - Analytics",
    category: "Deep Analytics",
    route: "/analytics",
    routeLabel: "Go to Analytics",
    imagePath: "/references/ref-03-analytics.png",
    description:
      "In-depth match metrics with game-by-game point trajectories, cumulative momentum index, shot breakdown donuts, serve speeds, and synthetic AI post-match reports.",
    tags: ["Rally Momentum", "Shot Breakdown", "Serve Speeds", "AI Report"],
  },
  {
    id: "players",
    title: "04. Player Profiles & H2H Comparison",
    subtitle: "Athlete intelligence, ranking ladders & surface win rates",
    badge: "Ref 04 - Players",
    category: "Athletes",
    route: "/players",
    routeLabel: "Go to Players",
    imagePath: "/references/ref-04-players.png",
    description:
      "Head-to-head comparison engine analyzing physical attributes, career head-to-head records, surface-by-surface win rate variances, recent match forms, strengths and weakness radar.",
    tags: ["Head-to-Head", "Rankings", "Surface Stats", "Form Dot Matrix"],
  },
  {
    id: "tournaments",
    title: "05. Tournament Circuit Hub",
    subtitle: "Interactive draw brackets, schedule timeline & venue insights",
    badge: "Ref 05 - Tournaments",
    category: "Circuit Operations",
    route: "/tournaments",
    routeLabel: "Go to Tournaments",
    imagePath: "/references/ref-05-tournaments.png",
    description:
      "ATP/WTA circuit command center displaying 6-day interactive timelines, multi-round tournament draw brackets, court schedules, live venue conditions, and race standings.",
    tags: ["Draw Bracket", "Live Fixtures", "Race to Turin", "Venue Conditions"],
  },
  {
    id: "scouting",
    title: "06. Opponent Scouting Intelligence",
    subtitle: "Serve direction gauges, pressure points & video clip tagging",
    badge: "Ref 06 - Scouting",
    category: "Tactical Scouting",
    route: "/scouting",
    routeLabel: "Go to Scouting",
    imagePath: "/references/ref-06-scouting.png",
    description:
      "Pre-match preparation board featuring first/second serve direction distribution gauges, return spatial court diagrams, break-point pressure conversion, weak-zone heatmap analysis, and tactical video clip bookmarks.",
    tags: ["Serve Direction", "Weak Zone Heatmap", "Tactical Clips", "Team Notes"],
  },
  {
    id: "agent",
    title: "07. Conversational Tennis AI Assistant",
    subtitle: "Match strategy dialogues, evidence retrieval & prompt suggestions",
    badge: "Ref 07 - Agent",
    category: "AI Copilot",
    route: "/agent",
    routeLabel: "Go to Agent",
    imagePath: "/references/ref-07-agent.png",
    description:
      "Autonomous conversational AI pairing tactical prompts with live computer-vision evidence, interactive shot distributions, tactical clips, and confidence meters.",
    tags: ["AI Dialogue", "Tactical Prompts", "Evidence Feed", "Model Confidence"],
  },
];

interface IntelligenceCarouselProps {
  className?: string;
  autoPlayInterval?: number;
}

export const IntelligenceCarousel: React.FC<IntelligenceCarouselProps> = ({
  className = "",
  autoPlayInterval = 0,
}) => {
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const currentSlide: IntelligenceSlide = INTELLIGENCE_SLIDES[currentIndex] ?? INTELLIGENCE_SLIDES[0]!;

  const handleNext = useCallback(() => {
    setCurrentIndex((prev) => (prev + 1) % INTELLIGENCE_SLIDES.length);
  }, []);

  const handlePrev = useCallback(() => {
    setCurrentIndex((prev) =>
      prev === 0 ? INTELLIGENCE_SLIDES.length - 1 : prev - 1
    );
  }, []);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isModalOpen) {
        if (e.key === "Escape") setIsModalOpen(false);
        if (e.key === "ArrowRight") handleNext();
        if (e.key === "ArrowLeft") handlePrev();
        return;
      }

      if (containerRef.current && containerRef.current.contains(document.activeElement)) {
        if (e.key === "ArrowRight") {
          e.preventDefault();
          handleNext();
        }
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          handlePrev();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleNext, handlePrev, isModalOpen]);

  // Autoplay support
  useEffect(() => {
    if (autoPlayInterval <= 0 || isModalOpen) return;
    const timer = setInterval(handleNext, autoPlayInterval);
    return () => clearInterval(timer);
  }, [autoPlayInterval, handleNext, isModalOpen]);

  // Touch Swipe Handlers
  const minSwipeDistance = 50;
  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0]?.clientX ?? null);
  };
  const onTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0]?.clientX ?? null);
  };
  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;
    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;
    if (isLeftSwipe) handleNext();
    if (isRightSwipe) handlePrev();
  };

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      className={`intelligence-carousel-card ${className}`}
      aria-roledescription="carousel"
      aria-label="Tennis AI Reference Dashboards"
      onTouchStart={onTouchStart}
      onTouchMove={onTouchMove}
      onTouchEnd={onTouchEnd}
    >
      {/* Header with Title and Nav Controls */}
      <div className="carousel-header">
        <div>
          <div className="carousel-eyebrow">
            <SquaresFour size={13} weight="bold" />
            <span>Architecture & Reference Matrix</span>
          </div>
          <h2 className="carousel-heading">Seven Core Intelligence Views</h2>
          <p className="carousel-subheading">
            Seamless multi-modal architecture aligning real-time tracking, scouting, and autonomous agent dialogues.
          </p>
        </div>

        <div className="carousel-controls">
          <div className="carousel-counter">
            <span className="font-bold">{String(currentIndex + 1).padStart(2, "0")}</span>
            <span className="text-muted">/</span>
            <span>{String(INTELLIGENCE_SLIDES.length).padStart(2, "0")}</span>
          </div>

          <button
            type="button"
            className="icon-button"
            onClick={handlePrev}
            aria-label="Previous intelligence view"
          >
            <CaretLeft size={18} weight="bold" />
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={handleNext}
            aria-label="Next intelligence view"
          >
            <CaretRight size={18} weight="bold" />
          </button>
        </div>
      </div>

      {/* Main Slide Presentation Stage */}
      <div className="carousel-stage-container">
        <div className="carousel-slide-content">
          {/* Metadata Pane */}
          <div className="carousel-meta-pane">
            <div className="carousel-meta-top">
              <span className="carousel-category-badge">{currentSlide.category}</span>
              <div className="carousel-badges">
                <span className="carousel-chip">{currentSlide.badge}</span>
              </div>
            </div>

            <div className="carousel-meta-main">
              <span className="carousel-slide-num">VIEW 0{currentIndex + 1}</span>
              <h3 className="carousel-slide-title">{currentSlide.title}</h3>
              <p className="carousel-slide-subtitle">{currentSlide.subtitle}</p>
              <p className="carousel-slide-desc">{currentSlide.description}</p>

              <div className="flex flex-wrap gap-1.5 mt-4">
                {currentSlide.tags.map((tag) => (
                  <span key={tag} className="status-badge">
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            <div className="carousel-meta-actions">
              <Link to={currentSlide.route} className="button button-primary">
                <span>{currentSlide.routeLabel}</span>
                <ArrowUpRight size={14} weight="bold" />
              </Link>
              <button
                type="button"
                className="button button-quiet"
                onClick={() => setIsModalOpen(true)}
              >
                <ArrowsOut size={14} weight="bold" />
                <span>Full Blueprint</span>
              </button>
            </div>
          </div>

          {/* Reference Image Preview Stage */}
          <div
            className="carousel-image-frame"
            onClick={() => setIsModalOpen(true)}
            role="button"
            tabIndex={0}
            aria-label={`View full resolution ${currentSlide.title}`}
          >
            <div className="carousel-image-wrapper">
              <img
                src={currentSlide.imagePath}
                alt={currentSlide.title}
                className="carousel-preview-img"
                loading="lazy"
              />
              <div className="carousel-image-overlay">
                <span className="carousel-expand-pill">
                  <ArrowsOut size={13} weight="bold" />
                  <span>Enlarge Blueprint</span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Indicator Navigation Bar */}
      <div className="carousel-indicators-bar">
        {INTELLIGENCE_SLIDES.map((slide, index) => {
          const isActive = index === currentIndex;
          return (
            <button
              key={slide.id}
              type="button"
              className={`carousel-dot-item ${isActive ? "carousel-dot-active" : ""}`}
              onClick={() => setCurrentIndex(index)}
              aria-label={`Slide ${index + 1}: ${slide.title}`}
            >
              <span className="carousel-dot-num">0{index + 1}</span>
              <span className="carousel-dot-label">{slide.title.replace(/^\d+\.\s*/, "")}</span>
            </button>
          );
        })}
      </div>

      {/* Full-Resolution Modal Dialog */}
      {isModalOpen && (
        <div
          className="carousel-modal-backdrop"
          onClick={() => setIsModalOpen(false)}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="carousel-modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="carousel-modal-header">
              <div className="carousel-modal-title">
                <strong>{currentSlide.title}</strong>
                <span>{currentSlide.subtitle}</span>
              </div>
              <div className="carousel-modal-actions">
                <Link
                  to={currentSlide.route}
                  className="button button-primary"
                  onClick={() => setIsModalOpen(false)}
                >
                  <span>Launch Live View</span>
                  <ArrowUpRight size={14} weight="bold" />
                </Link>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => setIsModalOpen(false)}
                  aria-label="Close blueprint viewer"
                >
                  <X size={18} weight="bold" />
                </button>
              </div>
            </div>
            <div className="carousel-modal-body">
              <img
                src={currentSlide.imagePath}
                alt={currentSlide.title}
                className="carousel-modal-img"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IntelligenceCarousel;
