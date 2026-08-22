import { memo, useId } from "react";
import type { AnalysisRun } from "../data/contracts";
import { mapCourtPoint } from "../lib/analytics";
import { cn } from "../lib/cn";

const WIDTH = 420;
const HEIGHT = 680;
const PAD = 34;

function pathFromPoints(points: Array<[number, number] | null>, stride = 1) {
  return points
    .filter((point, index): point is [number, number] => Boolean(point) && index % stride === 0)
    .map((point, index) => {
      const mapped = mapCourtPoint([Math.max(0, Math.min(10.97, point[0])), Math.max(0, Math.min(23.77, point[1]))], WIDTH, HEIGHT, PAD);
      return `${index === 0 ? "M" : "L"}${mapped.x.toFixed(1)},${mapped.y.toFixed(1)}`;
    })
    .join(" ");
}

export const CourtDiagram = memo(function CourtDiagram({
  run,
  selectedEventId = null,
  compact = false,
  showPlayers = true,
  showBall = true,
  showEvents = true,
  className,
}: {
  run: AnalysisRun;
  selectedEventId?: number | null;
  compact?: boolean;
  showPlayers?: boolean;
  showBall?: boolean;
  showEvents?: boolean;
  className?: string;
}) {
  const titleId = useId();
  const clipId = useId();
  const trajectories = run.trajectories;
  const ballPath = trajectories
    ? pathFromPoints(
        trajectories.ballTrajectory.map((point) =>
          point.court_x_m == null || point.court_y_m == null ? null : [point.court_x_m, point.court_y_m],
        ),
        compact ? 3 : 1,
      )
    : "";
  const player1Path = trajectories ? pathFromPoints(trajectories.player1CourtPositions, compact ? 5 : 2) : "";
  const player2Path = trajectories ? pathFromPoints(trajectories.player2CourtPositions, compact ? 5 : 2) : "";
  const innerLeft = mapCourtPoint([1.37, 0], WIDTH, HEIGHT, PAD).x;
  const innerRight = mapCourtPoint([9.6, 0], WIDTH, HEIGHT, PAD).x;
  const serviceNear = mapCourtPoint([0, 5.485], WIDTH, HEIGHT, PAD).y;
  const serviceFar = mapCourtPoint([0, 18.285], WIDTH, HEIGHT, PAD).y;
  const centerX = mapCourtPoint([5.485, 0], WIDTH, HEIGHT, PAD).x;
  const netY = mapCourtPoint([0, 11.885], WIDTH, HEIGHT, PAD).y;

  return (
    <svg
      className={cn("court-diagram", compact && "court-diagram-compact", className)}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-labelledby={titleId}
    >
      <title id={titleId}>Canonical tennis court with player movement, ball trajectory, events, and line calls</title>
      <defs><clipPath id={clipId}><rect x={PAD} y={PAD} width={WIDTH - PAD * 2} height={HEIGHT - PAD * 2} rx="3" /></clipPath></defs>
      <rect className="court-surface" x="1" y="1" width={WIDTH - 2} height={HEIGHT - 2} rx="14" />
      <g className="court-grid">
        <rect x={PAD} y={PAD} width={WIDTH - PAD * 2} height={HEIGHT - PAD * 2} />
        <line x1={innerLeft} y1={PAD} x2={innerLeft} y2={HEIGHT - PAD} />
        <line x1={innerRight} y1={PAD} x2={innerRight} y2={HEIGHT - PAD} />
        <line x1={PAD} y1={serviceNear} x2={WIDTH - PAD} y2={serviceNear} />
        <line x1={PAD} y1={serviceFar} x2={WIDTH - PAD} y2={serviceFar} />
        <line x1={centerX} y1={serviceNear} x2={centerX} y2={serviceFar} />
        <line className="court-net" x1={PAD - 12} y1={netY} x2={WIDTH - PAD + 12} y2={netY} />
        <path d={`M${centerX},${PAD}v13 M${centerX},${HEIGHT - PAD}v-13`} />
      </g>
      <g clipPath={`url(#${clipId})`}>
        {showPlayers && player1Path ? <path className="player-trail player-trail-one" d={player1Path} /> : null}
        {showPlayers && player2Path ? <path className="player-trail player-trail-two" d={player2Path} /> : null}
        {showBall && ballPath ? <path className="ball-trail" d={ballPath} /> : null}
      </g>
      {showEvents ? run.events.map((event) => {
        if (!event.court_position_m) return null;
        const point = mapCourtPoint(event.court_position_m, WIDTH, HEIGHT, PAD);
        const selected = event.event_id === selectedEventId;
        return (
          <g className={cn("court-event", selected && "court-event-selected")} key={event.event_id} transform={`translate(${point.x},${point.y})`}>
            <circle r={selected ? 10 : 6} />
            {event.event_type === "BOUNCE" ? <path d="M-4 0h8M0-4v8" /> : <path d="M-4-4l8 8M4-4l-8 8" />}
          </g>
        );
      }) : null}
      {run.lineCalls.map((call) => {
        const point = mapCourtPoint(call.bounce_position_m, WIDTH, HEIGHT, PAD);
        return <rect key={`line-${call.event_id}`} className="line-call-marker" x={point.x - 7} y={point.y - 7} width="14" height="14" rx="2" />;
      })}
      <g className="court-labels">
        <text x={PAD} y={20}>0 M</text>
        <text x={PAD} y={HEIGHT - 12}>23.77 M</text>
        <text x={WIDTH - PAD} y={20} textAnchor="end">10.97 M</text>
      </g>
    </svg>
  );
});

