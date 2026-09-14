import { copyFile, mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(scriptDir, "..");
const repoDir = path.resolve(frontendDir, "..");
const outputsDir = path.join(repoDir, "outputs");
const generatedDir = path.join(frontendDir, "src", "data", "generated");
const publicDemoDir = path.join(frontendDir, "public", "demo");
const sourceVideo = path.join(repoDir, "data", "sample_videos", "input_video.mp4");
const defaultRunId = "phase5_scoring_1";

const runLabels = {
  baseline_run_1: ["Baseline", "Baseline"],
  phase2_ball_1: ["Phase 2 Ball Tracking", "Phase 2"],
  phase2_yolo11_final: ["Phase 2 YOLO11", "Phase 2"],
  phase3_events_1: ["Phase 3 Events", "Phase 3"],
  phase4_line_calls_1: ["Phase 4 Line Calls", "Phase 4"],
  phase4_1_line_calls: ["Phase 4.1 Contact Correction", "Phase 4.1"],
  phase5_scoring_1: ["Phase 5 Scoring", "Phase 5"],
};

async function readJson(filePath, fallback = null) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return fallback;
    throw error;
  }
}

async function readYaml(filePath, fallback = null) {
  try {
    return YAML.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return fallback;
    throw error;
  }
}

function getEventCount(events) {
  return events?.events_count ?? events?.events?.length ?? 0;
}

function getLineCallCount(lineCalls) {
  return lineCalls?.line_calls_count ?? lineCalls?.line_calls?.length ?? 0;
}

function getFramesAndFps(detections, trajectories, metrics) {
  const frames =
    detections?.metadata?.frames ??
    trajectories?.total_frames ??
    trajectories?.metadata?.total_frames ??
    trajectories?.ball_trajectory?.length ??
    metrics?.video_metadata?.total_frames ??
    null;
  const fps =
    detections?.metadata?.fps ??
    trajectories?.fps ??
    trajectories?.metadata?.fps ??
    metrics?.video_metadata?.fps ??
    null;
  return { frames, fps };
}

function normalizePipelineMetrics(metrics, eventCount, lineCallCount) {
  if (!metrics) return null;
  return {
    pipelineFps:
      metrics.pipeline_fps ??
      metrics.pipeline_performance?.processing_fps ??
      metrics.pipeline_performance?.fps ??
      null,
    eventCount:
      metrics.events_scored ?? metrics.events ?? metrics.events_metrics?.events_count ?? eventCount,
    lineCallCount: metrics.line_calls ?? lineCallCount,
    courtReprojectionErrorPx:
      metrics.court_reprojection_error_px ??
      metrics.court_accuracy?.reprojection_error_px ??
      metrics.court_accuracy?.reprojection_error ??
      null,
    raw: metrics,
  };
}

function normalizePlayerMetrics(metrics) {
  if (!metrics) return null;
  const normalizePlayer = (player) => ({
    total_distance_m: player?.total_distance_m ?? player?.total_distance_meters ?? 0,
    avg_speed_kmh: player?.avg_speed_kmh ?? player?.average_speed_kmh ?? 0,
    coverage_pct: player?.coverage_pct ?? (player?.detection_rate != null ? player.detection_rate * 100 : 100),
  });
  return {
    player_1: normalizePlayer(metrics.player_1),
    player_2: normalizePlayer(metrics.player_2),
  };
}

function normalizeBallMetrics(metrics) {
  if (!metrics) return null;
  return {
    ...metrics,
    scientific_status: metrics.scientific_status ?? "EXPERIMENTAL_2D_ESTIMATE",
  };
}

function normalizeCourt(court) {
  if (!court) return null;
  return {
    courtKeypointsPx:
      court.court_keypoints_px ?? court.keypoints ?? court.detected_keypoints_14 ?? null,
    canonicalKeypointsM:
      court.canonical_keypoints_m ?? court.canonical_keypoints ?? court.canonical_keypoints_14 ?? null,
    homographyMatrix:
      court.homography_matrix ?? court.homography_matrix_3x3 ?? null,
    reprojectionErrorPx:
      court.reprojection_error_px ?? court.reprojection_error ?? null,
    isValid: court.is_valid ?? court.valid ?? false,
  };
}

function normalizeTrajectories(trajectories) {
  if (!trajectories) {
    return { ballTrajectory: [], player1CourtPositions: [], player2CourtPositions: [] };
  }
  return {
    ballTrajectory: trajectories.ball_trajectory ?? [],
    player1CourtPositions: trajectories.player_1_court_positions ?? [],
    player2CourtPositions: trajectories.player_2_court_positions ?? [],
  };
}

function normalizeDetections(detections) {
  if (!detections) return { metadata: null, frames: [] };
  return {
    metadata: detections.metadata ?? null,
    frames: (detections.frames ?? []).map((frame, index) => ({
      frame_index: frame.frame_index ?? index,
      player_1: frame.player_1?.bbox ? { bbox: frame.player_1.bbox } : null,
      player_2: frame.player_2?.bbox ? { bbox: frame.player_2.bbox } : null,
      ball: frame.ball
        ? {
            position_px: frame.ball.position_px ?? frame.ball.pixel_position ?? null,
            confidence: frame.ball.confidence ?? null,
            state: frame.ball.state ?? "MISSING",
          }
        : null,
    })),
  };
}

function normalizeLineCalls(lineCalls) {
  return (lineCalls?.line_calls ?? []).map((call) => ({
    ...call,
    decision: call.decision ?? "UNKNOWN",
    tracker_state: call.tracker_state ?? "MISSING",
    contact_patch_model:
      call.contact_patch_model ?? (call.effective_ball_radius_cm != null ? "LEGACY_EFFECTIVE_RADIUS" : "UNAVAILABLE"),
    contact_patch_radius_cm:
      call.contact_patch_radius_cm ?? call.effective_ball_radius_cm ?? 0,
    refinement_method: call.refinement_method ?? "UNAVAILABLE",
  }));
}

async function inspectRun(runEntry) {
  const runDir = path.join(outputsDir, runEntry.name);
  const files = (await readdir(runDir, { withFileTypes: true }))
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .sort();
  if (files.length === 0) {
    return null;
  }
  const fileStats = await Promise.all(files.map((name) => stat(path.join(runDir, name))));
  const analyzedAt = new Date(
    Math.max(...fileStats.map((entry) => entry.mtimeMs)),
  ).toISOString();

  const [detections, trajectories, court, playerMetrics, ballMetrics, events, lineCalls, metrics, matchState, scoringEvents, scoreHistory, runConfig] =
    await Promise.all([
      readJson(path.join(runDir, "detections.json")),
      readJson(path.join(runDir, "trajectories.json")),
      readJson(path.join(runDir, "court_geometry.json")),
      readJson(path.join(runDir, "player_metrics.json")),
      readJson(path.join(runDir, "ball_metrics.json")),
      readJson(path.join(runDir, "match_events.json")),
      readJson(path.join(runDir, "line_calls.json")),
      readJson(path.join(runDir, "metrics.json")),
      readJson(path.join(runDir, "match_state.json")),
      readJson(path.join(runDir, "scoring_events.json")),
      readJson(path.join(runDir, "score_history.json")),
      readYaml(path.join(runDir, "run_config.yaml")),
    ]);

  const { frames, fps } = getFramesAndFps(detections, trajectories, metrics);
  const [label, phase] = runLabels[runEntry.name] ?? [runEntry.name, "Analysis"];
  const sourceVideoPath = detections?.metadata?.video ?? "data/sample_videos/input_video.mp4";

  const summary = {
    id: runEntry.name,
    label,
    phase,
    status: "COMPLETED",
    analyzedAt,
    sourceVideoPath,
    frames,
    fps,
    durationSeconds: frames && fps ? Number((frames / fps).toFixed(3)) : null,
    eventCount: getEventCount(events),
    lineCallCount: getLineCallCount(lineCalls),
    artifacts: files,
    capabilities: {
      detections: Boolean(detections),
      trajectories: Boolean(trajectories),
      court: Boolean(court),
      players: Boolean(playerMetrics),
      ballMetrics: Boolean(ballMetrics),
      events: Boolean(events),
      lineCalls: Boolean(lineCalls),
      scoring: Boolean(matchState),
      video: files.includes("annotated.mp4"),
    },
  };

  return {
    summary,
    detail: {
      summary,
      detections: normalizeDetections(detections),
      trajectories: normalizeTrajectories(trajectories),
      court: normalizeCourt(court),
      playerMetrics: normalizePlayerMetrics(playerMetrics),
      ballMetrics: normalizeBallMetrics(ballMetrics),
      events: events?.events ?? [],
      lineCalls: normalizeLineCalls(lineCalls),
      pipelineMetrics: normalizePipelineMetrics(metrics, getEventCount(events), getLineCallCount(lineCalls)),
      matchState,
      scoringEvents: scoringEvents?.scoring_events ?? [],
      scoreHistory: Array.isArray(scoreHistory) ? scoreHistory : [],
      runConfig,
      videoUrl: runEntry.name === defaultRunId ? "/demo/input-video.mp4" : null,
    },
  };
}

async function main() {
  await mkdir(generatedDir, { recursive: true });
  await mkdir(publicDemoDir, { recursive: true });

  const runEntries = (await readdir(outputsDir, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory() && entry.name in runLabels)
    .sort((a, b) => a.name.localeCompare(b.name));

  const normalizedRuns = (await Promise.all(runEntries.map(inspectRun))).filter(Boolean);
  if (!normalizedRuns.some((run) => run.summary.id === defaultRunId)) {
    throw new Error(`Required demo run '${defaultRunId}' was not found in ${outputsDir}.`);
  }

  const snapshotDetails = Object.fromEntries(normalizedRuns.map((run) => {
    if (run.summary.id === defaultRunId) return [run.summary.id, run.detail];
    return [run.summary.id, {
      ...run.detail,
      detections: { metadata: run.detail.detections.metadata, frames: [] },
      trajectories: { ballTrajectory: [], player1CourtPositions: [], player2CourtPositions: [] },
    }];
  }));

  const snapshot = {
    schemaVersion: 1,
    generatedAt: normalizedRuns
      .map((run) => run.summary.analyzedAt)
      .sort()
      .at(-1),
    defaultRunId,
    runs: normalizedRuns.map((run) => run.summary),
    details: snapshotDetails,
  };

  await writeFile(
    path.join(generatedDir, "analysis-snapshot.json"),
    `${JSON.stringify(snapshot, null, 2)}\n`,
    "utf8",
  );
  await copyFile(sourceVideo, path.join(publicDemoDir, "input-video.mp4"));

  console.log(
    `Synced ${snapshot.runs.length} analysis runs and ${path.relative(repoDir, sourceVideo)}.`,
  );
}

await main();
