"""Recheck the frozen public-ball diagnostic comparison without inference.

Only a new JSON receipt may be written. Frozen labels, protocol, comparison,
and product evidence are read and hashed before and after the audit. To repeat
the audit, supply --receipt with another unused JSON name in this directory.
"""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
EXPECTED_PROTOCOL_SHA256 = "fb8eb06010f999262c2c3e911418650d435d4647397d433253e4e89c1302533b"
EXPECTED_COMPARISON_SHA256 = "677d47e20ea56557b2f0b84e01784e3635eecb877eda25ca42bc8c7f021fbe85"
EXPECTED_COMPARISON_SCRIPT_SHA256 = "a949aa1b2d2e3955bfea3457e2b9f0ced6a24a0bc783cb03bcff5942c7d11e03"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=BASE / "independent_saved_control_audit_receipt_v1.json")
    args = parser.parse_args()
    destination = args.receipt.resolve()
    require(destination.parent == BASE and destination.suffix == ".json", "Receipt must be a new JSON file in the audit directory.")
    require(not destination.exists(), "Existing files must not be overwritten; use a new receipt name.")

    protocol_path = BASE / "protocol.json"
    comparison_path = BASE / "saved_control_comparison.json"
    comparison_script = ROOT / "scripts/evaluate/compare_public_ball_blind_review.py"
    manifest_path = ROOT / "data/demos/sources.json"
    require(digest(protocol_path) == EXPECTED_PROTOCOL_SHA256, "Frozen protocol hash changed.")
    require(digest(comparison_path) == EXPECTED_COMPARISON_SHA256, "Frozen comparison hash changed.")
    require(digest(comparison_script) == EXPECTED_COMPARISON_SCRIPT_SHA256, "Comparison implementation hash changed.")
    protocol = read(protocol_path)
    comparison = read(comparison_path)
    require(comparison["protocolSha256"] == EXPECTED_PROTOCOL_SHA256, "Comparison references another protocol.")
    require(comparison["comparisonScriptSha256"] == EXPECTED_COMPARISON_SCRIPT_SHA256, "Comparison references another implementation.")
    require(comparison["centerTolerancePx"] == protocol["comparison"]["centerTolerancePx"] == 8, "Frozen center tolerance changed.")
    require(comparison["modelInferenceRun"] is False and comparison["candidateFilterRun"] is False, "Saved report is outside this non-inference audit scope.")
    require(digest(manifest_path) == protocol["manifestSha256"], "Frozen source manifest changed.")
    require(len(comparison["sources"]) == len(protocol["sources"]) == 3, "Expected exactly three frozen sources.")

    protected = {path: digest(path) for path in (protocol_path, comparison_path, comparison_script, manifest_path)}
    original_hashes_verified = 0
    aggregate = Counter()
    source_receipts = []
    total_missing = 0
    total_controls = 0
    for source in protocol["sources"]:
        matches = [item for item in comparison["sources"] if item["id"] == source["id"]]
        require(len(matches) == 1, "Source report identity is missing or duplicated.")
        entry = matches[0]
        labels_path = BASE / source["id"] / "labels.json"
        protected[labels_path] = digest(labels_path)
        require(protected[labels_path] == entry["labelsSha256"], "Frozen label hash changed.")
        labels = read(labels_path)
        require(labels["sourceId"] == source["id"] and labels["inputSha256"] == source["inputSha256"] and labels["protocolSha256"] == EXPECTED_PROTOCOL_SHA256 and labels["predictionsSeen"] is False, "Label provenance changed.")
        require(sorted(item["frame_index"] for item in labels["frames"]) == sorted(source["centers"]), "Center label identity or completeness changed.")
        require(len(labels["frames"]) == len(entry["outcomes"]) == 12, "Expected twelve frozen centers per source.")
        input_path = ROOT / source["inputPath"]
        protected[input_path] = digest(input_path)
        require(protected[input_path] == source["inputSha256"] == entry["inputSha256"], "Frozen source input changed.")
        images = {item["frame_index"]: item for item in source["frames"]}
        for original in source["frames"]:
            require(digest(ROOT / original["path"]) == original["sha256"], "Frozen original PNG changed.")
            original_hashes_verified += 1

        directory = ROOT / "outputs/product/analyses" / entry["analysisId"]
        for name in ("result.json", "detections.json", "trajectories.json"):
            path = directory / name
            protected[path] = digest(path)
            require(protected[path] == entry["protectedEvidenceHashes"][name], "Frozen product evidence changed.")
        raw = read(directory / "detections.json")["frames"]
        saved = read(directory / "result.json")
        published = saved["detections"]["frames"]
        trajectory = saved["trajectories"]["ballTrajectory"]
        standalone = read(directory / "trajectories.json")["ball_trajectory"]
        require(saved["summary"]["id"] == entry["analysisId"], "Saved analysis identity changed.")
        require(len(raw) == len(published) == len(trajectory) == len(standalone) == source["decodedFrames"] == entry["allPublishedControlFramesReproduced"], "Control frame counts disagree.")
        states = Counter()
        trajectory_fields = set()
        missing_indices = []
        for index, (actual, displayed, point, separate) in enumerate(zip(raw, published, trajectory, standalone)):
            expected = json.loads(json.dumps(actual))
            ball = expected["ball"]
            if ball["position_px"] is not None:
                ball["position_px"] = [round(value, 2) for value in ball["position_px"]]
            confidence = ball.get("confidence")
            ball["confidence"] = round(confidence, 3) if confidence is not None and ball["state"] != "MISSING" else None
            require(actual["frame_index"] == displayed["frame_index"] == point["frame_index"] == separate["frame_index"] == index and displayed == expected, "Raw and published frame identity or rounded observation differs.")
            require({key: point[key] for key in separate} == separate, "Standalone trajectory fields differ from the published trajectory.")
            trajectory_fields.update(separate)
            observation = displayed["ball"]
            require(observation["state"] == point["state"], "Published observation and trajectory state differ.")
            require((point["x_px"] is None) == (point["y_px"] is None), "Trajectory contains a partially null coordinate.")
            trajectory_position = None if point["x_px"] is None else [point["x_px"], point["y_px"]]
            require(observation["position_px"] == trajectory_position and observation["confidence"] == point["confidence"], "Published observation and trajectory coordinate/confidence differ.")
            states[observation["state"]] += 1
            if observation["state"] == "MISSING":
                require(all(value is None for value in (actual["ball"]["position_px"], actual["ball"]["confidence"], observation["position_px"], observation["confidence"], point["x_px"], point["y_px"], point["confidence"], separate["x_px"], separate["y_px"], separate["confidence"])), "A MISSING control retains coordinates or confidence.")
                missing_indices.append(index)
            else:
                require(observation["state"] == "DETECTED" and observation["position_px"] is not None, "Unexpected non-missing state or null detection.")

        counts = Counter()
        label_counts = Counter()
        eligible_missing = 0
        eligible_absence = 0
        outcomes = []
        for label, recorded in zip(labels["frames"], entry["outcomes"]):
            index = label["frame_index"]
            require(label == recorded["originalLabel"] and label["sha256"] == images[index]["sha256"], "Recorded outcome does not contain the frozen original label.")
            context = sorted({max(0, min(source["decodedFrames"] - 1, index + delta)) for delta in (-3, 0, 3)})
            require(sorted(label["context_frame_indices"]) == context, "Exact original context indices differ.")
            observation = published[index]["ball"]
            require(observation == recorded["savedObservation"], "Recorded sample observation differs from the frozen published frame.")
            position = observation["position_px"]
            visible = label["visibility"] == "VISIBLE"
            eligible = label["visibility"] == "CONFIRMED_ABSENT" or (visible and label["uncertainty_radius_px"] <= 4)
            distance = math.dist(label["center_px"], position) if visible and position is not None else None
            require(eligible == recorded["eligibleForBoundedComparison"] and distance == recorded["centerSeparationPx"], "Recorded eligibility or center separation differs.")
            if not eligible:
                outcome = "EXCLUDED_UNCERTAIN_OR_IMPRECISE_LABEL"
                counts["excluded"] += 1
            elif label["visibility"] == "CONFIRMED_ABSENT":
                outcome = "FP" if position is not None else "TN"
                counts[outcome] += 1
                counts["confirmedAbsentControls"] += 1
                eligible_absence += 1
            elif distance is not None and distance <= 8:
                outcome = "TP"
                counts["TP"] += 1
                counts["visibleControls"] += 1
            else:
                outcome = "FP_AND_FN" if position is not None else "FN"
                counts["FN"] += 1
                counts["FP"] += position is not None
                counts["visibleControls"] += 1
            counts["reviewedCenters"] += 1
            label_counts[label["visibility"]] += 1
            eligible_missing += eligible and observation["state"] == "MISSING"
            require(outcome == recorded["outcome"], "Recorded bounded outcome differs.")
            outcomes.append({"frame_index": index, "labelVisibility": label["visibility"], "uncertaintyRadiusPx": label["uncertainty_radius_px"], "savedState": observation["state"], "eligible": eligible, "centerSeparationPx": distance, "outcome": outcome})
        require(dict(counts) == entry["counts"], "Recorded per-clip bounded counts differ.")
        aggregate.update(counts)
        total_missing += len(missing_indices)
        total_controls += len(raw)
        source_receipts.append({"sourceId": source["id"], "analysisId": entry["analysisId"], "controlFramesVerified": len(raw), "standaloneTrajectoryFieldsMatched": sorted(trajectory_fields), "savedStateCounts": dict(states), "missingNullControlsVerified": len(missing_indices), "missingFrameIndices": missing_indices, "labelCounts": dict(label_counts), "boundedCounts": dict(counts), "eligibleMissingObservationControls": eligible_missing, "confirmedAbsentControls": eligible_absence, "outcomes": outcomes})

    require(dict(aggregate) == comparison["boundedCounts"], "Recorded aggregate bounded counts differ.")
    require(total_controls == 1477 and total_missing == 410, "Unexpected frozen control or missing count.")
    require(all(item["eligibleMissingObservationControls"] == item["confirmedAbsentControls"] == 0 for item in source_receipts), "The frozen pilot now contains eligible missing/absence controls.")
    require(all(digest(path) == expected for path, expected in protected.items()), "A protected frozen file changed during the audit.")
    boundary = next(item for item in source_receipts if item["sourceId"] == "dimitrov_tiafoe_2024")
    boundary = next(item for item in boundary["outcomes"] if item["frame_index"] == 64)
    receipt = {"auditedAt": datetime.now(timezone.utc).isoformat(), "scope": "INDEPENDENT_FROZEN_SAVED_CONTROL_AUDIT; SINGLE_AI_APPROXIMATE_LABELS; NOT_QUALIFICATION_OR_FULL_CLIP_ACCURACY", "pythonVersion": sys.version, "helperSha256": digest(Path(__file__)), "protocolSha256": EXPECTED_PROTOCOL_SHA256, "comparisonSha256": EXPECTED_COMPARISON_SHA256, "comparisonScriptSha256": EXPECTED_COMPARISON_SCRIPT_SHA256, "modelInferenceRun": False, "candidateFilterRun": False, "labelsModified": False, "frozenEvidenceUnchanged": True, "protectedHashes": {relative(path): value for path, value in protected.items()}, "originalPngHashesVerified": original_hashes_verified, "allControlFramesVerified": total_controls, "allMissingNullControlsVerified": total_missing, "boundedCounts": dict(aggregate), "sources": source_receipts, "untestedBranches": {"confirmedAbsentControls": 0, "eligibleMissingObservationControls": 0, "meaning": "No confirmed-negative TN/FP branch or eligible missing-on-visible FN branch is exercised. Sampled MISSING observations remain excluded."}, "interpretationLimits": [{"sourceId": "schiavone_2012", "frame_index": 394, "issue": "HELD_BALL_TARGET_IDENTITY", "meaning": "The eligible label identifies a genuine ball held by the player. Its inclusion makes the denominator broader than moving rally balls; the frozen protocol does not explicitly settle in-play-only versus any visible tennis-ball identity. No label or count was revised."}, {"sourceId": "schiavone_2012", "frame_index": 288, "issue": "POSSIBLE_HELD_SPARE_BALL", "meaning": "The frozen label leaves the object at the attendant's hand UNCERTAIN and excluded because target identity was unresolved."}, {"sourceId": "dimitrov_tiafoe_2024", "frame_index": 64, "issue": "CUTOFF_SENSITIVE_STREAK_LOCALIZATION", "centerSeparationPx": boundary["centerSeparationPx"], "centerTolerancePx": 8, "roughAnnotationUncertaintyRadiusPx": boundary["uncertaintyRadiusPx"], "outcome": boundary["outcome"], "meaning": "The saved center is just beyond the frozen 8-pixel cutoff. The rough 3-pixel annotation estimate makes this localization outcome sensitive to center placement; it is not equivalent evidence to the other large-separation mismatches. The rough radius is not a statistical confidence interval."}], "noFullClipAccuracyExtrapolation": True}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": relative(destination), "allControlFramesVerified": total_controls, "allMissingNullControlsVerified": total_missing, "boundedCounts": dict(aggregate), "frozenEvidenceUnchanged": True, "modelInferenceRun": False}))


if __name__ == "__main__":
    main()
