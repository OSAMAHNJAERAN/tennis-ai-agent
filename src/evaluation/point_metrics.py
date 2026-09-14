"""Visibility-aware, single-ball point metrics on explicitly annotated frames."""

import math


def evaluate_points(rows, tolerance_px=4., reference_size=(512., 288.)):
    """Wrong visible localization is both FP and FN; unlabeled frames excluded.

    Each row contains target_xy or None (explicit absent), prediction_xy or None,
    and source width/height. The tolerance is measured at reference_size.
    """
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("Point tolerance must be positive and finite")
    rows = list(rows)
    tp = fp = fn = tn = 0
    absent_false_detections = visible_abstentions = wrong_locations = 0
    errors = []
    for row in rows:
        target, predicted = row["target_xy"], row["prediction_xy"]
        width, height = row["width"], row["height"]
        if not all(math.isfinite(v) and v > 0 for v in (width, height, *reference_size)):
            raise ValueError("Frame dimensions must be positive and finite")
        for point in (target, predicted):
            if point is not None and (len(point) != 2 or not all(math.isfinite(v) for v in point)):
                raise ValueError("Point must have two finite coordinates")
        if target is None:
            tn += predicted is None
            fp += predicted is not None
            absent_false_detections += predicted is not None
        elif predicted is None:
            fn += 1
            visible_abstentions += 1
        else:
            error = math.hypot((target[0] - predicted[0]) * reference_size[0] / width,
                               (target[1] - predicted[1]) * reference_size[1] / height)
            errors.append(error)
            if error <= tolerance_px:
                tp += 1
            else:
                fp += 1
                fn += 1
                wrong_locations += 1
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "annotated_frames": len(rows),
            "visible_frames": tp + fn,
            "absent_frames": tn + absent_false_detections,
            "wrong_visible_localizations": wrong_locations,
            "visible_abstentions": visible_abstentions,
            "absent_false_detections": absent_false_detections,
            "absent_specificity": tn / (tn + absent_false_detections) if tn + absent_false_detections else None,
            "tolerance_px": tolerance_px, "reference_size": list(reference_size),
            "localization_errors_reference_px": errors}
