"""Evaluation adapter for the official WASB tennis HRNet checkpoint.

Uses the separately cloned MIT source and its original affine preprocessing.
Three chronological RGB frames produce three aligned heatmaps. End padding is
discarded. This offline model can use up to two future frames; it is not causal.
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.tracking.temporal_ball_tracker import BallObservation


class _Config(dict):
    def __getattr__(self, name):
        value = self[name]
        return _Config(value) if isinstance(value, dict) else value


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WASBBallDetector:
    def __init__(self, checkpoint, source="artifacts/research/WASB-SBDT", device="cuda", threshold=.5):
        source = Path(source)
        architecture = _load_file("wasb_hrnet", source / "src/models/hrnet.py")
        self.geometry = _load_file("wasb_image_geometry", source / "src/utils/image.py")
        cfg = _Config(yaml.safe_load((source / "src/configs/model/wasb.yaml").read_text()))
        self.model = architecture.HRNet(cfg)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights["model_state_dict"], strict=True)
        self.model.to(device).eval()
        self.device = device
        self.threshold = threshold
        if not 0 < threshold < 1:
            raise ValueError("Heatmap threshold must be in (0, 1)")

    def predict_triplet(self, frames):
        heatmaps, shape = self.predict_heatmaps(frames)
        return self.decode_heatmaps(heatmaps, shape)

    def predict_heatmaps(self, frames):
        if not 1 <= len(frames) <= 3 or any(frame.shape != frames[0].shape for frame in frames):
            raise ValueError("Provide one to three equally sized BGR frames")
        actual_count = len(frames)
        frames = list(frames) + [frames[-1]] * (3 - actual_count)
        height, width = frames[0].shape[:2]
        center = np.array([width / 2, height / 2], dtype=np.float32)
        transform = self.geometry.get_affine_transform(center, max(height, width), 0, (512, 288))
        inputs = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.warpAffine(rgb, transform, (512, 288), flags=cv2.INTER_LINEAR)
            normalized = (rgb.astype(np.float32) / 255. - np.array([.485, .456, .406], dtype=np.float32))
            normalized /= np.array([.229, .224, .225], dtype=np.float32)
            inputs.append(normalized.transpose(2, 0, 1))
        tensor = torch.from_numpy(np.concatenate(inputs)[None]).to(self.device)
        with torch.inference_mode():
            heatmaps = self.model(tensor)[0].sigmoid().cpu().numpy()[0]
        return heatmaps[:actual_count], (height, width)

    def decode_heatmaps(self, heatmaps, shape):
        height, width = shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        inverse = self.geometry.get_affine_transform(center, max(height, width), 0, (512, 288), inv=1)
        results = []
        for heatmap in heatmaps:
            count, labels = cv2.connectedComponents((heatmap > self.threshold).astype(np.uint8))
            candidates = []
            for label in range(1, count):
                ys, xs = np.where(labels == label)
                weights = heatmap[ys, xs]
                score = float(weights.sum())
                xy = np.array([float(xs @ weights / score), float(ys @ weights / score)])
                x, y = self.geometry.affine_transform(xy, inverse)
                if 0 <= x < width and 0 <= y < height:
                    # Rank by integrated heatmap mass as in upstream peak tracker.
                    candidates.append((score, BallObservation(float(x), float(y), float(weights.max()))))
            candidates.sort(key=lambda item: item[0], reverse=True)
            results.append([observation for _, observation in candidates])
        return results

    def predict_stream(self, frames):
        for heatmap, shape in self.predict_heatmap_stream(frames):
            yield self.decode_heatmaps([heatmap], shape)[0]

    def predict_heatmap_stream(self, frames):
        """Step-one overlapping inference with mean heatmaps, bounded memory.

        Every frame is emitted once. First/last frames use only real windows;
        fewer than three source frames use the documented end-padded triplet.
        """
        window, pending, count = [], {}, 0
        for frame in frames:
            window.append(frame)
            if len(window) < 3:
                continue
            heatmaps, shape = self.predict_heatmaps(window)
            for offset, heatmap in enumerate(heatmaps):
                index = count + offset
                if index in pending:
                    total, votes = pending[index]
                    pending[index] = (total + heatmap, votes + 1)
                else:
                    pending[index] = (heatmap.copy(), 1)
            total, votes = pending.pop(count)
            yield total / votes, shape
            count += 1
            window.pop(0)
        if count == 0:
            if window:
                heatmaps, shape = self.predict_heatmaps(window)
                for heatmap in heatmaps:
                    yield heatmap, shape
        else:
            for index in sorted(pending):
                total, votes = pending[index]
                yield total / votes, shape


class WASBEnsembleDetector(WASBBallDetector):
    """Two fixed checkpoints averaged in heatmap space before threshold decoding.

    This costs two forward passes per window. It shares the single-model stream
    alignment and keeps only the current temporal window in memory.
    """

    def __init__(self, checkpoint, second_checkpoint, second_weight=.75, **kwargs):
        if not 0 < second_weight < 1:
            raise ValueError('Ensemble weight must lie in (0,1)')
        super().__init__(checkpoint, **kwargs)
        self.second = WASBBallDetector(second_checkpoint, **kwargs)
        self.second_weight = second_weight

    def predict_heatmaps(self, frames):
        first, shape = super().predict_heatmaps(frames)
        second, second_shape = self.second.predict_heatmaps(frames)
        if shape != second_shape or first.shape != second.shape:
            raise ValueError('Ensemble model outputs are not aligned')
        return (1 - self.second_weight) * first + self.second_weight * second, shape
