"""Opt-in offline temporal spacing with one output per native video frame."""
import math
from collections import deque
from src.detection.tiled_wasb_candidates import predict_tiled_stream


def predict_spaced_tiled_stream(detector, frames, frame_size, stride, fraction=.6):
    """Infer each index-modulo-stride phase, then emit in native order.

    No frame is duplicated, averaged or dropped. Short phases inherit the
    existing detector's documented padding, unlike the real-triplet pilot.
    """
    if type(stride) is not int or stride < 1:
        raise ValueError('Stride must be a positive integer')
    source=iter(frames)
    queues=[deque() for _ in range(stride)]
    source_index=0
    exhausted=False

    def phase(which):
        nonlocal source_index, exhausted
        while True:
            while not queues[which] and not exhausted:
                try:
                    frame=next(source)
                except StopIteration:
                    exhausted=True
                    break
                queues[source_index % stride].append(frame)
                source_index+=1
            if not queues[which]:return
            yield queues[which].popleft()

    streams=[iter(predict_tiled_stream(detector,phase(i),frame_size,fraction=fraction)) for i in range(stride)]
    finished=set(); index=0
    while len(finished)<stride:
        which=index % stride
        if which not in finished:
            try:
                yield next(streams[which])
            except StopIteration:
                finished.add(which)
        index+=1
    if any(queues):raise RuntimeError('Unconsumed native frames')


def temporal_stride(fps):
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Require a positive finite native frame rate')
    return max(1, int(math.floor(fps / 30 + .5)))


def validate_spaced_ball_config(config):
    ball = config.get('ball_detection', {})
    spacing = ball.get('temporal_spacing', {}).get('enabled', False)
    detours = ball.get('temporal_detours', {}).get('enabled', False)
    if type(spacing) is not bool or type(detours) is not bool:
        raise ValueError('Temporal spacing and detour enabled flags must be booleans')
    if not spacing and not detours:
        return
    if (ball.get('backend') != 'wasb' or ball.get('temporal_step', 3) != 1
            or not ball.get('spatial_crops', {}).get('enabled', False)
            or config.get('temporal_tracking', {}).get('strategy') != 'model_top1'
            or config.get('event_detection', {}).get('authoritative_enabled', True)):
        raise ValueError('Spacing/detour research options require tiled WASB step1, model_top1 and withheld event authority')
    if detours and not ball.get('patch_persistence', {}).get('enabled', False):
        raise ValueError('Temporal detours require selected patch persistence first')
