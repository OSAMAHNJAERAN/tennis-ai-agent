"""Research-only interleaved temporal streams preserving every native frame."""
from collections import deque
from src.detection.tiled_wasb_candidates import predict_tiled_stream


def predict_spaced_tiled_stream(detector, frames, frame_size, stride):
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

    streams=[iter(predict_tiled_stream(detector,phase(i),frame_size)) for i in range(stride)]
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
