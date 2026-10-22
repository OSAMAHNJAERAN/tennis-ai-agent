"""Run the native ball-pass source block without loading the vision runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / 'src/pipeline/phase6_pipeline.py'


def run_ball_pass(namespace):
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
               and node.name == '_run_with_frame_exports')
    branch = next(node for node in run.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "self.ball_backend == 'wasb'")
    exec(compile(ast.Module(body=branch.orelse, type_ignores=[]), str(PATH), 'exec'), namespace)


class Source:
    def __init__(self, failure=None):
        self.starts, self.closed, self.failure = [], 0, failure

    def __iter__(self):
        pytest.fail('Native ball pass used the random-seeking sequence interface')

    def iter_decoded(self, start_index=0):
        self.starts.append(start_index)
        try:
            yield 'frame-0'
            if self.failure is not None:
                raise self.failure
            yield 'frame-1'
        finally:
            self.closed += 1


def namespace(source, *, disk=True, model_failure=None):
    calls = []

    def extract(frame, **options):
        calls.append((frame, options))
        if model_failure is not None:
            raise model_failure
        return [] if frame == 'frame-0' else ['injected-candidate']

    return dict(self=SimpleNamespace(ball_detector=SimpleNamespace(extract_candidates=extract)),
                frames=source, disk_players=disk, imgsz=1280, total_frames=2,
                raw_candidates_per_frame=[], raw_journal=None,
                tqdm=lambda rows, **kwargs: rows, calls=calls)


def test_native_pass_consumes_fresh_sequential_frames_and_records_empty_frames():
    source = Source()
    values = namespace(source)
    run_ball_pass(values)
    assert source.starts == [0] and source.closed == 1
    assert values['calls'] == [('frame-0', {'imgsz': 1280}), ('frame-1', {'imgsz': 1280})]
    assert values['raw_candidates_per_frame'] == [[], ['injected-candidate']]


@pytest.mark.parametrize('where', ['decoder', 'model'])
def test_failed_ball_pass_closes_decoder_and_keeps_original_error(where):
    failure = RuntimeError('injected ' + where + ' error')
    source = Source(failure if where == 'decoder' else None)
    values = namespace(source, model_failure=failure if where == 'model' else None)
    with pytest.raises(RuntimeError) as caught:
        run_ball_pass(values)
    assert caught.value is failure and source.closed == 1
    assert len(values['calls']) == 1
    assert values['raw_candidates_per_frame'] == ([[]] if where == 'decoder' else [])


def test_historical_ball_pass_accepts_existing_plain_frame_sequence():
    values = namespace(['frame-0', 'frame-1'], disk=False)
    run_ball_pass(values)
    assert values['raw_candidates_per_frame'] == [[], ['injected-candidate']]


@pytest.mark.parametrize('failure', [RuntimeError('native inference failure'), KeyboardInterrupt('interrupted')])
def test_decoder_close_failure_cannot_replace_original_model_failure(failure):
    class BrokenClose(Source):
        def iter_decoded(self, start_index=0):
            try:
                yield 'frame-0'
            finally:
                self.closed += 1
                raise ValueError('injected close failure')

    source = BrokenClose()
    values = namespace(source, model_failure=failure)
    with pytest.raises(type(failure)) as caught:
        run_ball_pass(values)
    assert caught.value is failure and source.closed == 1
    assert caught.value.__notes__ == ['Ball decoder cleanup also failed: injected close failure']


@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt])
def test_partial_tile_failure_never_commits_a_raw_global_frame(tmp_path, failure_type):
    import numpy as np
    from src.detection.yolo26_adapters import YOLO26BallDetector
    from src.utils.ball_inference_journal import BallInferenceJournal

    failure = failure_type('injected first tile interruption after full-frame prediction')

    class TileModel:
        def __init__(self, interrupt):
            self.interrupt, self.shapes = interrupt, []

        def predict(self, frame, **options):
            self.shapes.append(frame.shape)
            if self.interrupt and len(self.shapes) == 2:
                raise failure
            return [SimpleNamespace(boxes=None)]

    class DecodedFrames:
        def __init__(self):
            self.starts, self.closed = [], 0

        def iter_decoded(self, start_index=0):
            self.starts.append(start_index)
            try:
                for index in range(start_index, 2):
                    yield np.full((100, 200, 3), index, dtype=np.uint8)
            finally:
                self.closed += 1

    identity = {'scope': 'SYNTHETIC_TILE_ATOMICITY_NO_INFERENCE', 'frames': 2}
    frames = DecodedFrames()

    def values(model, journal):
        detector = YOLO26BallDetector('injected.pt', device='cpu', _model=model,
            tiles={'enabled': True, 'size_px': 100, 'overlap': 0.})
        return dict(self=SimpleNamespace(ball_detector=detector), frames=frames,
                    disk_players=True, imgsz=1280, total_frames=2, raw_journal=journal,
                    raw_candidates_per_frame=[], tqdm=lambda rows, **options: rows)

    first_model = TileModel(True)
    with BallInferenceJournal.create(tmp_path, identity, 2) as journal:
        with pytest.raises(failure_type) as caught:
            run_ball_pass(values(first_model, journal))
        assert caught.value is failure
        assert len(journal) == journal.next_frame == 0
    assert first_model.shapes == [(100, 200, 3), (100, 100, 3)]
    second_model = TileModel(False)
    with BallInferenceJournal.open(tmp_path, identity, 2) as journal:
        run_ball_pass(values(second_model, journal))
        assert len(journal) == journal.next_frame == 2
        assert list(journal) == [[], []]
    assert frames.starts == [0, 0] and frames.closed == 2
    assert second_model.shapes == [(100, 200, 3), (100, 100, 3), (100, 100, 3)] * 2
