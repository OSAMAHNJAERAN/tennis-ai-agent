"""Read-only independent corrected-shot review and deterministic edge probes."""
from contextlib import ExitStack
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['TENNIS_STORAGE_ROOT'] = str(HERE / 'isolated_storage')
FILES = [
    'src/scoring/corrected_shot_rebuild.py',
    'src/scoring/corrected_shot_sources.py',
    'tests/test_corrected_shot_rebuild.py',
    'tests/test_corrected_shot_sources.py',
    'src/scoring/corrected_rule_rebuild.py',
    'src/scoring/correction_sources.py',
    'src/scoring/rule_replay_source.py',
    'src/scoring/point_timeline.py',
    'src/shot_analysis/shot_classifier.py',
    'src/shot_analysis/shot_linker.py',
    'src/shot_analysis/pose_feature_extractor.py',
    'src/utils/event_analysis_sequence.py',
    'src/utils/shot_evidence_sequence.py',
    'src/utils/video_frame_sequence.py',
    'src/api/analysis_generation.py',
]


def pins():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}


before = pins()
attempts = []
original = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Independent software probe attempted a model import: ' + name)
    return original(name, *args, **kwargs)


builtins.__import__ = guarded
try:
    from src.scoring.corrected_shot_rebuild import _CropFrames, _LazyFrames, _WindowBoxes
    from types import SimpleNamespace

    source = [{0: {'analysis_allowed': True, 'segment_id': 'camera-0',
        'anchor_frame_index': 0, 'measurement_transition_allowed': frame > 0,
        'source_frame_index': frame, 'source_time_base': '1/30', 'source_pts': str(frame)},
        1: {'analysis_id': 'fixture', 'epoch': 0, 'revision': 0, 'observations': [
            {'track_key': 'track-B', 'player_id': 'B', 'state': 'CONFIRMED'}]}}
        for frame in range(12)]
    boxes = [object() for _ in source]
    probes = []
    for name, mutate in [
        ('measurement_transition', lambda frame: frame[0].update(measurement_transition_allowed=False)),
        ('nonmonotone_source_pts', lambda frame: frame[0].update(source_pts='5')),
        ('source_index_gap', lambda frame: frame[0].update(source_frame_index=70)),
        ('source_time_base_change', lambda frame: frame[0].update(source_time_base='1/60')),
        ('identity_revision', lambda frame: frame[1].update(revision=1)),
        ('identity_track', lambda frame: frame[1]['observations'][0].update(track_key='foreign-track')),
        ('identity_state', lambda frame: frame[1]['observations'][0].update(state='UNKNOWN')),
    ]:
        current = copy.deepcopy(source)
        mutate(current[7])
        bounded = _WindowBoxes(boxes, current, 8, 3)
        assert bounded.allowed == {8, 9, 10, 11}, (name, bounded.allowed)
        assert all(bounded[index] is None for index in (0, 5, 6, 7))
        probes.append({'name': name, 'allowed': sorted(bounded.allowed), 'pass': True})

    continuous = _WindowBoxes(boxes, source, 8, 3)
    assert continuous.allowed == set(range(5, 12))
    bounded_frames = _CropFrames(list(range(12)), continuous)
    assert bounded_frames[5] == 5 and bounded_frames[11] == 11
    for index in (0, 4, 12, True, slice(None)):
        try:
            bounded_frames[index]
        except ValueError:
            pass
        else:
            raise AssertionError('Crop frames accepted an out-of-window or unsupported index')
    probes.append({'name': 'exact_global_crop_window_and_index_types', 'pass': True})

    # The lazy frame wrapper must not load a decoder for length-only paths and
    # must refuse all later access once its owning context has closed it.
    with ExitStack() as stack:
        lazy = _LazyFrames(SimpleNamespace(snapshot=SimpleNamespace(frames=12)), stack)
        assert len(lazy) == 12 and lazy.sequence is None
        lazy.open = False
        for operation in (lambda: len(lazy), lambda: lazy[8]):
            try:
                operation()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Closed crop frame owner remained usable')
    probes.append({'name': 'lazy_no_decoder_and_closed_owner', 'pass': True})
finally:
    builtins.__import__ = original

after = pins()
assert before == after
assert not attempts
review = {
    'status': 'PASS_NO_P1_P2_FOUND',
    'reviewer': '/root/classifier_lazy_import',
    'authorship_disclosure': 'Reviewer authored only the small classifier TYPE_CHECKING import seam and its own tests; did not author corrected shot driver, adapter, or their integration tests.',
    'reviewed_file_hashes_before': before,
    'reviewed_file_hashes_after': after,
    'source_pins_stable': before == after,
    'findings': [],
    'deterministic_edge_probes': probes,
    'model_import_attempts': attempts,
    'reviewed_contracts': [
        'Exact EventAnalysisSequence.events alias enters native link_shots with caller-owned ShotEvidenceSequence.',
        'Fresh accepted scoring journal inputs and transitions are checked before native physical event materialization; reviewed signals stay out of physical events.',
        'Only new accepted bounce line calls supply court landing coordinates; airborne contact projections and old linked shots are withheld.',
        'Selected image boxes retain unrounded native coordinates, confidence, and physical source tracking ID.',
        'Native dead set, classifier branch conditions, fresh attach_shots, sealing, finish and native analytics are used.',
        'Pose enabled/disabled branching follows sealed configuration; reachable configured pose branch refuses missing provider.',
        'At most seven global crop indices are available; continuous source clock and camera/identity anchors bound the window.',
        'Lazy video frames have a fixed eight-frame decoder cache; duration arrays and native outputs remain disk-owned.',
        'Entry, post-provider, pre-yield and successful-exit current-generation checks prevent an obsolete base from succeeding.',
        'Source and configuration resolve through the committed generation; root aliases are not authoritative.',
        'ExitStack closes owned frame/event/shot storage on success and exception; incomplete work has no publication call.',
    ],
    'limits': [
        'Read-only review plus direct deterministic window/lifetime probes; no real model inference or vision-accuracy qualification.',
        'Trusted callback does not verify in-memory selected checkpoint identity; code/audit explicitly leave worker model admission pending.',
        'No claim of whole correction publication, forecasts, rendering, or authenticated correction workflow completion.',
    ],
}
(HERE / 'review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
print(json.dumps(review, indent=2))
