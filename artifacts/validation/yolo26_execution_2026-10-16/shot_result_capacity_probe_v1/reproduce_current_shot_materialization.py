"""Red-capable growth probe against the exact current shot-linker class AST."""
import ast
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Set, Tuple
import weakref

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.events.event_detector import TennisEvent, EventType
from src.shot_analysis.shot_types import (ShotType, ShotDirection, ShotOutcome,
    ShotClassificationSource, ShotEventEvidence, PlayerHandedness)
from src.shot_analysis.shot_direction import ShotDirectionClassifier
from src.shot_analysis.court_zones import CourtZoneEngine
from src.tracking.temporal_ball_tracker import TemporalBallPoint
from src.utils.bbox_utils import BBox
import numpy as np

BASE = Path(__file__).resolve().parent
SOURCE = ROOT / 'src/shot_analysis/shot_linker.py'
before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
tree = ast.parse(SOURCE.read_text())
linker_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TennisShotLinker')
namespace = dict(globals(), TennisShotClassifier=object)
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
cases = []
for count in (257, 1025):
    live_events, live_shots = weakref.WeakSet(), weakref.WeakSet()
    peaks = {'events': 0, 'shots': 0}
    class ObservedEvent(TennisEvent):
        __hash__ = object.__hash__
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            live_events.add(self)
            peaks['events'] = max(peaks['events'], len(live_events))
    class ObservedShot(ShotEventEvidence):
        __hash__ = object.__hash__
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            live_shots.add(self)
            peaks['shots'] = max(peaks['shots'], len(live_shots))
    class Events(Sequence):
        def __len__(self):
            return count * 2
        def __getitem__(self, index):
            if not 0 <= index < len(self):
                raise IndexError(index)
            return ObservedEvent(index+1, EventType.PLAYER_1_HIT if index % 2 == 0 else EventType.BOUNCE,
                index, index/25., 1 if index % 2 == 0 else None, (5., 6.), (5., 6.), .9, 'DETECTED')
    class Classifier:
        def classify_shot(self, **kwargs):
            return ShotType.UNKNOWN, .5, ShotClassificationSource.ABSTENTION_UNKNOWN, {}, 'Synthetic control boundary'
    namespace['ShotEventEvidence'] = ObservedShot
    exec(compile(ast.Module(body=[linker_class], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    result = namespace['TennisShotLinker'](Classifier()).link_shots(Events(), [], [], [])
    current, maximum = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cases.append({'hits': count, 'events': count*2, 'linked_shots': len(result), 'result_type': type(result).__name__,
        'peak_live_event_objects': peaks['events'], 'peak_live_shot_objects': peaks['shots'],
        'retained_live_shot_objects': len(live_shots), 'traced_peak_extra_bytes': maximum-baseline,
        'traced_retained_extra_bytes': current-baseline})
    result = None
after = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
receipt = dict(status='REPRODUCED_DURATION_SIZED_EVENT_AND_SHOT_MATERIALIZATION', started_utc=started,
    elapsed_seconds=time.perf_counter()-clock, cases=cases, source_sha256_before=before, source_sha256_after=after,
    source_unchanged=before==after, torch_imported='torch' in sys.modules,
    ultralytics_imported='ultralytics' in sys.modules, media_used=False, model_used=False,
    input_events_are_lazy=True, classifier_is_synthetic=True, qualification_evidence=False,
    full_match_capacity_proven=False)
with (BASE / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(receipt))
assert all(case['peak_live_event_objects'] <= 8 and case['peak_live_shot_objects'] <= 8 for case in cases), \
    'Shot linking retains all events and all resulting shots despite a lazy source'
