"""Metadata-only isolated reproduction of current actual event methods.

AST extraction avoids importing NumPy/Pandas/Torch. Only synthetic ordinary
Python scalar features are exercised; this is not end-to-end video evidence.
"""
from __future__ import annotations
import __future__
import ast
from bisect import bisect_left, bisect_right
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from enum import Enum
import gc
import hashlib
import json
import math
import operator
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import tracemalloc
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import weakref

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
EVENT = ROOT / 'src/events/event_detector.py'
BALL = ROOT / 'src/tracking/temporal_ball_tracker.py'
CODEC = ROOT / 'src/utils/temporal_ball_point_sequence.py'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def resident():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged', 'paged',
                                               'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    status = Counters()
    status.cb = ctypes.sizeof(status)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(status), status.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    if status.peak_rss > 100 * 1024 ** 2:
        raise MemoryError('Metadata-only probe exceeded its100MiB peak-RSS ceiling')
    return {'rss_bytes': status.rss, 'peak_rss_bytes': status.peak_rss}

def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)

class NeverNumPyScalar:
    pass

namespace = dict(globals(), np=SimpleNamespace(generic=NeverNumPyScalar, bool_=NeverNumPyScalar))
bindings = {str(path): digest(path) for path in (EVENT, BALL, CODEC)}
extract(BALL, {'BallState'}, namespace)
extract(CODEC, {'_encode_number', '_decode_number'}, namespace)
extract(EVENT, {'EventType', 'PhysicalEventType', 'TennisEvent', 'EventCandidate',
               'EventDetectorSettings', 'EventDetectionAnalysis', '_Feature',
               '_FEATURE_PAIRS', '_FEATURE_NUMBERS', '_FEATURE_REQUIRED_NUMBERS',
               '_FEATURE_FLAGS', '_FEATURE_FIELDS', '_feature_record', '_decode_feature',
               '_FeatureSequence', 'TennisEventDetector'}, namespace)
Feature = namespace['_Feature']
Store = namespace['_FeatureSequence']
Detector = namespace['TennisEventDetector']
original = namespace['EventCandidate']
live = weakref.WeakSet()
counts = {'peak': 0}

class ObservedCandidate(original):
    __hash__ = object.__hash__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        live.add(self)
        counts['peak'] = max(counts['peak'], len(live))

namespace['EventCandidate'] = ObservedCandidate

def populate(stored, count):
    for index in range(count):
        stored.append(Feature(index, index * .25, (50., 100.), (50., 100.),
                              'DETECTED', .9, score=.8, normalized_speed=.1,
                              strengths={'kinematic': .8}))
    stored.flush()

results = []
tracemalloc.start()
for count in (257, 1025, 4097):
    gc.collect()
    counts['peak'] = 0
    with Store(OUT) as stored:
        populate(stored, count)
        gc.collect()
        before, _ = tracemalloc.get_traced_memory()
        candidates = Detector()._candidates_from_features(stored, 25., 1280, 720, 'DISABLED')
        after, _ = tracemalloc.get_traced_memory()
        results.append({'kind': 'ACTUAL_SQLITE_FEATURE_SELECTOR_AND_CANDIDATE_FACTORY',
                        'input_features': count, 'candidate_container_type': type(candidates).__name__,
                        'retained_candidates': len(candidates), 'live_candidates': len(live),
                        'peak_live_candidates': counts['peak'],
                        'retained_tracemalloc_delta_bytes': after - before, **resident()})
        assert len(candidates) == count and len(live) == count
        candidates = None
    gc.collect()
    assert len(live) == 0

for count in (257, 1025):
    with Store(OUT) as stored:
        populate(stored, count)
        detector = Detector()
        detector._frame_size = lambda *args: (1280, 720, 'SYNTHETIC_METADATA_ONLY')
        detector._positions = lambda *args: (None, 'DISABLED')
        detector._features = lambda *args, **kwargs: stored
        detector._refined_frame = lambda candidate, *args: candidate.frame_index
        detector._legacy_feature_fallback = lambda feature, *args: feature

        def synthetic_verify(candidate_id, candidate, feature, *args, **kwargs):
            event = namespace['TennisEvent'](0, namespace['EventType'].BOUNCE, feature.frame,
                feature.timestamp, None, (50., 100.), None, .8, 'DETECTED', {'synthetic_only': True})
            return event, {'candidate_id': candidate_id, 'stage_pass': {}, 'synthetic_only': True}

        detector._verify = synthetic_verify
        timeline = [None] * count
        analysis = detector._analyze(timeline, timeline, timeline, None, 25., (1280, 720),
                                     None, None, feature_output=stored)
        results.append({'kind': 'ACTUAL_ANALYZE_CONTROL_FLOW_WITH_SYNTHETIC_VERIFICATION',
                        'input_features': count, 'retained_candidates': len(analysis.candidates),
                        'retained_events': len(analysis.events), 'retained_traces': len(analysis.verification_traces),
                        'container_types': [type(analysis.candidates).__name__, type(analysis.events).__name__,
                                            type(analysis.verification_traces).__name__], **resident()})
        assert len(analysis.events) == count and len(analysis.verification_traces) == count
        analysis = detector = timeline = None
    gc.collect()

receipt = {'completed_utc': datetime.now(timezone.utc).isoformat(),
           'status': 'REPRODUCED_DURATION_DEPENDENT_EVENT_RESULT_MATERIALIZATION',
           'scope': 'AST_ISOLATED_ACTUAL_METHODS_AND_SQLITE_SYNTHETIC_SCALARS_ONLY',
           'source_bindings': bindings,
           'source_files_changed': [path for path, expected in bindings.items() if digest(Path(path)) != expected],
           'results': results, 'remaining_feature_scratch_files': [str(path) for path in OUT.glob('event_features_*')],
           'heavy_modules_imported': [name for name in ('numpy', 'pandas', 'torch', 'ultralytics', 'cv2', 'PIL', 'psutil') if name in sys.modules],
           'video_or_model_used': False, 'qualification_evidence': False,
           'full_match_capacity_proven': False, **resident()}
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
if '--assert-bounded' in sys.argv:
    assert max(row['peak_live_candidates'] for row in results if 'peak_live_candidates' in row) <= 8, 'Current scratch selector still retains every selected EventCandidate in a list'
