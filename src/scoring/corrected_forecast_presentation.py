"""Owned web/report forecasts verified against a fresh causal native replay.

The presentation is explicitly unqualified. It preserves unavailable prefixes,
source clocks and actual results; neither serialization nor replay establishes
independent calibration or prediction accuracy.
"""
from contextlib import closing
from itertools import zip_longest

from src.api.result_storage import ResultStorage
from src.scoring.corrected_exports import _admission
from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
from src.scoring.corrected_rule_rebuild import _canonical, _digest, _scratch


def _require(condition, message):
    if not condition:
        raise ValueError('Corrected forecast presentation: ' + message)


def build_corrected_forecast_presentation(rebuilt, shot_analysis, forecast, storage):
    """Copy validated native rows into this open result owner's disk series.

    All correction/forecast contexts and the supplied writable ResultStorage
    must stay open. Reconstructing the expected causal native history compares
    complete phase, exact source clock, event UID/frontier, raw state, posterior
    and probability payloads, instead of trusting a self-consistent JSON header
    or duplicating the predictor's causal protocol in presentation code.
    """
    _, _, binding = _admission(rebuilt, shot_analysis, forecast, forecast.scratch)
    _require(isinstance(storage, ResultStorage), 'a native owned ResultStorage is required')
    storage._require_open(write=True)
    _scratch(storage.directory, rebuilt.snapshot.files_directory.parent.parent.resolve())
    audit, current = forecast.audit, forecast.current
    _require(_digest(audit['model_spec']) == audit['model_signature'], 'model specification signature differs')
    history = storage.series()
    sentinel = object()
    count = 0
    with rebuild_corrected_forecasts(rebuilt, shot_analysis, forecast.scratch) as expected:
        _require(_canonical(expected.audit) == _canonical(audit), 'fresh native forecast audit differs')
        _require(_canonical(expected.current) == _canonical(current), 'fresh native current forecast differs')
        with closing(iter(forecast.history)) as actual_rows, closing(iter(expected.history)) as expected_rows:
            for actual, native in zip_longest(actual_rows, expected_rows, fillvalue=sentinel):
                _require(actual is not sentinel and native is not sentinel,
                         'native causal history cadence or count differs')
                _require(_canonical(actual) == _canonical(native),
                         'native causal history source, phase, frontier or payload differs')
                # Append detaches one JSON record. Never retain a lazy view
                # backed by ForecastSequence or a different ResultStorage.
                history.append(actual)
                count += 1
        _require(count == audit['history_count'] == 1+2*audit['accepted_input_count'],
                 'native INITIAL/BEFORE/AFTER history count differs')
        _require(_canonical(history[-1]['forecast']) == _canonical(current),
                 'current forecast differs from copied owned history')
    _admission(rebuilt, shot_analysis, forecast, forecast.scratch)
    return {'schemaVersion': '1.0', 'status': 'UNQUALIFIED',
        'analysisId': binding['analysis_id'], 'analysisRevision': binding['analysis_revision'],
        'sourceSha256': binding['source_sha256'], 'baseGeneration': binding['base_generation'],
        'modelSignature': audit['model_signature'], 'calibrationStatus': 'not_calibrated',
        'independentAccuracyValidated': False, 'humanReviewDependent': True,
        'modelSpec': audit['model_spec'], 'historyPolicy': audit['history_policy'],
        'current': current, 'history': history}
