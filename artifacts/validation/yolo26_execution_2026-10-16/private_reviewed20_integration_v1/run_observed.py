"""Passive bounded observations around the unchanged private entrypoint."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.evaluate import run_yolo26_candidate as runner

original_instrument = runner.instrument_models
observed = {'schema_version': 1, 'scope': 'PRIVATE_PASSIVE_ENGINEERING_OBSERVATION_NOT_QUALIFICATION',
            'started_utc': datetime.now(timezone.utc).isoformat(), 'model_objects': {}, 'event_analysis_calls': 0}
hooks = []


def instrument(pipeline):
    stats = original_instrument(pipeline)
    owners = {'ball': pipeline.ball_detector, 'players': pipeline.player_detector, 'court': pipeline.court_detector}
    for role, owner in owners.items():
        record = observed['model_objects'][role] = {'python_object_id': id(owner.model.model),
            'model_path': owner.model_path, 'source_input_shapes': {}, 'forward_tensor_shapes': {},
            'predict_calls': 0, 'forward_calls_including_library_warmup': 0}
        original_predict = owner.model.predict

        def predict(*args, _original=original_predict, _record=record, _owner=owner, **kwargs):
            source = args[0] if args else kwargs.get('source')
            shape = str(tuple(source.shape)) if hasattr(source, 'shape') else type(source).__name__
            _record['source_input_shapes'][shape] = _record['source_input_shapes'].get(shape, 0) + 1
            _record['predict_calls'] += 1
            result = _original(*args, **kwargs)
            settings = _owner.model.predictor.args
            _record['effective_predictor_settings'] = {key: getattr(settings, key) for key in
                ('imgsz', 'conf', 'rect', 'end2end', 'nms', 'half', 'device')}
            return result

        owner.model.predict = predict

        def forward(_module, args, _record=record):
            tensor = args[0]
            key = str((tuple(tensor.shape), str(tensor.dtype), str(tensor.device)))
            _record['forward_tensor_shapes'][key] = _record['forward_tensor_shapes'].get(key, 0) + 1
            _record['forward_calls_including_library_warmup'] += 1

        hooks.append(owner.model.model.register_forward_pre_hook(forward))
    original_analyze = pipeline.event_detector.analyze

    def analyze(*args, **kwargs):
        result = original_analyze(*args, **kwargs)
        output = kwargs.get('result_output')
        observed['event_analysis_calls'] += 1
        observed['event_result_storage'] = {'owner_type': type(output).__name__,
            'owner_open_at_analysis_return': output._connection is not None,
            'scratch_path': str(output.path), 'scratch_exists_at_analysis_return': output.path.exists(),
            'candidate_sequence_type': type(result.candidates).__name__, 'events_sequence_type': type(result.events).__name__,
            'trace_sequence_type': type(result.verification_traces).__name__,
            'candidates': len(result.candidates), 'events': len(result.events), 'traces': len(result.verification_traces)}
        return result

    pipeline.event_detector.analyze = analyze
    return stats


runner.instrument_models = instrument
try:
    runner.main()
finally:
    for hook in hooks:
        hook.remove()
    observed['completed_utc'] = datetime.now(timezone.utc).isoformat()
    values = [value['python_object_id'] for value in observed['model_objects'].values()]
    observed['distinct_model_objects'] = len(set(values))
    storage = observed.get('event_result_storage')
    if storage:
        storage['scratch_exists_after_pipeline_exit'] = Path(storage['scratch_path']).exists()
    output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
    if output.is_dir():
        with (output / 'integration_observer.json').open('x', encoding='utf-8') as stream:
            json.dump(observed, stream, indent=2, allow_nan=False)
            stream.write('\n')
