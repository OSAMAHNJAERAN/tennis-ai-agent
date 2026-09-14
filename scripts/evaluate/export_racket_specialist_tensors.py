"""Export one immutable author checkpoint using scoped, argument-restricted metadata support.

This is not a general checkpoint loader. Torch's standard weights-only unpickler
remains in use. The real Python getattr is never added to its allowlist.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA = 'e6ad74371259d844b11529a64b09052edaec4277ce9ebeeca64d77b9131a19cd'
HISTORY_SHA = 'a1cca4a5ba106caa37782877ca70f66d3e760be6a155cd6b4ea8411d0ecbfca1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reviewed_history_class():
    path = ROOT/'artifacts/research/mmengine_history_v0_10_7/history_buffer.py'
    if digest(path) != HISTORY_SHA:
        raise ValueError('Reviewed HistoryBuffer source changed')
    name = '_tennis_reviewed_mmengine_history_0_10_7'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name].HistoryBuffer


def history_method_resolver(history_class):
    # Capture exact function identities, not arbitrary attribute names.
    methods = {'min': history_class.min, 'max': history_class.max,
               'current': history_class.current, 'mean': history_class.mean}

    def resolve(owner, name):
        if owner is not history_class or type(name) is not str or name not in methods:
            raise ValueError('Only the four reviewed HistoryBuffer class functions may be resolved')
        return methods[name]

    return resolve


def main():
    raise RuntimeError('Real checkpoint loading is blocked by automatic approval review pending explicit user approval; do not remove this gate without that approval.')
    import numpy as np
    import torch
    folder = ROOT/'artifacts/models/racket/rtmdet_research'
    checkpoint = folder/'epoch_300.pth'
    output = folder/'racket_state_dict.pt'
    if output.exists():
        raise FileExistsError('Tensor export already exists; inspect its report rather than overwrite')
    if checkpoint.stat().st_size != 411293859 or digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Only the immutable reviewed author checkpoint is accepted')
    expected = {'builtins.getattr', 'mmengine.logging.history_buffer.HistoryBuffer',
        'numpy.core.multiarray._reconstruct', 'numpy.core.multiarray.scalar', 'numpy.dtype', 'numpy.ndarray'}
    if set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)) != expected:
        raise ValueError('Checkpoint globals differ from static review')
    history = load_reviewed_history_class()
    if history._statistics_methods:
        raise ValueError('Private HistoryBuffer registry must start empty')
    resolver = history_method_resolver(history)
    allowed = [(history, 'mmengine.logging.history_buffer.HistoryBuffer'), (resolver, 'builtins.getattr'),
        (np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
        (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.ndarray, np.dtype,
        np.dtypes.Float64DType, np.dtypes.Int64DType]
    previous_globals = torch.serialization.get_safe_globals()
    with torch.serialization.safe_globals(allowed):
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if torch.serialization.get_safe_globals() != previous_globals:
        raise RuntimeError('Scoped safe globals were not restored')
    if not isinstance(loaded, dict) or not isinstance(loaded.get('state_dict'), dict):
        raise ValueError('Expected author state_dict is absent')
    state = loaded['state_dict']
    if not state or any(type(k) is not str or type(v) is not torch.Tensor for k,v in state.items()):
        raise ValueError('Model state must contain only string keys and plain tensors')
    tensors = {}
    for key, tensor in state.items():
        if tensor.layout != torch.strided or tensor.dtype not in (torch.float32, torch.float16, torch.float64, torch.int64, torch.int32):
            raise ValueError(f'Unexpected model tensor type: {key}')
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite model tensor: {key}')
        tensors[key] = tensor.detach().cpu().contiguous().clone()
    # Preserve useful scalar/config metadata as text only, never execute its config.
    metadata = loaded.get('meta', {})
    report = {'complete': False, 'qualification_evidence': False, 'source_checkpoint_sha256': CHECKPOINT_SHA,
        'history_source_sha256': HISTORY_SHA, 'script_sha256': digest(Path(__file__)),
        'weights_only': True, 'unpickler_replaced': False, 'real_getattr_allowlisted': False,
        'resolver_scope': ['HistoryBuffer.min', 'HistoryBuffer.max', 'HistoryBuffer.current', 'HistoryBuffer.mean'],
        'source_top_level_keys': list(loaded), 'state_tensor_count': len(tensors),
        'state_shapes': {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in tensors.items()},
        'source_meta_repr': repr(metadata),
        'private_history_registry': sorted(history._statistics_methods),
        'limitations': 'Tensor extraction only; no specialist inference, architecture parity or accuracy established'}
    for key, function in history._statistics_methods.items():
        if key not in ('min', 'max', 'current', 'mean'):
            raise ValueError('Unexpected restored statistics method')
        if resolver(history,key) is not function:
            raise ValueError('Restored history function differs')
    del loaded, state
    history._statistics_methods.clear()
    torch.save(tensors, output)
    # Verify fresh default restricted loading: the exported model has no metadata globals.
    nondefault = torch.serialization.get_unsafe_globals_in_checkpoint(output)
    if nondefault:
        raise ValueError(f'Export contains nondefault globals: {nondefault}')
    reloaded = torch.load(output, map_location='cpu', weights_only=True)
    if list(reloaded) != list(tensors) or any(not torch.equal(reloaded[k],v) for k,v in tensors.items()):
        raise ValueError('Exported tensor values do not round-trip exactly')
    report.update({'complete':True, 'tensor_export':output.name, 'tensor_export_sha256':digest(output),
                   'tensor_export_bytes':output.stat().st_size, 'default_reload_all_tensors_exact':True})
    (folder/'tensor_export.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('state_shapes','source_meta_repr')},indent=2))


if __name__ == '__main__':
    main()
