"""Verify and export the pinned official TOTNet tennis model for research."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/research/TOTNet'
FOLDER = ROOT / 'artifacts/models/ball/totnet_research'
SHA = '36caadd2453cf1a37f26afb0024b861fd3285ca6e9a91e9e4a20d35e90a0a24a'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checkpoint = FOLDER / 'tennis_best.pth'
    assert digest(checkpoint) == SHA
    manifest = json.loads((SOURCE / 'source_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        assert digest(SOURCE / 'source' / entry['path']) == entry['sha256']
    sys.path.insert(0, str(SOURCE / 'vendor'))
    import torch
    from collections import OrderedDict
    # Torch 2.13 rejects SETITEMS on EasyDict subclasses. Read config metadata
    # as a built-in data container; never import or execute EasyDict hooks.
    # Model tensors are unchanged and the standard weights-only loader is used.
    assert torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint) == ['easydict.EasyDict']
    with torch.serialization.safe_globals([(OrderedDict, 'easydict.EasyDict')]):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = payload['state_dict']
    assert state and all(type(k) is str and type(v) is torch.Tensor for k, v in state.items())
    assert all(not v.is_floating_point() or torch.isfinite(v).all() for v in state.values())
    config = payload['configs']
    selected = {key: config.get(key) for key in (
        'model_choice', 'dataset_choice', 'num_channels', 'num_frames', 'img_size',
        'bidirect', 'loss_function', 'ball_size', 'weighting_list', 'seed')}
    spec = importlib.util.spec_from_file_location('tennis_totnet_official', SOURCE / 'source/src/model/TOTNet.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TemporalConvNet(input_shape=tuple(config['img_size']),
                                   spatial_channels=config['num_channels'], num_frames=config['num_frames'])
    model.load_state_dict(state, strict=True)
    output = FOLDER / 'tennis_state_dict.pt'
    if output.exists():
        raise FileExistsError(output)
    torch.save(dict(state), output)
    assert not torch.serialization.get_unsafe_globals_in_checkpoint(output)
    restored = torch.load(output, map_location='cpu', weights_only=True)
    assert list(restored) == list(state) and all(torch.equal(restored[k], v) for k, v in state.items())
    report = {'complete': True, 'qualification_evidence': False, 'checkpoint_sha256': SHA,
              'revision': manifest['revision'], 'script_sha256': digest(Path(__file__)),
              'tensor_export_sha256': digest(output), 'state_tensor_count': len(state),
              'strict_architecture_match': True, 'metadata_read_as_builtin_ordered_dict': True, 'default_restricted_roundtrip_exact': True,
              'checkpoint_epoch': payload.get('epoch'), 'config': selected,
              'torch_version': torch.__version__, 'parameters': sum(p.numel() for p in model.parameters()),
              'limitations': 'No accuracy measurement or inference validation yet.'}
    (FOLDER / 'tensor_export.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
