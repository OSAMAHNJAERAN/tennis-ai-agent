"""Verify the author's racket checkpoint and probe default restricted loading only."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
ROOT = Path(__file__).resolve().parents[2]
REVISION = 'a3760773233a0988c9605259743fbdd87c59d3a3'
EXPECTED = 'e6ad74371259d844b11529a64b09052edaec4277ce9ebeeca64d77b9131a19cd'
SIZE = 411293859


def main():
    folder = ROOT/'artifacts/models/racket/rtmdet_research'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'epoch_300.pth'
    url = f'https://huggingface.co/linfeng302/RacketVision-Models/resolve/{REVISION}/checkpoints/epoch_300.pth'
    if not path.exists():
        temporary = folder/'epoch_300.pth.partial'
        if temporary.exists():
            raise FileExistsError('Existing partial download requires inspection')
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as target:
            total = 0
            while block := response.read(4*1024*1024):
                target.write(block)
                total += len(block)
                if total % (64*1024*1024) == 0:
                    print(json.dumps({'download_bytes':total,'expected':SIZE}),flush=True)
        if temporary.stat().st_size != SIZE or hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED:
            raise ValueError('Downloaded checkpoint differs from publisher LFS digest')
        temporary.rename(path)
    if path.stat().st_size != SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Checkpoint changed')
    report = {'complete':False,'url':url,'repository_revision':REVISION,'checkpoint_sha256':EXPECTED,'bytes':SIZE,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'weights_only_forced':True,
        'scope':'Static inspection and default restricted-load probe; no inference or arbitrary class allowlisting'}
    with zipfile.ZipFile(path) as archive:
        report['zip_members'] = len(archive.infolist())
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError('Checkpoint ZIP CRC failure')
        report['zip_crc_verified'] = True
    import torch
    report['declared_nondefault_globals'] = sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    try:
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        report['default_weights_only_load'] = 'SUCCEEDED'
        report['top_level_keys'] = list(loaded) if isinstance(loaded,dict) else []
        state = loaded.get('state_dict') if isinstance(loaded,dict) else None
        if isinstance(state,dict):
            report['state_shapes'] = {str(k):list(v.shape) for k,v in state.items() if isinstance(v,torch.Tensor)}
    except Exception as exc:
        report['default_weights_only_load'] = 'REJECTED'
        report['exception_type'] = type(exc).__name__
        report['exception'] = str(exc)
    report['complete'] = True
    (folder/'inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
