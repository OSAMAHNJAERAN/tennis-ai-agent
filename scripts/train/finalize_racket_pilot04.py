"""Finalize metrics for an intact selected checkpoint after a save-copy failure."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluate import benchmark_racket_pilot04_restricted as safe
from ultralytics import YOLO


def main():
    project=ROOT/'artifacts/training/vision_upgrade'
    output=project/'racket_yolo11m_pilot04'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=json.loads((project/'racket_yolo11m_pilot04_protocol.json').read_text())
    rows=list(csv.DictReader((output/'results.csv').open()))
    if [int(r['epoch']) for r in rows]!=[1,2,3,4,5]:
        raise ValueError('Five completed validation epochs required')
    selected=max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
    if int(selected['epoch'])!=4:
        raise ValueError('Unexpected internally selected epoch')
    best=output/'weights/best.pt';last=output/'weights/last.pt'
    if sha(best)!=sha(output/'weights/epoch3.pt'):
        raise ValueError('Best checkpoint differs from selected epoch')
    for path in (best,last):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError('Checkpoint archive CRC mismatch')
    allowed=[safe.tasks.DetectionModel,safe.IterableSimpleNamespace,safe.BboxLoss,safe.DFLoss,safe.v8DetectionLoss,
             safe.TaskAlignedAssigner,safe.np.dtype,safe.np._core.multiarray.scalar,safe.np.dtypes.Float64DType,safe.np.dtypes.Float32DType,
             safe.torch.nn.SiLU,safe.torch.nn.BatchNorm2d,safe.torch.nn.ModuleList,safe.torch.nn.Sequential,
             safe.torch.nn.Conv2d,safe.torch.nn.Identity,safe.torch.nn.BCEWithLogitsLoss,safe.torch.nn.MaxPool2d,safe.torch.nn.Upsample,
             safe.block.Attention,safe.block.Bottleneck,safe.block.C2PSA,safe.block.C3k,safe.block.C3k2,safe.block.DFL,safe.block.PSABlock,
             safe.block.SPPF,safe.conv.Concat,safe.conv.Conv,safe.conv.DWConv,safe.head.Detect]
    with safe.torch.serialization.safe_globals(allowed):
        for path,epoch in ((best,3),(last,4)):
            saved=safe.torch.load(path,map_location='cpu',weights_only=True)
            if saved['epoch']!=epoch:
                raise ValueError('Checkpoint epoch metadata mismatch')
            del saved
        model=YOLO(str(best))
        metrics=model.val(data=str(ROOT/'data/external/racketvision_racket_rehearsal_pilot04/data.yaml'),
                          imgsz=640,batch=2,conf=.001,iou=.7,classes=[38],device=0,workers=0,
                          half=False,plots=False,project=str(project),name='racket_yolo11m_pilot04_final_selected')
    if sha(ROOT/'yolo11m.pt')!=report['initial_checkpoint_sha256']:
        raise ValueError('Original model changed')
    report.update(complete=True,training_epochs_completed=5,selected_epoch=4,
                  best_checkpoint=str(best.relative_to(ROOT)),best_checkpoint_sha256=sha(best),
                  last_checkpoint_sha256=sha(last),finalization_script_sha256=sha(Path(__file__)),
                  resume_script_sha256=sha(ROOT/'scripts/train/resume_racket_pilot04.py'),
                  interruption=json.loads((output/'interruption.json').read_text()),
                  final_save_failure='Epoch-five redundant epoch4.pt copy failed with ENOSPC; last.pt and best.pt intact, CRC and restricted loading verified; empty copy preserved',
                  final_internal_metrics={k:float(v) for k,v in metrics.results_dict.items()})
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'best_checkpoint_sha256':report['best_checkpoint_sha256'],'metrics':report['final_internal_metrics']}),flush=True)


if __name__=='__main__':
    main()
