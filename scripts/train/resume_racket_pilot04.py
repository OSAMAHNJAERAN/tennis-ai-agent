"""Resume a verified locally produced pilot checkpoint with restricted loading."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

# All library checkpoint reads in this process use PyTorch's restricted loader.
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn import tasks
from ultralytics.nn.modules import block, conv, head
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import BboxLoss, DFLoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner

ROOT=Path(__file__).resolve().parents[2]
EXPECTED='348548f35ce763985c480adb98251be451bbf9b74e7e1f70a62d014507e9eda0'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project=ROOT/'artifacts/training/vision_upgrade'
    output=project/'racket_yolo11m_pilot04'
    checkpoint=output/'weights/last.pt'
    if digest(checkpoint)!=EXPECTED:
        raise ValueError('Resume checkpoint differs from inspected local training output')
    report=json.loads((project/'racket_yolo11m_pilot04_protocol.json').read_text())
    if digest(ROOT/'scripts/train/finetune_racket_yolo.py')!=report['script_sha256']:
        raise ValueError('Original training source changed')
    dataset=ROOT/'data/external/racketvision_racket_rehearsal_pilot04'
    if digest(dataset/'manifest.json')!=report['dataset_manifest_sha256']:
        raise ValueError('Training data manifest changed')
    for row in json.loads((dataset/'manifest.json').read_text())['records']:
        for key in ('image','label'):
            if digest(dataset/row[key])!=row[key+'_sha256']:
                raise ValueError('Training data file changed')
    # Explicit known classes from the inspected local pickle; no dynamic import
    # or automatic acceptance of classes supplied by checkpoint contents.
    allowed=[tasks.DetectionModel, IterableSimpleNamespace, BboxLoss, DFLoss,
             v8DetectionLoss, TaskAlignedAssigner, np.dtype, np._core.multiarray.scalar,
             np.dtypes.Float64DType, np.dtypes.Float32DType,
             torch.nn.SiLU, torch.nn.BatchNorm2d, torch.nn.ModuleList,
             torch.nn.Sequential, torch.nn.Conv2d, torch.nn.Identity,
             torch.nn.BCEWithLogitsLoss, torch.nn.MaxPool2d, torch.nn.Upsample,
             block.Attention, block.Bottleneck, block.C2PSA, block.C3k,
             block.C3k2, block.DFL, block.PSABlock, block.SPPF,
             conv.Concat, conv.Conv, conv.DWConv, head.Detect]
    with torch.serialization.safe_globals(allowed):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['epoch']!=1 or saved.get('optimizer') is None or saved.get('ema') is None:
            raise ValueError('Checkpoint is not the verified two-epoch resumable state')
        del saved
        backup=output/'resume_source_epoch02.pt'
        if not backup.exists():
            shutil.copyfile(checkpoint,backup)
        if digest(backup)!=EXPECTED:
            raise ValueError('Preserved resume source changed')
        failure={'status':'HOST_MEMORY_ALLOCATION_FAILURE_AFTER_TWO_COMPLETED_EPOCHS',
                 'error':'OpenCV could not allocate 1228800 bytes during augmentation',
                 'concurrent_model_tests_finished':True,'resume_source_sha256':EXPECTED,
                 'resume_scope':'Optimizer/EMA/epoch restored; exact uninterrupted RNG sequence not asserted'}
        (output/'interruption.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        started=time.perf_counter()
        model=YOLO(str(checkpoint))
        def progress(trainer):
            state={'epoch_finished':trainer.epoch+1,'resume_elapsed_seconds':time.perf_counter()-started,
                   'metrics':{k:float(v) for k,v in trainer.metrics.items()}}
            (project/'racket_yolo11m_pilot04_progress.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        model.add_callback('on_fit_epoch_end',progress)
        result=model.train(resume=True)
    best=output/'weights/best.pt'
    if digest(ROOT/'yolo11m.pt')!=report['initial_checkpoint_sha256']:
        raise ValueError('Original checkpoint changed')
    report.update(complete=True,best_checkpoint=str(best.relative_to(ROOT)),best_checkpoint_sha256=digest(best),
                  resume_script_sha256=digest(Path(__file__)),resume_source_sha256=EXPECTED,
                  resume_elapsed_seconds=time.perf_counter()-started,
                  interruption=failure,final_internal_metrics={k:float(v) for k,v in result.results_dict.items()})
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'checkpoint_sha256':report['best_checkpoint_sha256'],
                      'metrics':report['final_internal_metrics']}),flush=True)


if __name__=='__main__':
    main()
