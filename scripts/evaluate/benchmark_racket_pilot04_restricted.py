"""Run fixed external racket checks with restricted loading of verified weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from ultralytics.nn import tasks
from ultralytics.nn.modules import block,conv,head
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import BboxLoss,DFLoss,v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('coco','video'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((ROOT/'artifacts/training/vision_upgrade/racket_yolo11m_pilot04/report.json').read_text())
    checkpoint=ROOT/report['best_checkpoint']
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if not report['complete'] or sha(checkpoint)!=report['best_checkpoint_sha256']:
        raise ValueError('Selected local training checkpoint changed or run incomplete')
    if sha(ROOT/'yolo11m.pt')!=report['initial_checkpoint_sha256']:
        raise ValueError('Original person detector changed')
    allowed=[tasks.DetectionModel,IterableSimpleNamespace,BboxLoss,DFLoss,v8DetectionLoss,
             TaskAlignedAssigner,np.dtype,np._core.multiarray.scalar,np.dtypes.Float64DType,np.dtypes.Float32DType,
             torch.nn.SiLU,torch.nn.BatchNorm2d,torch.nn.ModuleList,torch.nn.Sequential,
             torch.nn.Conv2d,torch.nn.Identity,torch.nn.BCEWithLogitsLoss,torch.nn.MaxPool2d,torch.nn.Upsample,
             block.Attention,block.Bottleneck,block.C2PSA,block.C3k,block.C3k2,block.DFL,block.PSABlock,
             block.SPPF,conv.Concat,conv.Conv,conv.DWConv,head.Detect]
    module='scripts.evaluate.'+('benchmark_coco_tennis_rackets' if args.task=='coco' else 'benchmark_racket_detection')
    sys.argv=[module,'--mode','person_crops640' if args.task=='coco' else 'player_crops',
              '--model',str(ROOT/'yolo11m.pt'),'--racket-model',str(checkpoint),'--output',str(args.output)]
    if args.task=='video':
        sys.argv+=['--imgsz','640']
    with torch.serialization.safe_globals(allowed):
        runpy.run_module(module,run_name='__main__')
    result=json.loads(args.output.read_text())
    result['restricted_loading']={'weights_only_forced':True,'wrapper_sha256':sha(Path(__file__)),
                                 'selected_training_report_sha256':sha(ROOT/'artifacts/training/vision_upgrade/racket_yolo11m_pilot04/report.json')}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
