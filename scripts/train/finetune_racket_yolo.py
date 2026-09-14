"""Bounded racket-only fine-tuning pilot with disjoint internal selection clips."""

import argparse
import contextlib

import hashlib

import json

from pathlib import Path

import time



from ultralytics import YOLO



ROOT = Path(__file__).resolve().parents[2]





def digest(path):

    return hashlib.sha256(path.read_bytes()).hexdigest()





def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/external/racketvision_racket_crop_pilot01')
    parser.add_argument('--name', default='racket_yolo11m_pilot02')
    args = parser.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in args.name):
        raise ValueError('Pilot name must contain lowercase letters, digits and underscores only')
    dataset = args.dataset.resolve()

    project = ROOT / 'artifacts/training/vision_upgrade'

    name = args.name

    output = project / name

    protocol_path = project / f'{name}_protocol.json'

    if output.exists() or protocol_path.exists():

        raise ValueError('Require a fresh pilot name; do not overwrite training evidence')

    manifest = json.loads((dataset / 'manifest.json').read_text())

    if not manifest['complete'] or set(manifest['split_clips']['train']) & set(manifest['split_clips']['val']):

        raise ValueError('Incomplete dataset or overlapping internal split')

    for record in manifest['records']:

        if digest(dataset / record['image']) != record['image_sha256'] or digest(dataset / record['label']) != record['label_sha256']:

            raise ValueError('Prepared image or label changed')

    checkpoint = ROOT / 'yolo11m.pt'

    configuration = dict(epochs=5, imgsz=640, batch=2, optimizer='AdamW', lr0=.0001,

                         lrf=.1, warmup_epochs=1., warmup_bias_lr=.0001, freeze=10, patience=5,

                         mosaic=0., mixup=0., close_mosaic=0, degrees=10., translate=.1,

                         scale=.3, shear=0., perspective=.0002, hsv_h=.015, hsv_s=.5,

                         hsv_v=.4, fliplr=.5, flipud=0., workers=0, device=0,

                         amp=False, cache=False, seed=17, deterministic=True,

                         plots=False, classes=[38], save=True, save_period=1)

    protocol = {'complete': False, 'qualification_evidence': False,

                'scope': 'RACKET_ONLY_CANDIDATE; PERSON_AND_OTHER_CLASSES_UNLABELED; NOT_A_REPLACEMENT_PERSON_MODEL',

                'selection': manifest.get('selection_scope', 'ULTRALYTICS_BEST_FITNESS_ON_EIGHT_INTERNAL_SOURCE_IDS; EXTERNAL_REPORTS_NOT_USED'),

                'dataset_manifest_sha256': digest(dataset / 'manifest.json'),

                'initial_checkpoint_sha256': digest(checkpoint), 'script_sha256': digest(Path(__file__)),

                'configuration': configuration, 'dataset_counts': manifest['counts']}

    project.mkdir(parents=True, exist_ok=True)

    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')

    started = time.perf_counter()

    with (project / f'{name}.log').open('w', encoding='utf-8') as log:

        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):

            model = YOLO(str(checkpoint))

            baseline = model.val(data=str(dataset / 'data.yaml'), imgsz=640, batch=2,

                                 conf=.001, iou=.7, classes=[38], device=0, workers=0,

                                 half=False, plots=False, project=str(project), name=f'{name}_baseline')

            protocol['baseline_internal_metrics'] = {k: float(v) for k, v in baseline.results_dict.items()}

            protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')



            def progress(trainer):

                state = {'epoch_finished': trainer.epoch + 1,

                         'elapsed_seconds': time.perf_counter() - started,

                         'metrics': {k: float(v) for k, v in trainer.metrics.items()}}

                (project / f'{name}_progress.json').write_text(json.dumps(state, indent=2), encoding='utf-8')

                log.flush()



            # Validation can fuse its model in-place. Reload the original unfused

            # checkpoint and verify every persistent tensor after trainer setup.

            model = YOLO(str(checkpoint))

            if model.model.is_fused():

                raise ValueError('Training requires an unfused checkpoint')

            expected_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}



            def verify_initialization(trainer):

                actual = trainer.model.state_dict()

                if set(actual) != set(expected_state):

                    raise ValueError('Training architecture state keys changed')

                import torch

                if any(not torch.equal(v.detach().cpu(), expected_state[k]) for k, v in actual.items()):

                    raise ValueError('Training did not load the original persistent tensors exactly')

                protocol['initialization_verified_tensors'] = len(actual)

                protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')



            model.add_callback('on_pretrain_routine_end', verify_initialization)

            model.add_callback('on_fit_epoch_end', progress)

            result = model.train(data=str(dataset / 'data.yaml'), project=str(project),

                                 name=name, exist_ok=False, **configuration)

    if digest(checkpoint) != protocol['initial_checkpoint_sha256']:

        raise ValueError('Original detector checkpoint was modified')

    best = output / 'weights/best.pt'

    if not best.is_file():

        raise ValueError('Training did not produce a selected checkpoint')

    protocol.update(complete=True, elapsed_seconds=time.perf_counter()-started,

                    best_checkpoint=str(best.relative_to(ROOT)), best_checkpoint_sha256=digest(best),

                    final_internal_metrics={k: float(v) for k, v in result.results_dict.items()})

    (output / 'report.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')

    print(json.dumps({'complete': True, 'best_checkpoint': str(best), 'metrics': protocol['final_internal_metrics']}), flush=True)





if __name__ == '__main__':

    main()

