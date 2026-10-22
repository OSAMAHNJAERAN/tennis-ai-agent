"""Execute the real crop consumer and pipeline seams with injected models.

This is a model-free contract test, not a GPU, checkpoint or accuracy test.
Only the extractor's local ``import torch`` is replaced by an inert no-grad
context; its crop selection, feature calculation and exception handling run.
"""
import ast
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from src.shot_analysis.shot_types import PlayerHandedness
from src.utils.bbox_utils import BBox
from tests.test_yolo26_crop_provider import Model, detector


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / 'src/pipeline/phase6_pipeline.py'
EXTRACTOR = ROOT / 'src/shot_analysis/pose_feature_extractor.py'


def pipeline_parts():
    tree = ast.parse(PIPELINE.read_text(encoding='utf-8'))
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
               and node.name == '_run_with_frame_exports')
    prefix_branch = next(node for node in run.body if isinstance(node, ast.If)
                         and ast.unparse(node.test) == 'prefix is not None')
    transition = next(node for node in prefix_branch.orelse if isinstance(node, ast.If)
                      and 'finish_tracking_for_crops' in ast.unparse(node))
    guard = next(node for node in run.body if isinstance(node, ast.If)
                 and 'player_crop_provider.assert_success()' in ast.unparse(node))
    metrics = next(node for node in run.body if isinstance(node, ast.With)
                   and '"metrics.json"' in ast.unparse(node).replace("'", '"'))
    returned = next(node for node in reversed(run.body) if isinstance(node, ast.Return))
    return run, transition, guard, metrics, returned


def execute(statement, namespace):
    exec(compile(ast.Module(body=[statement], type_ignores=[]), str(PIPELINE), 'exec'), namespace)


def crop_extractor(model):
    tree = ast.parse(EXTRACTOR.read_text(encoding='utf-8'))
    extractor = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == 'PoseFeatureExtractor')
    local_imports = [node for node in ast.walk(extractor) if isinstance(node, ast.Import)]
    assert len(local_imports) == 1 and [alias.name for alias in local_imports[0].names] == ['torch']

    class InertNoGrad(ast.NodeTransformer):
        def visit_Import(self, node):
            assert [alias.name for alias in node.names] == ['torch']
            return ast.copy_location(ast.Pass(), node)

    def forbidden_load(*args, **kwargs):
        pytest.fail('Injected crop consumer attempted another model load')

    namespace = dict(math=math, List=List, Optional=Optional, Tuple=Tuple, Dict=Dict,
                     Any=Any, np=np, BBox=BBox, PlayerHandedness=PlayerHandedness,
                     YOLO=forbidden_load, torch=SimpleNamespace(no_grad=nullcontext))
    code = ast.fix_missing_locations(ast.Module(body=[InertNoGrad().visit(extractor)], type_ignores=[]))
    exec(compile(code, str(EXTRACTOR), 'exec'), namespace)
    return namespace['PoseFeatureExtractor'](model=model)


class Keypoints:
    def __init__(self):
        self.points = np.zeros((17, 3), dtype=float)
        self.points[:, 2] = .9
        self.points[5, :2], self.points[6, :2] = (10, 20), (30, 20)
        self.points[9, :2], self.points[10, :2] = (5, 30), (40, 30)

    def cpu(self):
        return self

    def numpy(self):
        return self.points


class PoseModel(Model):
    def predict(self, image, **kwargs):
        super().predict(image, **kwargs)
        return [SimpleNamespace(keypoints=SimpleNamespace(data=[Keypoints()]))]


def setup_run(tmp_path, *, model=None, disk=True, pose=True):
    model = model if model is not None else PoseModel()
    player = detector(model)
    extractor = crop_extractor(model) if pose else None
    namespace = dict(self=SimpleNamespace(player_detector=player, pose_extractor=extractor,
                         ball_backend='yolo26', tracking_strategy='temporal'),
                     player_crop_provider=None, inference_resume_record=None,
                     disk_players=disk, output_dir=str(tmp_path),
                     json=json, os=os, fps_proc=0., t_total=0., total_frames=3,
                     shot_evidences=[], rallies=[], authoritative_enabled=False,
                     match_analytics={'available': False}, native_match_analytics_summary=lambda value: value)
    execute(pipeline_parts()[1], namespace)
    return model, player, extractor, namespace


def consume(extractor, boxes):
    frames = [np.full((96, 96, 3), index, dtype=np.uint8) for index in range(3)]
    return frames, extractor.extract_hit_window_features(
        frames, boxes, hit_frame=1, player_id=1,
        handedness=PlayerHandedness.RIGHT_HANDED, court_orientation_sign=1., window_half_size=1)


def publish(namespace):
    _, _, guard, metrics, returned = pipeline_parts()
    execute(guard, namespace)
    execute(metrics, namespace)
    return eval(compile(ast.Expression(returned.value), str(PIPELINE), 'eval'), namespace)


def test_real_crop_consumer_uses_clean_shared_model_and_records_actual_calls(tmp_path):
    model, player, extractor, namespace = setup_run(tmp_path)
    provider = namespace['player_crop_provider']
    assert player.model is provider.model is model and extractor.model is provider
    assert model.predictor is None and model.hook.removed == model.callback_resets == 1
    frames, features = consume(extractor, [BBox(20, 20, 65, 85)] * 3)
    assert features == {'valid': True, 'normalized_wrist_displacement': 1.,
                        'mean_pose_confidence': pytest.approx(.9), 'sample_count': 3,
                        'shoulder_angle_mean': 0.}
    assert len(model.calls) == 3
    for (image, arguments), original in zip(model.calls, frames):
        assert np.shares_memory(image, original) and image.shape != original.shape
        # The extractor's historical cuda:0 default cannot override admission.
        assert arguments['device'] == 'cuda:2' and arguments['imgsz'] == [640, 672]
        assert arguments['classes'] is None and arguments['mode'] == 'predict'
    returned = publish(namespace)
    metrics = json.loads((tmp_path / 'metrics.json').read_text())
    assert metrics['pose_crop_execution'] == returned['pose_crop_execution'] == provider.record()
    assert metrics['pose_crop_execution']['prediction_calls'] == 3
    assert metrics['pose_crop_execution']['completed_prediction_calls'] == 3
    assert metrics['production_qualified'] is False


def test_real_extractor_cannot_hide_crop_failure_and_publish_metrics(tmp_path):
    failure = RuntimeError('injected selected model failure')
    model, _, extractor, namespace = setup_run(tmp_path, model=PoseModel(error=failure))
    _, features = consume(extractor, [BBox(20, 20, 65, 85)] * 3)
    assert features['valid'] is False  # The existing consumer catches the error.
    assert len(model.calls) == 1  # Later crops never forward after that failure.
    with pytest.raises(RuntimeError, match='crop inference failed') as caught:
        publish(namespace)
    assert caught.value.__cause__ is failure
    assert not (tmp_path / 'metrics.json').exists()
    assert namespace['player_crop_provider'].record()['failed'] is True


@pytest.mark.parametrize('boxes', [[None] * 3, [BBox(30, 30, 31, 31)] * 3])
def test_absent_or_tiny_crops_do_not_invent_model_invocations(tmp_path, boxes):
    model, _, extractor, namespace = setup_run(tmp_path)
    _, features = consume(extractor, boxes)
    assert features['valid'] is False and not model.calls
    record = publish(namespace)['pose_crop_execution']
    assert record['prediction_calls'] == record['completed_prediction_calls'] == 0
    assert record['failed'] is False


def test_disabled_pose_still_finishes_tracking_without_running_crops(tmp_path):
    model, _, extractor, namespace = setup_run(tmp_path, pose=False)
    assert extractor is None and model.predictor is None and model.callback_resets == 1
    record = publish(namespace)['pose_crop_execution']
    assert record['prediction_calls'] == record['completed_prediction_calls'] == 0
    assert not model.calls


def test_historical_branch_keeps_existing_model_and_omits_crop_record(tmp_path):
    model, _, extractor, namespace = setup_run(tmp_path, disk=False)
    assert extractor.model is model and model.callback_resets == 0
    assert namespace['player_crop_provider'] is None
    assert 'pose_crop_execution' not in publish(namespace)
    assert 'pose_crop_execution' not in json.loads((tmp_path / 'metrics.json').read_text())


def test_transition_and_failure_guard_follow_the_complete_player_pass():
    run, transition, guard, metrics, returned = pipeline_parts()
    prefix_branch = next(node for node in run.body if isinstance(node, ast.If)
                         and ast.unparse(node.test) == 'prefix is not None')
    body = prefix_branch.orelse + run.body
    role_selection = next(node for node in body if 'role_tracker.select(' in ast.unparse(node))
    person_export = next(node for node in body if isinstance(node, ast.For)
                         and 'image_people_records(' in ast.unparse(node))
    ball_pass = next(node for node in body if isinstance(node, ast.Assign)
                     and 'BallCandidateSequence(' in ast.unparse(node))
    shot_link = next(node for node in body if isinstance(node, ast.Assign)
                     and 'self.shot_linker.link_shots(' in ast.unparse(node))
    render = next(node for node in body if isinstance(node, ast.With)
                  and 'cv2.VideoWriter(' in ast.unparse(node))
    assert role_selection.lineno < person_export.lineno < transition.lineno < ball_pass.lineno
    assert ball_pass.lineno < shot_link.lineno < guard.lineno < render.lineno < metrics.lineno < returned.lineno
    assert ast.unparse(transition.test) == 'disk_players'
    assert 'admitted_device=str(self.player_detector.model.device)' in ast.unparse(transition)
