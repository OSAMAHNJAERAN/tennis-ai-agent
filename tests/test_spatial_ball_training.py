import cv2
import numpy as np
import pytest
import torch

from scripts.train.finetune_wasb_spatial import (
    SelectionViews, crop_target, focal_loss, split_training_clips,
)


def manifest(clips, split='TRAINING_ONLY', revision='pinned'):
    return {'split': split, 'revision': revision, 'selected_clips': clips}


def test_internal_selection_keeps_match_rallies_together():
    clips = [(f'match{i}', '000') for i in range(10)] + [('match4', '001')]
    result = split_training_clips([manifest(clips[:5]), manifest(clips[5:])])
    assert {clip for clip, split in result.items() if split == 'selection'} == {
        'match4_000', 'match4_001', 'match9_000'}
    assert len(result) == 11


@pytest.mark.parametrize('problem', ['validation', 'duplicate', 'revision'])
def test_split_rejects_data_leakage_or_inconsistent_provenance(problem):
    clips = [(f'match{i}', '000') for i in range(10)]
    inputs = [manifest(clips)]
    if problem == 'validation':
        inputs[0]['split'] = 'VALIDATION_ONLY'
    elif problem == 'duplicate':
        inputs.append(manifest([clips[0]]))
    else:
        inputs.append(manifest([('new', '000')], revision='changed'))
    with pytest.raises(ValueError):
        split_training_clips(inputs)


class KnownGeometry:
    @staticmethod
    def get_affine_transform(center, scale, rotation, size):
        # Independent affine fixture: uniform scaling with center preserved.
        factor = size[0] / scale
        return np.array([[factor, 0, size[0] / 2 - center[0] * factor],
                         [0, factor, size[1] / 2 - center[1] * factor]])

    @staticmethod
    def affine_transform(point, matrix):
        return matrix @ np.array([point[0], point[1], 1])


def test_crop_target_is_centered_and_outside_label_is_absent():
    box = (768, 432, 1920, 1080)
    point, _ = crop_target([1344, 756], box, KnownGeometry)
    assert point == [256, 144]
    for source in [None, [767, 756], [1920, 756], [1344, 1080]]:
        assert crop_target(source, box, KnownGeometry)[0] is None


def test_selection_loader_is_deterministic_and_uses_only_labeled_center(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f'{index}.png'
        assert cv2.imwrite(str(path), np.full((288, 512, 3), 20 + 30 * index, np.uint8))
        paths.append(str(path))
    dataset = SelectionViews([{'views': [{'context_paths': paths, 'target': [100, 150]},
                                       {'context_paths': paths, 'target': None}]}])
    frames, target, slot = dataset[0]
    assert slot == 1
    assert frames.shape == (9, 288, 512)
    assert float(frames[3, 0, 0]) == pytest.approx((80 / 255 - .485) / .229, abs=1e-6)
    assert target[150, 100] == 1
    assert torch.equal(frames, dataset[0][0])
    assert dataset[1][1].sum() == 0


def test_focal_loss_penalizes_confident_false_ball_and_has_finite_gradient():
    logits = torch.tensor([[-20., 20.]], requires_grad=True)
    good = torch.tensor([[0., 1.]])
    bad = torch.tensor([[1., 0.]])
    assert focal_loss(logits, good) < 1e-8
    loss = focal_loss(logits, bad)
    assert loss > 19
    loss.backward()
    assert torch.isfinite(logits.grad).all()
