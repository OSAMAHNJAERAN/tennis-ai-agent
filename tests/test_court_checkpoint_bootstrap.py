"""Saved court checkpoints must initialize offline and reject incomplete states."""

import pytest
import torch
from torch import nn

from src.court import court_keypoint_detector as module


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.fc = nn.Linear(2, 1000)


def checkpoint_model():
    model = TinyResNet()
    model.fc = nn.Linear(2, 28)
    return model


def test_saved_full_checkpoint_builds_without_pretrained_weights(tmp_path, monkeypatch):
    expected = checkpoint_model().state_dict()
    path = tmp_path / 'court.pth'
    torch.save(expected, path)
    calls = []

    def offline_resnet(*, weights):
        calls.append(weights)
        if weights is not None:
            raise AssertionError('Saved checkpoint initialization attempted a pretrained download')
        return TinyResNet()

    monkeypatch.setattr(module, 'resnet50', offline_resnet)
    detector = module.CourtKeypointDetector(str(path), device='cpu')
    assert calls == [None]
    assert not detector.model.training
    assert all(torch.equal(detector.model.state_dict()[key], tensor) for key, tensor in expected.items())


@pytest.mark.parametrize('failure', ['missing_parameter', 'wrong_shape'])
def test_saved_checkpoint_rejects_incomplete_or_incompatible_state(tmp_path, monkeypatch, failure):
    state = checkpoint_model().state_dict()
    if failure == 'missing_parameter':
        del state['backbone.weight']
    else:
        state['fc.weight'] = torch.zeros(27, 2)
    path = tmp_path / 'invalid.pth'
    torch.save(state, path)
    monkeypatch.setattr(module, 'resnet50', lambda *, weights: TinyResNet())
    with pytest.raises(RuntimeError, match='Missing key|size mismatch'):
        module.CourtKeypointDetector(str(path), device='cpu')


def test_no_checkpoint_preserves_pretrained_training_initialization(monkeypatch):
    calls = []

    def resnet(*, weights):
        calls.append(weights)
        return TinyResNet()

    monkeypatch.setattr(module, 'resnet50', resnet)
    detector = module.CourtKeypointDetector(device='cpu')
    assert calls == [module.ResNet50_Weights.DEFAULT]
    assert detector.model.fc.out_features == 28
