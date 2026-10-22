"""GPU ID normalization uses real torch device parsing without GPU work."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import ultralytics

from src.detection import yolo26_adapters as adapters


@pytest.mark.parametrize('role', ['players', 'ball', 'court'])
@pytest.mark.parametrize('device,expected', [('0', 'cuda:0'), (0, 'cuda:0'), ('2', 'cuda:2'),
                                           (np.int64(2), 'cuda:2'), ('cuda:1', 'cuda:1'),
                                           ('cpu', 'cpu'), (torch.device('cpu'), 'cpu'),
                                           ('auto', None)])
def test_each_adapter_normalizes_only_torch_placement_and_retains_inference_device(tmp_path, monkeypatch, role, device, expected):
    weight = tmp_path / 'pinned-shape-fixture.pt'
    weight.write_bytes(b'schema fixture, not an inference checkpoint')
    created = []
    class Model:
        def __init__(self, path, task):
            self.task = task
            self.placements, self.inference = [], []
            schema = {'yaml_file': 'yolo26-pose.yaml' if task == 'pose' else 'yolo26.yaml',
                      'scale': 'm' if role == 'ball' else 's', 'end2end': True}
            if role != 'ball':
                schema['kpt_shape'] = [17 if role == 'players' else 14, 3]
            self.model = SimpleNamespace(yaml=schema)
            created.append(self)
        def to(self, value):
            # This actual PyTorch parser rejects the original string "0" bug,
            # while parsing cuda:index never allocates or initializes a GPU.
            self.placements.append(torch.device(value))
            return self
        def predict(self, frame, **kwargs):
            self.inference.append(kwargs)
            return [SimpleNamespace(boxes=None, keypoints=None)]
        def track(self, frame, **kwargs):
            return self.predict(frame, **kwargs)
    monkeypatch.setattr(ultralytics, 'YOLO', Model)
    if role == 'players':
        detector = adapters.YOLO26PlayerPoseDetector(weight, tracker='pinned.yaml', device=device)
        detector.detect_and_track(np.zeros((4, 4, 3), np.uint8))
    elif role == 'ball':
        detector = adapters.YOLO26BallDetector(weight, device=device)
        detector.extract_candidates(np.zeros((4, 4, 3), np.uint8))
    else:
        detector = adapters.YOLO26CourtPoseDetector(weight, device=device)
        detector.predict(np.zeros((4, 4, 3), np.uint8))
    assert len(created) == 1
    assert [str(value) for value in created[0].placements] == ([] if expected is None else [expected])
    arguments = created[0].inference[0]
    if device == 'auto':
        assert 'device' not in arguments
    else:
        assert arguments['device'] == device
    assert arguments['end2end'] is True
    assert arguments['nms'] is False


@pytest.mark.parametrize('device', [True, np.bool_(True), -1, np.int64(-2)])
def test_invalid_integer_gpu_ids_cannot_become_boolean_dtype_or_negative_device(device):
    with pytest.raises(ValueError, match='nonnegative integer'):
        adapters._torch_device(device)
