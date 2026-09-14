import cv2
import numpy as np
import pytest
import random

from scripts.train.finetune_wasb import SparseBallDataset, add_stationary_graphic


def test_synthetic_graphic_is_identical_across_time_and_preserves_target_region():
    images = [np.full((288, 512, 3), value, np.uint8) for value in (40, 60, 80)]
    target = np.zeros((288, 512), np.float32)
    target[10:60, 10:120] = 1
    original = target.copy()
    output = add_stationary_graphic(images, target, random.Random(7))
    changed = np.any(output[0] != images[0], axis=2)
    assert changed.any()
    assert not changed[target > 0].any()
    for image in output[1:]:
        np.testing.assert_array_equal(output[0][changed], image[changed])
    np.testing.assert_array_equal(target, original)
    for image, value in zip(images, (40, 60, 80)):
        assert (image == value).all()  # no mutation of cached source frames


def test_synthetic_graphic_abstains_if_all_locations_cover_targets():
    images = [np.zeros((288, 512, 3), np.uint8) for _ in range(3)]
    target = np.ones((288, 512), np.float32)
    output = add_stationary_graphic(images, target, random.Random(7))
    assert all((image == 0).all() for image in output)


@pytest.mark.parametrize('slot', [0, 1, 2])
@pytest.mark.parametrize('visible', [True, False])
def test_sparse_supervision_keeps_labeled_frame_in_selected_output_slot(tmp_path, monkeypatch, slot, visible):
    paths = []
    for index in range(5):
        path = tmp_path / f'{index}.png'
        image = np.full((288, 512, 3), 20 + 30 * index, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        paths.append(str(path))
    monkeypatch.setattr('scripts.train.finetune_wasb.random.randrange', lambda _: slot)
    monkeypatch.setattr('scripts.train.finetune_wasb.random.random', lambda: 1.)
    monkeypatch.setattr('scripts.train.finetune_wasb.random.uniform', lambda *_: 1.)
    dataset = SparseBallDataset([{'context_paths': paths, 'target': [100, 150] if visible else None}])
    frames, target, output_slot = dataset[0]
    assert frames.shape == (9, 288, 512)
    assert output_slot == slot
    # Original context index 2 is always the sole annotated frame.
    assert float(frames[slot * 3, 0, 0]) == pytest.approx((80 / 255 - .485) / .229, abs=1e-6)
    assert float(target[150, 100]) == (1. if visible else 0.)
    if not visible:
        assert float(target.sum()) == 0.
