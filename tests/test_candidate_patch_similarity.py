import numpy as np
import pytest

from src.tracking.candidate_patch_similarity import normalized_patch_similarity, shifted_patch_similarity


def test_photometric_gain_and_brightness_do_not_hide_repeated_texture():
    patch = np.arange(25, dtype=float).reshape(5, 5) ** 2
    assert normalized_patch_similarity(patch, patch * .8 + 13) == pytest.approx(1.)
    assert normalized_patch_similarity(patch, -patch) == pytest.approx(-1.)


def test_uniform_or_border_patch_is_unknown_not_static_evidence():
    image = np.ones((25, 25), dtype=np.uint8)
    assert normalized_patch_similarity(image, image) is None
    assert shifted_patch_similarity(image, image, (1, 1)) is None


def test_translation_search_recovers_same_texture_without_changing_input():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (25, 25), dtype=np.uint8)
    shifted = np.roll(image, 2, axis=1)
    assert shifted_patch_similarity(image, shifted, (12, 12), shift=2) == pytest.approx(1.)
    assert shifted_patch_similarity(image, shifted, (12, 12), shift=0) < .9


def test_nonfinite_or_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        normalized_patch_similarity(np.zeros((5, 5)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        normalized_patch_similarity(np.full((5, 5), np.nan), np.zeros((5, 5)))


def test_outer_ring_fit_preserves_new_local_ball_change():
    from src.tracking.candidate_patch_similarity import aligned_local_residual
    rng = np.random.default_rng(17)
    previous = rng.uniform(20,120,(25,25))
    current = previous*.8+15
    assert aligned_local_residual(current,previous,(12,12),shift=0) == pytest.approx(0.,abs=1e-10)
    current[12,11:14] += 40
    assert aligned_local_residual(current,previous,(12,12),shift=0) == pytest.approx(40.)


def test_untextured_ring_or_invalid_point_has_no_supported_fit():
    from src.tracking.candidate_patch_similarity import aligned_local_residual
    image = np.zeros((25,25))
    image[12,12] = 100
    assert aligned_local_residual(image,image,(12,12),shift=0) is None
    with pytest.raises(ValueError):
        aligned_local_residual(image,image,(np.nan,12))
