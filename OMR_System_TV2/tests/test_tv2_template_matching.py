import tracemalloc

import cv2
import numpy as np
import pytest

from tv2 import match_template_ncc


def test_matches_opencv_and_finds_peak():
    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, (150, 150), dtype=np.uint8)
    template = np.zeros((20, 20), dtype=np.uint8)
    template[4:16, 4:16] = 255
    img[40:60, 70:90] = template

    ours = match_template_ncc(img, template)
    ref = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    assert ours.shape == ref.shape and ours.dtype == np.float32
    assert np.max(np.abs(ours - ref)) < 1e-4
    assert np.unravel_index(np.argmax(ours), ours.shape) == (40, 70)


def test_flat_windows_and_flat_template_give_zero():
    img = np.full((50, 50), 128, np.uint8)
    img[30:, 30:] = 0
    template = np.zeros((8, 8), np.uint8)
    template[2:6, 2:6] = 255
    out = match_template_ncc(img, template)
    assert np.all(out[:10, :10] == 0.0)          # cửa sổ hoàn toàn phẳng
    assert np.all(match_template_ncc(img, np.full((8, 8), 7, np.uint8)) == 0.0)  # template phẳng


def test_float_input_matches_opencv():
    rng = np.random.default_rng(3)
    img = rng.random((80, 90)).astype(np.float32)
    tmpl = img[20:36, 30:46].copy()
    ref = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    assert np.max(np.abs(match_template_ncc(img, tmpl) - ref)) < 1e-3


def test_memory_stays_small_for_marker_search():
    """Vùng góc 250x200 + template 32x32: bản cũ cấp phát ~300 MB, bản mới phải < 20 MB."""
    rng = np.random.default_rng(1)
    roi = rng.integers(0, 256, (250, 200), dtype=np.uint8)
    tmpl = roi[100:132, 50:82].copy()
    tracemalloc.start()
    match_template_ncc(roi, tmpl)
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    assert peak_mb < 20.0


@pytest.mark.parametrize("img, tmpl", [
    (np.zeros((10, 10)), np.zeros((12, 5))),
    (np.zeros((10, 10, 3)), np.zeros((3, 3))),
])
def test_invalid_shapes_raise(img, tmpl):
    with pytest.raises(ValueError):
        match_template_ncc(img, tmpl)
