import numpy as np
import pytest

from tv2 import destination_corners, order_corners, quad_size


def test_order_corners_scrambled():
    scrambled = np.array([[480, 640], [20, 30], [25, 620], [490, 40]], dtype=np.float32)
    ordered, (w, h) = order_corners(scrambled)
    np.testing.assert_allclose(ordered, [[20, 30], [490, 40], [480, 640], [25, 620]])
    assert (w, h) == quad_size(ordered)
    assert ordered.dtype == np.float32


def test_order_corners_accepts_contour_shape():
    contour = np.array([[[10, 10]], [[110, 12]], [[108, 210]], [[8, 205]]], dtype=np.int32)
    ordered, _ = order_corners(contour)
    np.testing.assert_allclose(ordered[0], [10, 10])


def test_order_corners_45_degree_gives_distinct_points():
    diamond = np.array([[400, 100], [700, 400], [400, 700], [100, 400]], dtype=np.float32)
    ordered, _ = order_corners(diamond)
    assert len({tuple(p) for p in ordered.tolist()}) == 4
    # Thứ tự phải cùng chiều kim đồng hồ (diện tích có hướng > 0 trong hệ trục y hướng xuống)
    x, y = ordered[:, 0], ordered[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert signed_area > 0


@pytest.mark.parametrize("bad", [np.zeros((3, 2)), np.zeros((5, 2)), np.array([[0, 0], [1, np.nan], [2, 2], [3, 3]])])
def test_order_corners_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        order_corners(bad)


def test_destination_corners_uses_pixel_centers():
    np.testing.assert_allclose(destination_corners((800, 1000)), [[0, 0], [799, 0], [799, 999], [0, 999]])
