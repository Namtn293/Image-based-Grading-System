import cv2
import numpy as np
import pytest

from tv2 import solve_homography_svd

SRC = np.array([[50.0, 60.0], [450.0, 30.0], [470.0, 580.0], [40.0, 550.0]], dtype=np.float32)
DST = np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 500.0], [0.0, 500.0]], dtype=np.float32)


def test_matches_opencv_get_perspective_transform():
    H = solve_homography_svd(SRC, DST)
    assert H.shape == (3, 3)
    assert H[2, 2] == pytest.approx(1.0)
    assert np.max(np.abs(H - cv2.getPerspectiveTransform(SRC, DST))) < 1e-4


def test_maps_source_corners_onto_destination():
    H = solve_homography_svd(SRC, DST)
    mapped = np.column_stack([SRC, np.ones(4)]) @ H.T
    np.testing.assert_allclose(mapped[:, :2] / mapped[:, 2:], DST, atol=1e-6)


def test_large_pixel_coordinates_stay_accurate():
    src = np.array([[1203.5, 845.2], [3950.1, 790.8], [4020.7, 5480.3], [1150.9, 5390.6]])
    dst = np.array([[0, 0], [799, 0], [799, 999], [0, 999]], dtype=np.float64)
    H = solve_homography_svd(src, dst)
    H_cv = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    assert np.max(np.abs(H - H_cv) / (np.abs(H_cv) + 1e-9)) < 1e-4


@pytest.mark.parametrize("src", [
    np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float64),      # thẳng hàng
    np.array([[0, 0], [0, 0], [10, 10], [0, 10]], dtype=np.float64),   # trùng điểm
])
def test_degenerate_points_raise(src):
    with pytest.raises(ValueError):
        solve_homography_svd(src, DST)
