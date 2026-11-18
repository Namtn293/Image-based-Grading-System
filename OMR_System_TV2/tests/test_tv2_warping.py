import os

import cv2
import numpy as np
import pytest

from conftest import DATA_INPUT_DIR
from tv2 import order_corners, solve_homography_svd, warp_perspective_bilinear, warp_perspective_sheet


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def test_bilinear_matches_opencv_gray(checkerboard_image):
    src = np.array([[60, 50], [420, 80], [450, 520], [40, 500]], dtype=np.float32)
    out_size = (400, 500)
    H = solve_homography_svd(src, np.array([[0, 0], [399, 0], [399, 499], [0, 499]], np.float32))
    ours = warp_perspective_bilinear(checkerboard_image, H, out_size)
    ref = cv2.warpPerspective(checkerboard_image, H, out_size, flags=cv2.INTER_LINEAR)
    assert ours.shape == (500, 400) and ours.dtype == np.uint8
    assert _psnr(ours, ref) > 48.0


def test_bilinear_color_image(checkerboard_image):
    color = np.dstack([checkerboard_image, 255 - checkerboard_image, checkerboard_image // 2])
    H = np.array([[0.9, 0.05, 10], [0.02, 0.95, 5], [1e-5, 2e-5, 1]])
    ours = warp_perspective_bilinear(color, H, (300, 300))
    ref = cv2.warpPerspective(color, H, (300, 300), flags=cv2.INTER_LINEAR)
    assert ours.shape == (300, 300, 3)
    assert _psnr(ours, ref) > 48.0


def test_singular_homography_raises(checkerboard_image):
    with pytest.raises(ValueError):
        warp_perspective_bilinear(checkerboard_image, np.zeros((3, 3)), (10, 10))


def test_sheet_warp_matches_opencv_on_sample_data():
    img = cv2.imread(os.path.join(DATA_INPUT_DIR, "sample_tilted_sheet.png"), cv2.IMREAD_GRAYSCALE)
    corners = np.loadtxt(os.path.join(DATA_INPUT_DIR, "input_corners.txt"))
    ours = warp_perspective_sheet(img, corners, 800, 1000, use_scratch=True)
    ref = warp_perspective_sheet(img, corners, 800, 1000, use_scratch=False)
    assert ours.shape == ref.shape == (1000, 800)
    assert _psnr(ours, ref) > 48.0


def test_sheet_corners_land_on_output_borders():
    """Góc TL/BR của phiếu phải rơi đúng pixel (0,0) và (W-1,H-1) (không lệch 1 pixel)."""
    img = np.zeros((200, 200), np.uint8)
    corners = np.array([[20, 30], [170, 25], [180, 190], [15, 180]], np.float32)
    ordered, _ = order_corners(corners)
    for (x, y) in ordered.astype(int):
        img[y, x] = 255
    out = warp_perspective_sheet(img, corners, 100, 120)
    assert out[0, 0] > 0 and out[0, 99] > 0 and out[119, 99] > 0 and out[119, 0] > 0
