"""Các hàm OpenCV đối chứng (chỉ dùng để so sánh độ chính xác & tốc độ).

Module này KHÔNG được import bởi luồng thuật toán chính của TV2; nó chỉ được nạp
khi gọi warp_perspective_sheet(use_scratch=False) hoặc khi chạy benchmark.
"""

from typing import Tuple

import cv2
import numpy as np

from .geometry import destination_corners, order_corners


def get_perspective_transform_cv(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(np.asarray(src, np.float32), np.asarray(dst, np.float32))


def warp_perspective_cv(img: np.ndarray, H: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(img, H, (int(out_size[0]), int(out_size[1])), flags=cv2.INTER_LINEAR)


def match_template_cv(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    return cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)


def warp_perspective_sheet_cv(
    image: np.ndarray,
    corners: np.ndarray,
    target_width: int = 800,
    target_height: int = 1000,
) -> np.ndarray:
    """Bản OpenCV của warp_perspective_sheet (cùng quy ước sắp xếp góc và tọa độ đích)."""
    ordered, _ = order_corners(corners)
    size = (target_width, target_height)
    H = get_perspective_transform_cv(ordered, destination_corners(size))
    return warp_perspective_cv(image, H, size)
