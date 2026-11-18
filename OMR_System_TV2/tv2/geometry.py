"""Sắp xếp 4 đỉnh tứ giác [TL, TR, BR, BL] và tính kích thước phẳng đích."""

from typing import Tuple

import numpy as np


def _as_quad(pts: np.ndarray) -> np.ndarray:
    """Chuẩn hóa đầu vào về mảng float32 shape (4, 2); chấp nhận cả dạng contour (4, 1, 2)."""
    arr = np.asarray(pts, dtype=np.float32)
    if arr.size == 8:
        arr = arr.reshape(4, 2)
    if arr.shape != (4, 2):
        raise ValueError(f"Tọa độ đầu vào phải có shape (4, 2), nhận được: {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Tọa độ 4 góc chứa giá trị NaN/Inf.")
    return arr


def _order_by_angle(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 đỉnh theo góc cực quanh trọng tâm (dự phòng khi quy tắc tổng/hiệu suy biến).

    Trong hệ tọa độ ảnh (trục y hướng xuống), sắp xếp tăng dần theo atan2(dy, dx)
    cho thứ tự cùng chiều kim đồng hồ: TL -> TR -> BR -> BL. Sau đó xoay vòng để
    đỉnh có tổng (x + y) nhỏ nhất đứng đầu.
    """
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def quad_size(ordered: np.ndarray) -> Tuple[int, int]:
    """Kích thước phẳng đích (W_dst, H_dst) = cạnh ngang / cạnh dọc dài nhất của tứ giác [TL, TR, BR, BL]."""
    tl, tr, br, bl = np.asarray(ordered, dtype=np.float64)
    w_dst = int(round(max(np.hypot(*(tr - tl)), np.hypot(*(br - bl)))))
    h_dst = int(round(max(np.hypot(*(bl - tl)), np.hypot(*(br - tr)))))
    return max(w_dst, 1), max(h_dst, 1)


def order_corners(pts: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Sắp xếp 4 đỉnh của tứ giác theo thứ tự chuẩn:
    [Top-Left (TL), Top-Right (TR), Bottom-Right (BR), Bottom-Left (BL)]
    và tính kích thước (W_dst, H_dst) cho ảnh phẳng đích.

    Bản chất toán học:
    - TL: Điểm có tổng (x + y) nhỏ nhất.
    - BR: Điểm có tổng (x + y) lớn nhất.
    - TR: Điểm có hiệu (y - x) nhỏ nhất (tương đương x - y lớn nhất).
    - BL: Điểm có hiệu (y - x) lớn nhất.

    Khi tứ giác bị xoay gần 45°, quy tắc tổng/hiệu có thể chọn trùng một đỉnh cho
    hai vị trí. Trường hợp đó được phát hiện và chuyển sang sắp xếp theo góc cực
    quanh trọng tâm để luôn trả về 4 đỉnh phân biệt.

    Args:
        pts: Mảng shape (4, 2) (hoặc (4, 1, 2)) chứa 4 tọa độ (x, y) bất kỳ.

    Returns:
        ordered: Mảng float32 shape (4, 2) theo thứ tự [TL, TR, BR, BL].
        (W_dst, H_dst): Chiều rộng và chiều cao đích (làm tròn int).
    """
    quad = _as_quad(pts)

    s = quad.sum(axis=1)
    diff = quad[:, 1] - quad[:, 0]
    idx = [int(np.argmin(s)), int(np.argmin(diff)), int(np.argmax(s)), int(np.argmax(diff))]

    ordered = quad[idx] if len(set(idx)) == 4 else _order_by_angle(quad)
    return ordered.astype(np.float32), quad_size(ordered)


def destination_corners(size: Tuple[int, int]) -> np.ndarray:
    """4 góc đích [TL, TR, BR, BL] tại tâm các pixel biên: [0, W-1] x [0, H-1] (quy ước của OpenCV)."""
    w, h = int(size[0]), int(size[1])
    return np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float64)
