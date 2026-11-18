"""Định vị Marker đặc biệt bằng NCC và chuẩn hóa hướng phiếu về 0°."""

from typing import Dict, List, Tuple

import numpy as np

from .config import DEFAULT_COLOR_ORDER
from .template_matching import match_template_ncc

# Góc xoay thuận chiều kim đồng hồ của phiếu, suy ra từ vị trí Anchor Marker (chuẩn ở TR)
ANGLE_BY_ANCHOR_CORNER: Dict[str, int] = {"TR": 0, "BR": 90, "BL": 180, "TL": 270}


def to_gray(img: np.ndarray, color_order: str = DEFAULT_COLOR_ORDER) -> np.ndarray:
    """Chuyển ảnh màu sang mức xám BT.601 (float64); ảnh 2D được trả nguyên."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Ảnh màu phải có shape (H, W, >=3), nhận được {img.shape}")
    order = color_order.lower()
    if order == "bgr":
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
    elif order == "rgb":
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
    else:
        raise ValueError("color_order chỉ nhận 'bgr' hoặc 'rgb'.")
    return 0.299 * r.astype(np.float64) + 0.587 * g + 0.114 * b


def detect_markers_and_correct_rotation(
    warped_img: np.ndarray,
    anchor_template: np.ndarray,
    search_margin_ratio: float = 0.25,
    color_order: str = DEFAULT_COLOR_ORDER,
) -> Tuple[np.ndarray, int, List[Tuple[str, float]]]:
    """
    Xác định vị trí Marker đặc biệt (Anchor Marker) ở 4 góc để tự động
    nhận biết góc xoay của phiếu thi (0°, 90°, 180°, 270°) và xoay phiếu về 0°.

    Giả định thiết kế phiếu chuẩn:
    - Anchor Marker đặt ở góc Top-Right (TR) khi phiếu ở hướng 0°;
      3 góc còn lại dùng marker khác mẫu, nếu không thì không thể phân biệt hướng.
    - np.rot90 với k > 0 xoay NGƯỢC chiều kim đồng hồ:
      + Anchor ở TR --> 0°                       --> giữ nguyên.
      + Anchor ở BR --> phiếu xoay 90° thuận     --> np.rot90(k=1).
      + Anchor ở BL --> phiếu xoay 180°          --> np.rot90(k=2).
      + Anchor ở TL --> phiếu xoay 270° thuận    --> np.rot90(k=3).

    Args:
        warped_img: Ảnh phiếu đã nắn thẳng (mức xám hoặc màu).
        anchor_template: Mẫu Marker đặc biệt (mức xám 2D).
        search_margin_ratio: Tỷ lệ cạnh ảnh dùng làm vùng quét ở mỗi góc (mặc định 25%).
        color_order: Thứ tự kênh khi warped_img là ảnh màu ('bgr' | 'rgb').

    Returns:
        corrected_img: Ảnh đã xoay về hướng 0°.
        detected_angle: Góc xoay phát hiện được (0, 90, 180, 270).
        corner_scores: Điểm NCC cực đại tại 4 góc [TL, TR, BR, BL].
    """
    anchor = np.asarray(anchor_template)
    if anchor.ndim != 2:
        raise ValueError("anchor_template phải là ảnh mức xám 2D.")

    gray = to_gray(warped_img, color_order)
    h, w = gray.shape[:2]
    mh = int(h * search_margin_ratio)
    mw = int(w * search_margin_ratio)

    rois = {
        "TL": gray[0:mh, 0:mw],
        "TR": gray[0:mh, w - mw:w],
        "BR": gray[h - mh:h, w - mw:w],
        "BL": gray[h - mh:h, 0:mw],
    }

    scores: Dict[str, float] = {}
    for corner, roi in rois.items():
        if roi.shape[0] >= anchor.shape[0] and roi.shape[1] >= anchor.shape[1]:
            scores[corner] = float(np.max(match_template_ncc(roi, anchor)))
        else:
            scores[corner] = -1.0

    angle = ANGLE_BY_ANCHOR_CORNER[max(scores, key=scores.get)]
    k_fix = angle // 90
    corrected = np.rot90(warped_img, k=k_fix).copy() if k_fix else np.array(warped_img, copy=True)
    return corrected, angle, list(scores.items())


def corners_for_angle(ordered: np.ndarray, angle: int) -> np.ndarray:
    """Gán lại 4 đỉnh trên ảnh gốc cho đúng vị trí [TL, TR, BR, BL] của phiếu ở hướng 0°.

    Phiếu xoay thuận chiều kim đồng hồ `angle` độ thì đỉnh TL thật nằm ở vị trí
    thứ (angle / 90) theo chiều kim đồng hồ -> dịch vòng mảng đỉnh đi -(angle / 90).
    """
    if angle not in (0, 90, 180, 270):
        raise ValueError(f"Góc xoay chỉ nhận 0/90/180/270, nhận được {angle}")
    return np.roll(ordered, -(angle // 90), axis=0)
