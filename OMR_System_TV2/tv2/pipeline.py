"""Hàm đóng gói Giai đoạn 2: nắn phẳng phối cảnh + (tùy chọn) chuẩn hóa hướng phiếu."""

from typing import List, Optional, Tuple

import numpy as np

from .config import DEFAULT_COLOR_ORDER, PerspectiveConfig
from .geometry import destination_corners, order_corners, quad_size
from .homography import solve_homography_svd
from .registration import corners_for_angle, detect_markers_and_correct_rotation
from .warping import warp_perspective_bilinear


def _warp(img: np.ndarray, corners: np.ndarray, size: Tuple[int, int], tol: float) -> Tuple[np.ndarray, np.ndarray]:
    H = solve_homography_svd(corners, destination_corners(size), tol=tol)
    return warp_perspective_bilinear(img, H, size), H


def process_perspective_pipeline(
    img: np.ndarray,
    raw_corners: np.ndarray,
    anchor_template: Optional[np.ndarray] = None,
    target_size: Optional[Tuple[int, int]] = None,
    search_margin_ratio: float = 0.25,
    color_order: str = DEFAULT_COLOR_ORDER,
    config: Optional[PerspectiveConfig] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Pipeline của Thành viên 2: nhận 4 góc từ TV1 -> trả ảnh phẳng chuẩn 0° cho TV3.

    Các bước:
    1. Sắp xếp 4 đỉnh [TL, TR, BR, BL] và tính kích thước phẳng tự động.
    2. Giải Homography bằng Normalized DLT + SVD.
    3. Nắn phẳng bằng Inverse Warping + Bilinear Interpolation.
    4. (Tùy chọn) Dò Anchor Marker bằng NCC trên ảnh nắn thử để tìm góc xoay; gán lại
       4 đỉnh cho đúng hướng 0° rồi nắn lại MỘT lần từ ảnh gốc — ảnh ra luôn đúng
       target_size (không méo khi xoay 90°/270°) và chỉ chịu một lần nội suy.

    Args:
        img: Ảnh gốc (mức xám hoặc màu).
        raw_corners: 4 đỉnh tứ giác từ TV1, shape (4, 2).
        anchor_template: Mẫu Anchor Marker (mức xám 2D) để định hướng; None = bỏ qua bước 4.
        target_size: Kích thước ép buộc (W, H); None = tự tính từ 4 góc.
        search_margin_ratio: Tỷ lệ vùng góc quét Marker.
        color_order: Thứ tự kênh khi ảnh là ảnh màu ('bgr' | 'rgb').
        config: Nếu truyền, ghi đè target_size / search_margin_ratio / color_order.

    Returns:
        warped_sheet: Ảnh phiếu phẳng chuẩn 0°.
        info: Metadata (ordered_corners, homography_matrix, output_size, detected_angle, marker_scores).
    """
    tol = PerspectiveConfig().degenerate_tol
    if config is not None:
        target_size = config.target_size
        search_margin_ratio = config.search_margin_ratio
        color_order = config.color_order
        tol = config.degenerate_tol

    ordered_pts, auto_size = order_corners(raw_corners)
    detected_angle = 0
    scores: List[Tuple[str, float]] = []

    if anchor_template is None:
        final_corners = ordered_pts
        out_size = tuple(target_size) if target_size is not None else auto_size
    else:
        # Lượt 1: nắn theo đúng tỉ lệ khung hình đang thấy trên ảnh để dò Marker
        preview_size = auto_size
        if target_size is not None:
            tw, th = target_size
            same_orientation = (auto_size[0] > auto_size[1]) == (tw > th)
            preview_size = (tw, th) if same_orientation else (th, tw)
        preview, _ = _warp(img, ordered_pts, preview_size, tol)
        _, detected_angle, scores = detect_markers_and_correct_rotation(
            preview, anchor_template, search_margin_ratio=search_margin_ratio, color_order=color_order
        )

        # Lượt 2: gán lại 4 đỉnh cho hướng 0° và nắn trực tiếp từ ảnh gốc
        final_corners = corners_for_angle(ordered_pts, detected_angle)
        out_size = tuple(target_size) if target_size is not None else quad_size(final_corners)

    warped, H = _warp(img, final_corners, out_size, tol)
    info = {
        "ordered_corners": final_corners,
        "homography_matrix": H,
        "output_size": out_size,
        "detected_angle": detected_angle,
        "marker_scores": scores,
    }
    return warped, info


def warp_perspective_sheet(
    image: np.ndarray,
    corners: np.ndarray,
    target_width: int = 800,
    target_height: int = 1000,
    use_scratch: bool = True,
    anchor_template: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Hàm giao tiếp chuẩn với Pipeline Coordinator của nhóm (TV4).

    Args:
        image: Ảnh đầu vào (mức xám hoặc màu).
        corners: Tọa độ 4 góc (shape 4x2).
        target_width, target_height: Kích thước ảnh chuẩn (mặc định 800 x 1000).
        use_scratch: True = 100% NumPy của TV2; False = OpenCV đối chứng
                     (OpenCV chỉ được import khi nhánh này thực sự được gọi).
        anchor_template: (Chế độ Scratch) Nếu truyền vào, tự động sửa hướng 90°/180°/270° bằng NCC.

    Returns:
        Ảnh đã nắn thẳng, shape (target_height, target_width[, C]).
    """
    if not use_scratch:
        from .baseline_opencv import warp_perspective_sheet_cv
        return warp_perspective_sheet_cv(image, corners, target_width, target_height)

    warped, _ = process_perspective_pipeline(
        image, corners, anchor_template=anchor_template, target_size=(target_width, target_height)
    )
    return warped
