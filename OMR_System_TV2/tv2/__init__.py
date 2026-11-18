"""TV2 - Nắn thẳng phối cảnh & Định vị Marker (Perspective & Registration).

Toàn bộ thuật toán cài đặt từ đầu bằng NumPy; gói này không import OpenCV.
OpenCV chỉ nằm trong `tv2.baseline_opencv` (nạp khi cần đối chứng) và `tv2.benchmark`.
"""

from .config import PerspectiveConfig
from .geometry import destination_corners, order_corners, quad_size
from .homography import build_dlt_matrix, solve_homography_svd
from .pipeline import process_perspective_pipeline, warp_perspective_sheet
from .registration import corners_for_angle, detect_markers_and_correct_rotation, to_gray
from .template_matching import match_template_ncc
from .warping import warp_perspective_bilinear

__all__ = [
    "PerspectiveConfig",
    "order_corners",
    "quad_size",
    "destination_corners",
    "build_dlt_matrix",
    "solve_homography_svd",
    "warp_perspective_bilinear",
    "match_template_ncc",
    "to_gray",
    "detect_markers_and_correct_rotation",
    "corners_for_angle",
    "process_perspective_pipeline",
    "warp_perspective_sheet",
]
