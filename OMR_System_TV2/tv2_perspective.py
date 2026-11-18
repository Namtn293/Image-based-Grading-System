"""
Module: TV2 - Nắn thẳng phối cảnh & Định vị Marker (Perspective & Registration)
Đề tài: Hệ thống Chấm bài Trắc nghiệm Tự động (OMR)
Môn học: Xử lý ảnh số (INT13146) - Học viện Công nghệ Bưu chính Viễn thông (PTIT)

Lớp vỏ tương thích (compatibility facade): mã nguồn thực nằm trong gói `tv2/`.
TV4 (app.py, pipeline_coordinator.py, tv4_enhancement_eval.py) vẫn `import tv2_perspective`
như cũ nên tên module và chữ ký các hàm công khai được giữ nguyên.
"""

from tv2 import (  # noqa: F401  (re-export)
    PerspectiveConfig,
    corners_for_angle,
    destination_corners,
    detect_markers_and_correct_rotation,
    match_template_ncc,
    order_corners,
    process_perspective_pipeline,
    quad_size,
    solve_homography_svd,
    warp_perspective_bilinear,
    warp_perspective_sheet,
)
from tv2 import __all__  # noqa: F401
