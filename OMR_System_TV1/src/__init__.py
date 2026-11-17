"""Package Initialization: OMR System - Module Thành viên 1 (Stage 1 Preprocessing)
Học viện Công nghệ Bưu chính Viễn thông (PTIT)
Môn học: Xử lý ảnh số (INT13146)
"""

from .tv1_grayscale_blur import (
    rgb_to_gray_scratch,
    grayscale_luminance,
    get_gaussian_kernel_1d,
    get_gaussian_kernel_2d,
    pad_image_reflect,
    convolve2d_scratch,
    gaussian_blur_2d,
    gaussian_blur_scratch
)

from .tv1_canny import (
    sobel_gradients,
    non_maximum_suppression,
    double_threshold_hysteresis,
    canny_edge_detector,
    canny_edge_scratch
)

from .tv1_hough import (
    hough_lines_standard,
    intersect_lines_polar,
    order_quad_corners,
    convex_hull_2d,
    ramer_douglas_peucker,
    simplify_closed_polygon,
    apply_parameter_space_nms,
    otsu_threshold_scratch,
    collect_boundary_extremes,
    score_lines_by_edge_support,
    detect_quad_from_hough_lines,
    detect_quad_from_boundary_scratch
)

from .tv1_preprocessing import (
    detect_sheet_corners,
    process_preprocessing_pipeline,
    run_unit_benchmarks,
    calculate_mse,
    calculate_psnr,
    calculate_binary_iou_dice,
    calculate_line_match_rate
)

__all__ = [
    "rgb_to_gray_scratch",
    "grayscale_luminance",
    "get_gaussian_kernel_1d",
    "get_gaussian_kernel_2d",
    "pad_image_reflect",
    "convolve2d_scratch",
    "gaussian_blur_2d",
    "gaussian_blur_scratch",
    "sobel_gradients",
    "non_maximum_suppression",
    "double_threshold_hysteresis",
    "canny_edge_detector",
    "canny_edge_scratch",
    "hough_lines_standard",
    "intersect_lines_polar",
    "order_quad_corners",
    "convex_hull_2d",
    "ramer_douglas_peucker",
    "simplify_closed_polygon",
    "apply_parameter_space_nms",
    "otsu_threshold_scratch",
    "collect_boundary_extremes",
    "score_lines_by_edge_support",
    "detect_quad_from_hough_lines",
    "detect_quad_from_boundary_scratch",
    "detect_sheet_corners",
    "process_preprocessing_pipeline",
    "run_unit_benchmarks",
    "calculate_mse",
    "calculate_psnr",
    "calculate_binary_iou_dice",
    "calculate_line_match_rate"
]
