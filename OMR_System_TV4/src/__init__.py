"""OMR System - Automatic Optical Mark Recognition Package
Thành viên 4: Cải thiện chất lượng ảnh số, Đo lường định lượng & Chấm điểm OMR
"""

__version__ = "1.0.0"

from .tv4_filters import (
    histogram_equalization_scratch,
    median_filter_scratch,
    create_anchor_marker_template,
    benchmark_hist_equalization,
    benchmark_median_filter,
)
from .tv4_metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_difference_map,
    calculate_binary_iou_dice,
    calculate_ccl_component_iou,
    calculate_otsu_delta_t,
    calculate_corner_error,
    calculate_mark_accuracy,
    benchmark_all_modules,
    run_ablation_study,
)
from .tv4_grader import (
    save_matrix_to_csv,
    load_matrix_from_csv,
    OMRGrader,
)
from .pipeline_coordinator import OMRPipeline

__all__ = [
    "histogram_equalization_scratch",
    "median_filter_scratch",
    "create_anchor_marker_template",
    "benchmark_hist_equalization",
    "benchmark_median_filter",
    "calculate_mse",
    "calculate_psnr",
    "calculate_difference_map",
    "calculate_binary_iou_dice",
    "calculate_ccl_component_iou",
    "calculate_otsu_delta_t",
    "calculate_corner_error",
    "calculate_mark_accuracy",
    "benchmark_all_modules",
    "run_ablation_study",
    "save_matrix_to_csv",
    "load_matrix_from_csv",
    "OMRGrader",
    "OMRPipeline",
]
