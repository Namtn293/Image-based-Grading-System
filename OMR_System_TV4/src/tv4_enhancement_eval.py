"""Module TV4: Image Enhancement, Benchmarking, CSV Matrix I/O, and Grading Logic
Thành viên 4: Cải thiện chất lượng ảnh số, Đo lường định lượng & Chấm điểm OMR

Kiến trúc mô-đun hóa (Modular Architecture):
  1. tv4_filters.py : Thuật toán xử lý ảnh số Scratch (Lọc trung vị, Cân bằng lược đồ mức xám).
  2. tv4_metrics.py : Bộ đo lường định lượng (MSE, PSNR, IoU, Dice, Delta-tau, Benchmark & Ablation).
  3. tv4_grader.py  : Nghiệp vụ chấm thi OMR, I/O Ma trận CSV & Xuất báo cáo Excel chuyên nghiệp.

File này đóng vai trò Facade / Re-export tập trung nhằm đảm bảo tương thích ngược 100%
với toàn bộ unit tests, pipeline coordinator, demo script và giao diện Streamlit.
"""

from __future__ import annotations

# 1. Image Enhancement & Scratch Noise Filtering
try:
    from .tv4_filters import (
        histogram_equalization_scratch,
        _reflect_index,
        _pad_2d_scratch,
        median_filter_scratch,
        benchmark_hist_equalization,
        benchmark_median_filter,
    )
except ImportError:
    from tv4_filters import (
        histogram_equalization_scratch,
        _reflect_index,
        _pad_2d_scratch,
        median_filter_scratch,
        benchmark_hist_equalization,
        benchmark_median_filter,
    )

# 2. Benchmarks, Metrics & Ablation Studies
try:
    from .tv4_metrics import (
        calculate_mse,
        calculate_psnr,
        calculate_difference_map,
        calculate_binary_iou_dice,
        calculate_otsu_delta_t,
        calculate_corner_error,
        VALID_OMR_OPTIONS,
        BLANK_TOKENS,
        _normalize_mark_value,
        calculate_mark_accuracy,
        benchmark_all_modules,
        _extract_bubble_answers_from_binary,
        run_ablation_study,
    )
except ImportError:
    from tv4_metrics import (
        calculate_mse,
        calculate_psnr,
        calculate_difference_map,
        calculate_binary_iou_dice,
        calculate_otsu_delta_t,
        calculate_corner_error,
        VALID_OMR_OPTIONS,
        BLANK_TOKENS,
        _normalize_mark_value,
        calculate_mark_accuracy,
        benchmark_all_modules,
        _extract_bubble_answers_from_binary,
        run_ablation_study,
    )

# 3. Grading Engine, CSV Matrix I/O & Excel Reports
try:
    from .tv4_grader import (
        save_matrix_to_csv,
        load_matrix_from_csv,
        OMRGrader,
    )
except ImportError:
    from tv4_grader import (
        save_matrix_to_csv,
        load_matrix_from_csv,
        OMRGrader,
    )

__all__ = [
    # Scratch Algorithms & Boundary Handling
    "histogram_equalization_scratch",
    "_reflect_index",
    "_pad_2d_scratch",
    "median_filter_scratch",
    "benchmark_hist_equalization",
    "benchmark_median_filter",
    # Evaluation Metrics
    "calculate_mse",
    "calculate_psnr",
    "calculate_difference_map",
    "calculate_binary_iou_dice",
    "calculate_otsu_delta_t",
    "calculate_corner_error",
    "VALID_OMR_OPTIONS",
    "BLANK_TOKENS",
    "_normalize_mark_value",
    "calculate_mark_accuracy",
    "benchmark_all_modules",
    "_extract_bubble_answers_from_binary",
    "run_ablation_study",
    # Matrix CSV I/O & Grader Engine
    "save_matrix_to_csv",
    "load_matrix_from_csv",
    "OMRGrader",
]
