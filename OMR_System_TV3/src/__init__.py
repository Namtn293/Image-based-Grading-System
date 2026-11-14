"""
DỰ ÁN XỬ LÝ ẢNH (INT13146) - NHÓM 4
Gói mã nguồn chính (src package)

Module Thành viên 3:
    - tv3_thresholding: Phân ngưỡng Otsu toàn cục & Adaptive Mean Thresholding
    - tv3_morphology_ccl: Hình thái học nhị phân & Two-Pass CCL
    - tv3_roi_classifier: Trích xuất Circular ROI, Fill Ratio & Bộ phân loại 4 trạng thái
    - tv3_segmentation: Pipeline tích hợp toàn diện và kiểm thử nghiệm thu
"""

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from src.tv3_thresholding import (
        compute_histogram_scratch,
        otsu_threshold_scratch,
        compute_integral_image_scratch,
        adaptive_mean_threshold_scratch,
        calculate_segmentation_metrics,
        benchmark_otsu_vs_opencv,
        benchmark_adaptive_vs_opencv,
    )

    from src.tv3_morphology_ccl import (
        create_structuring_element,
        erode_binary_scratch,
        dilate_binary_scratch,
        opening_binary_scratch,
        closing_binary_scratch,
        morphology_binary_scratch,
        DisjointSet,
        two_pass_ccl_scratch,
        filter_bubble_components,
        calc_binary_metrics,
        benchmark_morphology_vs_opencv,
        benchmark_ccl_vs_opencv,
    )

    from src.tv3_roi_classifier import (
        STATUS_VALID,
        STATUS_BLANK,
        STATUS_ERASED_VALID,
        STATUS_MULTIPLE_INVALID,
        DEFAULT_TAU_MIN,
        DEFAULT_DELTA_TAU_MIN,
        DEFAULT_BUBBLE_RADIUS,
        DEFAULT_OPTIONS,
        BubbleROI,
        QuestionResult,
        OMRRecognitionResult,
        create_circular_mask_scratch,
        compute_bubble_fill_ratio_scratch,
        classify_question_bubbles,
        generate_grid_bubble_rois,
        auto_align_grid_from_centroids,
        recognize_all_questions,
    )

    from src.tv3_segmentation import (
        TV3PipelineConfig,
        TV3PipelineResult,
        TV3SegmentationPipeline,
        run_tv3_pipeline,
        extract_bubble_answers,
        generate_synthetic_omr_sheet,
        visualize_pipeline_steps,
    )
except ModuleNotFoundError:
    from tv3_thresholding import (
        compute_histogram_scratch,
        otsu_threshold_scratch,
        compute_integral_image_scratch,
        adaptive_mean_threshold_scratch,
        calculate_segmentation_metrics,
        benchmark_otsu_vs_opencv,
        benchmark_adaptive_vs_opencv,
    )

    from tv3_morphology_ccl import (
        create_structuring_element,
        erode_binary_scratch,
        dilate_binary_scratch,
        opening_binary_scratch,
        closing_binary_scratch,
        morphology_binary_scratch,
        DisjointSet,
        two_pass_ccl_scratch,
        filter_bubble_components,
        calc_binary_metrics,
        benchmark_morphology_vs_opencv,
        benchmark_ccl_vs_opencv,
    )

    from tv3_roi_classifier import (
        STATUS_VALID,
        STATUS_BLANK,
        STATUS_ERASED_VALID,
        STATUS_MULTIPLE_INVALID,
        DEFAULT_TAU_MIN,
        DEFAULT_DELTA_TAU_MIN,
        DEFAULT_BUBBLE_RADIUS,
        DEFAULT_OPTIONS,
        BubbleROI,
        QuestionResult,
        OMRRecognitionResult,
        create_circular_mask_scratch,
        compute_bubble_fill_ratio_scratch,
        classify_question_bubbles,
        generate_grid_bubble_rois,
        auto_align_grid_from_centroids,
        recognize_all_questions,
    )

    from tv3_segmentation import (
        TV3PipelineConfig,
        TV3PipelineResult,
        TV3SegmentationPipeline,
        run_tv3_pipeline,
        extract_bubble_answers,
        generate_synthetic_omr_sheet,
        visualize_pipeline_steps,
    )

__all__ = [
    # Thresholding
    "compute_histogram_scratch",
    "otsu_threshold_scratch",
    "compute_integral_image_scratch",
    "adaptive_mean_threshold_scratch",
    "calculate_segmentation_metrics",
    "benchmark_otsu_vs_opencv",
    "benchmark_adaptive_vs_opencv",
    # Morphology & CCL
    "create_structuring_element",
    "erode_binary_scratch",
    "dilate_binary_scratch",
    "opening_binary_scratch",
    "closing_binary_scratch",
    "morphology_binary_scratch",
    "DisjointSet",
    "two_pass_ccl_scratch",
    "filter_bubble_components",
    "calc_binary_metrics",
    "benchmark_morphology_vs_opencv",
    "benchmark_ccl_vs_opencv",
    # ROI & Classification
    "STATUS_VALID",
    "STATUS_BLANK",
    "STATUS_ERASED_VALID",
    "STATUS_MULTIPLE_INVALID",
    "DEFAULT_TAU_MIN",
    "DEFAULT_DELTA_TAU_MIN",
    "DEFAULT_BUBBLE_RADIUS",
    "DEFAULT_OPTIONS",
    "BubbleROI",
    "QuestionResult",
    "OMRRecognitionResult",
    "create_circular_mask_scratch",
    "compute_bubble_fill_ratio_scratch",
    "classify_question_bubbles",
    "generate_grid_bubble_rois",
    "auto_align_grid_from_centroids",
    "recognize_all_questions",
    # Pipeline & Demo
    "TV3PipelineConfig",
    "TV3PipelineResult",
    "TV3SegmentationPipeline",
    "run_tv3_pipeline",
    "extract_bubble_answers",
    "generate_synthetic_omr_sheet",
    "visualize_pipeline_steps",
]
