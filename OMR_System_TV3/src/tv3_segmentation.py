"""
DỰ ÁN XỬ LÝ ẢNH (INT13146) - HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM TỰ ĐỘNG (OMR)
Module: tv3_segmentation.py (GIAI ĐOẠN 3: PHÂN ĐOẠN ẢNH & NHẬN DIỆN Ô TÔ ĐÁP ÁN)
Phân công: Thành viên 3 - Trưởng nhóm Kỹ thuật Phân đoạn ảnh

Nhiệm vụ:
    1. Re-export toàn bộ các hàm và lớp từ 3 module thành phần:
       - tv3_thresholding.py: Phân ngưỡng Otsu toàn cục & Adaptive Mean Thresholding qua Integral Image.
       - tv3_morphology_ccl.py: 4 phép toán Hình thái học nhị phân & Two-Pass CCL + DisjointSet.
       - tv3_roi_classifier.py: Circular ROI Mask, Fill Ratio tau_{q,c}, phân loại 4 trạng thái & gom lưới.
    2. Cung cấp Pipeline tích hợp đầu-cuối: `TV3SegmentationPipeline` và `run_tv3_pipeline(...)`.
    3. Hỗ trợ đối chứng song song Scratch vs OpenCV (MSE, PSNR, IoU, Dice, Delta T, Latency).
    4. Trình diễn Demo độc lập khi chạy `python src/tv3_segmentation.py`:
       - Tự động sinh ảnh phiếu OMR mô phỏng (chuẩn, bóng đổ DS-03, tẩy xóa, tô đúp).
       - Xuất toàn bộ ảnh các bước trung gian vào `outputs/tv3/`.
       - Xuất báo cáo số liệu thực nghiệm CSV và JSON.

Nguyên tắc bắt buộc:
    - 100% luồng xử lý chính chạy bằng thuật toán tự cài đặt (Code from Scratch) bằng NumPy.
    - OpenCV chỉ được gọi độc lập để đối chứng số học và thời gian thực thi.
"""

from __future__ import annotations

import os
import sys
import time
import json
import csv
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ==============================================================================
# 1. RE-EXPORT ĐẦY ĐỦ CÁC THÀNH PHẦN TỪ CÁC MODULE CON CỦA THÀNH VIÊN 3
# ==============================================================================
if __package__:
    from .tv3_thresholding import (
        compute_histogram_scratch,
        otsu_threshold_scratch,
        compute_integral_image_scratch,
        adaptive_mean_threshold_scratch,
        calculate_segmentation_metrics,
        benchmark_otsu_vs_opencv,
        benchmark_adaptive_vs_opencv,
    )

    from .tv3_morphology_ccl import (
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

    from .tv3_roi_classifier import (
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
else:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    if _current_dir not in sys.path:
        sys.path.append(_current_dir)

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

# Aliases tương thích ngược cho các module tích hợp (như TV4)
morphology_open_scratch = opening_binary_scratch
morphology_close_scratch = closing_binary_scratch


# ==============================================================================
# 2. CẤU HÌNH VÀ KẾT QUẢ PIPELINE TÍCH HỢP
# ==============================================================================

@dataclass
class TV3PipelineConfig:
    """Cấu hình tham số cho Pipeline Phân đoạn & Nhận diện OMR (Thành viên 3)."""
    # 1. Tham số phân ngưỡng
    threshold_method: str = "adaptive"  # 'adaptive' (khuyên dùng cho OMR/bóng đổ) hoặc 'otsu'
    block_size: int = 25                # Kích thước cửa sổ b x b cho Adaptive Mean (lẻ >= 3)
    C: float = 10.0                     # Hằng số C trừ đi từ ngưỡng trung bình cục bộ
    binary_method: str = "BINARY_INV"   # 'BINARY_INV': chữ/ô chì màu trắng (255), nền đen (0)

    # 2. Tham số Hình thái học (Morphology)
    apply_opening: bool = True          # Phép Mở: xóa nhiễu muối tiêu, bụi chì và gai nét
    apply_closing: bool = True          # Phép Đóng: lấp đầy lỗ hổng bên trong vết chì tô
    morph_kernel_shape: str = "RECT"    # 'RECT', 'CROSS', 'ELLIPSE'
    morph_kernel_size: Tuple[int, int] = (3, 3)

    # 3. Tham số Gán nhãn thành phần liên thông (Two-Pass CCL)
    connectivity: int = 8               # 8-lân cận hoặc 4-lân cận
    filter_bubbles: bool = True         # Bật/tắt bộ lọc hình học chọn lọc ô trắc nghiệm
    min_area: int = 40                  # Diện tích tối thiểu của ô tròn (pixel)
    max_area: int = 2500                # Diện tích tối đa của ô tròn (pixel)
    aspect_ratio_range: Tuple[float, float] = (0.6, 1.6)
    min_extent: float = 0.40

    # 4. Tham số Nhận diện ROI và Phân loại đáp án (dùng chung cho Scratch & OpenCV)
    num_questions: int = 20             # Số lượng câu hỏi trên phiếu (mặc định 20 câu)
    options: Sequence[str] = field(default_factory=lambda: ("A", "B", "C", "D"))
    bubble_radius: float = DEFAULT_BUBBLE_RADIUS  # Bán kính ô tròn ngoài R (pixel)
    inner_radius_offset: float = 0.0              # Độ co bán kính vùng đo fill (ví dụ 3.0 -> R - 3) để loại viền in sẵn
    tau_min: float = DEFAULT_TAU_MIN              # Ngưỡng lấp đầy cơ sở (0.38)
    delta_tau_min: float = DEFAULT_DELTA_TAU_MIN  # Ngưỡng phân định vết tẩy xóa (0.15)
    use_auto_grid: bool = False                   # Tự động suy biến lưới từ tâm CCL nếu question_grid=None
    execution_mode: str = "scratch"               # 'scratch' (NumPy) hoặc 'opencv' (OpenCV baseline)

    @property
    def effective_fill_radius(self) -> float:
        """Bán kính vùng đo fill ratio thực tế (loại trừ viền in sẵn của ô)."""
        if self.inner_radius_offset > 0:
            return float(max(self.bubble_radius - self.inner_radius_offset, 4.0))
        return float(self.bubble_radius)


OMRSharedConfig = TV3PipelineConfig


@dataclass
class TV3PipelineResult:
    """Cấu trúc dữ liệu chứa toàn bộ kết quả phân đoạn và nhận diện OMR."""
    # Ảnh gốc đầu vào (đã nắn phẳng phối cảnh)
    warped_gray: np.ndarray

    # Ảnh nhị phân các bước trung gian (Scratch hoặc OpenCV theo execution_mode)
    binary_thresholded: np.ndarray
    threshold_value: Optional[float]
    binary_morph: np.ndarray

    # Kết quả Two-Pass CCL
    num_components: int
    labels: np.ndarray
    raw_stats: List[Dict[str, Any]]
    filtered_bubble_stats: List[Dict[str, Any]]

    # Lưới tọa độ ROI và Kết quả nhận diện câu hỏi
    question_grid: Dict[int, Dict[str, Tuple[float, float]]]
    omr_result: OMRRecognitionResult

    # Thống kê thời gian thực thi (Latency in ms)
    latency_ms: Dict[str, float]

    # Báo cáo đối chứng số học với OpenCV (nếu được kích hoạt)
    benchmark_opencv: Optional[Dict[str, Any]] = None
    execution_mode: str = "scratch"

    def get_answer_vector(self) -> Dict[int, Optional[str]]:
        """Trả về vector đáp án nhận diện {1: 'A', 2: 'B', ...}."""
        return self.omr_result.answers_vector

    def get_summary(self) -> Dict[str, int]:
        """Trả về thống kê số lượng câu theo từng trạng thái (VALID, BLANK, ...)."""
        return self.omr_result.summary

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi toàn bộ kết quả thành dictionary JSON serializable."""
        return {
            "num_questions": len(self.omr_result.questions),
            "summary": self.get_summary(),
            "answers_vector": self.get_answer_vector(),
            "questions_detail": {qid: qres.to_dict() for qid, qres in self.omr_result.questions.items()},
            "num_components_ccl": self.num_components,
            "num_filtered_bubbles": len(self.filtered_bubble_stats),
            "threshold_value": self.threshold_value,
            "latency_ms": {k: round(v, 3) for k, v in self.latency_ms.items()},
            "benchmark_opencv": self.benchmark_opencv,
        }

    def save_json(self, output_path: str) -> None:
        """Lưu kết quả nhận diện ra file JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    def save_csv(self, output_path: str) -> None:
        """Lưu bảng kết quả nhận diện từng câu ra file CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Question_ID", "Selected_Option", "Status", "Is_Valid",
                "Tau_A", "Tau_B", "Tau_C", "Tau_D", "Delta_Tau", "Warning"
            ])
            for q_id, q_res in sorted(self.omr_result.questions.items()):
                fr = q_res.fill_ratios
                writer.writerow([
                    q_id,
                    q_res.selected or "None",
                    q_res.status,
                    1 if q_res.is_valid else 0,
                    f"{fr.get('A', 0.0):.4f}",
                    f"{fr.get('B', 0.0):.4f}",
                    f"{fr.get('C', 0.0):.4f}",
                    f"{fr.get('D', 0.0):.4f}",
                    f"{q_res.delta_tau:.4f}",
                    q_res.warning or ""
                ])


# ==============================================================================
# 3. LỚP PIPELINE TÍCH HỢP CHÍNH (TV3SegmentationPipeline)
# ==============================================================================

class TV3SegmentationPipeline:
    """
    Pipeline tích hợp toàn diện Module Phân đoạn ảnh và Nhận diện đáp án OMR (Thành viên 3).
    
    Quy trình xử lý tuần tự (100% Code from Scratch bằng NumPy):
        1. Phân ngưỡng nhị phân: Otsu toàn cục hoặc Adaptive Mean (Ảnh tích phân O(1)).
        2. Xử lý hình thái học (Binary Morphology): Phép Mở khử nhiễu và Phép Đóng lấp đầy lỗ.
        3. Gán nhãn thành phần liên thông 2 lượt (Two-Pass CCL) & Lọc ô tròn trắc nghiệm.
        4. Xác định lưới ROI (dùng lưới chỉ định trước hoặc suy biến từ tâm CCL).
        5. Tính Fill Ratio tròn tau_{q,c} và phân loại 4 trạng thái (VALID, BLANK, ERASED_VALID, MULTIPLE_INVALID).
        6. Chạy đối chứng song song với OpenCV để đo MSE, PSNR, IoU, độ lệch ngưỡng, tốc độ.
    """

    def __init__(self, config: Optional[TV3PipelineConfig] = None):
        self.config = config or TV3PipelineConfig()
        # Tạo phần tử cấu trúc Morphology trước để tái sử dụng
        self.structuring_element = create_structuring_element(
            shape=self.config.morph_kernel_shape,
            ksize=self.config.morph_kernel_size
        )

    def run(
        self,
        warped_gray: np.ndarray,
        question_grid: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
        run_opencv_benchmark: bool = True,
        execution_mode: Optional[str] = None
    ) -> TV3PipelineResult:
        """
        Thực thi pipeline trên ảnh xám phiếu thi đã được nắn thẳng phối cảnh.

        Tham số:
            warped_gray (np.ndarray): Ảnh mức xám 2D (uint8) phiếu thi OMR phẳng.
            question_grid (Optional[Dict]): Lưới tọa độ tâm ô đáp án. Nếu None, tự động sinh
                                            lưới chuẩn hoặc phân cụm từ CCL.
            run_opencv_benchmark (bool): Nếu True, chạy thêm pipeline đối chứng OpenCV để đo đạc.
            execution_mode (Optional[str]): 'scratch' hoặc 'opencv'. Nếu None lấy từ config.execution_mode.
                                            Cả hai chế độ dùng chung 100% bộ tham số trong self.config.

        Trả về:
            TV3PipelineResult: Đối tượng chứa toàn bộ kết quả phân đoạn, nhận diện và đối chứng.
        """
        if not isinstance(warped_gray, np.ndarray) or warped_gray.ndim != 2:
            raise ValueError(f"warped_gray phải là mảng 2 chiều np.ndarray, nhận được ndim={getattr(warped_gray, 'ndim', None)}")

        mode = (execution_mode or self.config.execution_mode or "scratch").strip().lower()
        if mode not in ("scratch", "opencv"):
            raise ValueError(f"execution_mode phải là 'scratch' hoặc 'opencv', nhận được: {mode}")

        img_gray = np.clip(warped_gray, 0, 255).astype(np.uint8)
        H, W = img_gray.shape
        latencies: Dict[str, float] = {}
        t_start_pipeline = time.perf_counter()

        if mode == "scratch":
            # ------------------------------------------------------------------
            # BƯỚC 1: PHÂN NGƯỠNG TỰ ĐỘNG (SCRATCH)
            # ------------------------------------------------------------------
            t0 = time.perf_counter()
            thresh_val: Optional[float] = None

            if self.config.threshold_method.lower() == "otsu":
                thresh_val, bin_thresh = otsu_threshold_scratch(
                    img_gray,
                    method=self.config.binary_method,
                    max_val=255
                )
            else:
                bin_thresh = adaptive_mean_threshold_scratch(
                    img_gray,
                    block_size=self.config.block_size,
                    C=self.config.C,
                    method=self.config.binary_method,
                    max_val=255,
                    border_type="replicate"
                )
            latencies["thresholding_ms"] = (time.perf_counter() - t0) * 1000.0

            # ------------------------------------------------------------------
            # BƯỚC 2: XỬ LÝ HÌNH THÁI HỌC NHỊ PHÂN (SCRATCH)
            # ------------------------------------------------------------------
            t0 = time.perf_counter()
            bin_morph = bin_thresh.copy()
            if self.config.apply_opening:
                bin_morph = opening_binary_scratch(bin_morph, self.structuring_element)
            if self.config.apply_closing:
                bin_morph = closing_binary_scratch(bin_morph, self.structuring_element)
            latencies["morphology_ms"] = (time.perf_counter() - t0) * 1000.0

            # ------------------------------------------------------------------
            # BƯỚC 3: GÁN NHÃN THÀNH PHẦN LIÊN THÔNG (TWO-PASS CCL SCRATCH)
            # ------------------------------------------------------------------
            t0 = time.perf_counter()
            num_components, labels, raw_stats = two_pass_ccl_scratch(
                bin_morph,
                connectivity=self.config.connectivity
            )

            filtered_stats = raw_stats
            if self.config.filter_bubbles and num_components > 0:
                _, filtered_stats = filter_bubble_components(
                    labels,
                    raw_stats,
                    min_area=self.config.min_area,
                    max_area=self.config.max_area,
                    aspect_ratio_range=self.config.aspect_ratio_range,
                    min_extent=self.config.min_extent
                )
            latencies["ccl_ms"] = (time.perf_counter() - t0) * 1000.0
        else:
            # ------------------------------------------------------------------
            # CHẾ ĐỘ OPENCV DÙNG CHUNG BỘ THAM SỐ CONFIG VỚI SCRATCH
            # ------------------------------------------------------------------
            import cv2

            cv_flag = cv2.THRESH_BINARY_INV if self.config.binary_method.upper() in ("BINARY_INV", "INV") else cv2.THRESH_BINARY
            t0 = time.perf_counter()
            thresh_val = None
            if self.config.threshold_method.lower() == "otsu":
                ret, bin_thresh = cv2.threshold(img_gray, 0, 255, cv_flag + cv2.THRESH_OTSU)
                thresh_val = float(ret)
            else:
                bin_thresh = cv2.adaptiveThreshold(
                    img_gray,
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv_flag,
                    self.config.block_size,
                    self.config.C
                )
            latencies["thresholding_ms"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            kh, kw = self.config.morph_kernel_size
            cv_kshape = {
                "RECT": cv2.MORPH_RECT,
                "CROSS": cv2.MORPH_CROSS,
                "ELLIPSE": cv2.MORPH_ELLIPSE
            }.get(self.config.morph_kernel_shape.upper(), cv2.MORPH_RECT)
            kernel_cv = cv2.getStructuringElement(cv_kshape, (kw, kh))

            bin_morph = bin_thresh.copy()
            if self.config.apply_opening:
                bin_morph = cv2.morphologyEx(bin_morph, cv2.MORPH_OPEN, kernel_cv)
            if self.config.apply_closing:
                bin_morph = cv2.morphologyEx(bin_morph, cv2.MORPH_CLOSE, kernel_cv)
            latencies["morphology_ms"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            num_labels_cv, labels_cv, stats_cv, cents_cv = cv2.connectedComponentsWithStats(
                bin_morph,
                connectivity=self.config.connectivity
            )
            num_components = max(0, int(num_labels_cv) - 1)
            labels = labels_cv.astype(np.int32)
            raw_stats = []
            for k in range(1, num_labels_cv):
                bx, by, bw, bh, area_k = [int(v) for v in stats_cv[k, :5]]
                raw_stats.append({
                    "label": k,
                    "area": area_k,
                    "bbox": (bx, by, bw, bh),
                    "centroid": (float(cents_cv[k, 0]), float(cents_cv[k, 1])),
                    "aspect_ratio": float(bw / max(1, bh)),
                    "extent": float(area_k / max(1, bw * bh)),
                })

            filtered_stats = raw_stats
            if self.config.filter_bubbles and num_components > 0:
                _, filtered_stats = filter_bubble_components(
                    labels,
                    raw_stats,
                    min_area=self.config.min_area,
                    max_area=self.config.max_area,
                    aspect_ratio_range=self.config.aspect_ratio_range,
                    min_extent=self.config.min_extent
                )
            latencies["ccl_ms"] = (time.perf_counter() - t0) * 1000.0

        # ----------------------------------------------------------------------
        # BƯỚC 4: THIẾT LẬP LƯỚI TỌA ĐỘ ROI CÁC Ô TRÒN
        # ----------------------------------------------------------------------
        final_grid = question_grid
        if final_grid is None:
            if self.config.use_auto_grid and len(filtered_stats) >= (self.config.num_questions * len(self.config.options)):
                centroids = [item["centroid"] for item in filtered_stats]
                try:
                    final_grid = auto_align_grid_from_centroids(
                        centroids=centroids,
                        num_questions=self.config.num_questions,
                        options=self.config.options
                    )
                except Exception:
                    final_grid = None

            if final_grid is None:
                final_grid = generate_grid_bubble_rois(
                    start_x=W * 0.15,
                    start_y=H * 0.15,
                    step_x=W * 0.05,
                    step_y=H * 0.038,
                    num_questions=self.config.num_questions,
                    options=self.config.options,
                    num_columns=1 if self.config.num_questions <= 10 else 2,
                    col_spacing_x=W * 0.45,
                    col_start_y=H * 0.15
                )

        # ----------------------------------------------------------------------
        # BƯỚC 5: TÍNH TỶ LỆ LẤP ĐẦY VÀ PHÂN LOẠI ĐÁP ÁN OMR (DÙNG CHUNG LUẬT)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        omr_result = recognize_all_questions(
            binary_image=bin_morph,
            question_grid=final_grid,
            radius=self.config.effective_fill_radius,
            tau_min=self.config.tau_min,
            delta_tau_min=self.config.delta_tau_min
        )
        latencies["roi_recognition_ms"] = (time.perf_counter() - t0) * 1000.0
        latencies["total_pipeline_scratch_ms"] = (time.perf_counter() - t_start_pipeline) * 1000.0

        # ----------------------------------------------------------------------
        # BƯỚC 6: CHẠY BENCHMARK ĐỐI CHỨNG VỚI OPENCV (NẾU YÊU CẦU)
        # ----------------------------------------------------------------------
        benchmark_results = None
        if run_opencv_benchmark and mode == "scratch":
            benchmark_results = self._run_opencv_counterpart(
                img_gray=img_gray,
                bin_thresh_scratch=bin_thresh,
                bin_morph_scratch=bin_morph,
                thresh_val_scratch=thresh_val,
                ccl_count_scratch=num_components
            )

        return TV3PipelineResult(
            warped_gray=img_gray,
            binary_thresholded=bin_thresh,
            threshold_value=thresh_val,
            binary_morph=bin_morph,
            num_components=num_components,
            labels=labels,
            raw_stats=raw_stats,
            filtered_bubble_stats=filtered_stats,
            question_grid=final_grid,
            omr_result=omr_result,
            latency_ms=latencies,
            benchmark_opencv=benchmark_results,
            execution_mode=mode
        )

    def _run_opencv_counterpart(
        self,
        img_gray: np.ndarray,
        bin_thresh_scratch: np.ndarray,
        bin_morph_scratch: np.ndarray,
        thresh_val_scratch: Optional[float],
        ccl_count_scratch: int
    ) -> Dict[str, Any]:
        """Thực thi pipeline OpenCV đối chứng để so sánh độ chính xác và tốc độ."""
        import cv2

        t_cv_start = time.perf_counter()
        cv_flag = cv2.THRESH_BINARY_INV if self.config.binary_method.upper() in ("BINARY_INV", "INV") else cv2.THRESH_BINARY

        # 1. Phân ngưỡng OpenCV
        t0 = time.perf_counter()
        thresh_cv_val: Optional[float] = None
        if self.config.threshold_method.lower() == "otsu":
            ret, bin_cv_thresh = cv2.threshold(img_gray, 0, 255, cv_flag + cv2.THRESH_OTSU)
            thresh_cv_val = float(ret)
        else:
            bin_cv_thresh = cv2.adaptiveThreshold(
                img_gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv_flag,
                self.config.block_size,
                self.config.C
            )
        cv_t_thresh = (time.perf_counter() - t0) * 1000.0

        # 2. Hình thái học OpenCV
        t0 = time.perf_counter()
        kw, kh = self.config.morph_kernel_size
        cv_kshape = {
            "RECT": cv2.MORPH_RECT,
            "CROSS": cv2.MORPH_CROSS,
            "ELLIPSE": cv2.MORPH_ELLIPSE
        }.get(self.config.morph_kernel_shape.upper(), cv2.MORPH_RECT)
        kernel_cv = cv2.getStructuringElement(cv_kshape, (kw, kh))

        bin_cv_morph = bin_cv_thresh.copy()
        if self.config.apply_opening:
            bin_cv_morph = cv2.morphologyEx(bin_cv_morph, cv2.MORPH_OPEN, kernel_cv)
        if self.config.apply_closing:
            bin_cv_morph = cv2.morphologyEx(bin_cv_morph, cv2.MORPH_CLOSE, kernel_cv)
        cv_t_morph = (time.perf_counter() - t0) * 1000.0

        # 3. CCL OpenCV
        t0 = time.perf_counter()
        num_labels_cv, _, _, _ = cv2.connectedComponentsWithStats(
            bin_cv_morph,
            connectivity=self.config.connectivity
        )
        ccl_cv_count = num_labels_cv - 1
        cv_t_ccl = (time.perf_counter() - t0) * 1000.0
        cv_t_total = (time.perf_counter() - t_cv_start) * 1000.0

        # Đo độ khớp định lượng
        metrics_thresh = calculate_segmentation_metrics(bin_thresh_scratch, bin_cv_thresh)
        metrics_morph = calculate_segmentation_metrics(bin_morph_scratch, bin_cv_morph)

        delta_t = None
        if thresh_val_scratch is not None and thresh_cv_val is not None:
            delta_t = abs(thresh_val_scratch - thresh_cv_val)

        return {
            "threshold_metrics": metrics_thresh,
            "morphology_metrics": metrics_morph,
            "delta_threshold": round(delta_t, 2) if delta_t is not None else None,
            "threshold_scratch": thresh_val_scratch,
            "threshold_opencv": thresh_cv_val,
            "ccl_scratch_count": ccl_count_scratch,
            "ccl_opencv_count": ccl_cv_count,
            "ccl_count_match": (ccl_count_scratch == ccl_cv_count),
            "latency_opencv_ms": {
                "thresholding_ms": round(cv_t_thresh, 3),
                "morphology_ms": round(cv_t_morph, 3),
                "ccl_ms": round(cv_t_ccl, 3),
                "total_opencv_ms": round(cv_t_total, 3),
            },
            "acceptance_criteria": {
                "delta_otsu_pass": (delta_t is None) or (delta_t <= 1.0),
                "iou_morph_pass": metrics_morph["iou_percent"] >= 98.5,
                "ccl_match_pass": abs(ccl_count_scratch - ccl_cv_count) <= 2,
            }
        }


# ==============================================================================
# 4. HÀM GIAO DIỆN HỢP NHẤT TOÀN CỤC run_tv3_pipeline(...)
# ==============================================================================

def run_tv3_pipeline(
    warped_gray: np.ndarray,
    threshold_method: str = "adaptive",
    block_size: int = 25,
    C: float = 10.0,
    bubble_radius: float = DEFAULT_BUBBLE_RADIUS,
    tau_min: float = DEFAULT_TAU_MIN,
    delta_tau_min: float = DEFAULT_DELTA_TAU_MIN,
    num_questions: int = 20,
    question_grid: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
    run_opencv_benchmark: bool = True
) -> TV3PipelineResult:
    """
    Hàm top-level thực thi trọn gói pipeline của Thành viên 3 một cách tiện lợi.

    Tham số:
        warped_gray: Ảnh xám phiếu thi OMR đã nắn phẳng.
        threshold_method: 'adaptive' hoặc 'otsu'.
        block_size: Kích thước cửa sổ b x b cho Adaptive Mean.
        C: Hằng số C của Adaptive Mean.
        bubble_radius: Bán kính ô tròn trắc nghiệm.
        tau_min: Ngưỡng lấp đầy ô tô hợp lệ.
        delta_tau_min: Ngưỡng phân định vết tẩy xóa.
        num_questions: Số lượng câu hỏi (mặc định 20).
        question_grid: Lưới tọa độ tâm các ô (hoặc None để tự động tạo).
        run_opencv_benchmark: Có đo đối chứng với OpenCV không.

    Trả về:
        TV3PipelineResult: Kết quả phân đoạn và nhận diện.
    """
    cfg = TV3PipelineConfig(
        threshold_method=threshold_method,
        block_size=block_size,
        C=C,
        bubble_radius=bubble_radius,
        tau_min=tau_min,
        delta_tau_min=delta_tau_min,
        num_questions=num_questions
    )
    pipeline = TV3SegmentationPipeline(cfg)
    return pipeline.run(
        warped_gray=warped_gray,
        question_grid=question_grid,
        run_opencv_benchmark=run_opencv_benchmark
    )


def detect_bubble_grid_scratch(
    warped_gray: np.ndarray,
    num_questions: int = 20,
    options: Sequence[str] = DEFAULT_OPTIONS,
    config: Optional[TV3PipelineConfig] = None,
    use_scratch: bool = True,
) -> Tuple[Dict[int, Dict[str, Tuple[float, float]]], List[Dict[str, Any]]]:
    """
    Dò lưới các ô tròn trắc nghiệm hoàn toàn bằng thuật toán TV3 (Adaptive Mean + Morphology + Two-Pass CCL),
    thay thế cv2.adaptiveThreshold + cv2.findContours trong bước dò lưới của TV4 (Mục 1.1).

    Trả về:
        (question_grid, filtered_bubble_stats)
    """
    cfg = config or TV3PipelineConfig(num_questions=num_questions, options=options)
    img_gray = np.clip(warped_gray, 0, 255).astype(np.uint8)

    if use_scratch:
        bin_thresh = adaptive_mean_threshold_scratch(
            img_gray,
            block_size=cfg.block_size,
            C=cfg.C,
            method=cfg.binary_method,
            max_val=255,
            border_type="replicate",
        )
        se = create_structuring_element(cfg.morph_kernel_shape, cfg.morph_kernel_size)
        bin_morph = opening_binary_scratch(bin_thresh, se) if cfg.apply_opening else bin_thresh
        if cfg.apply_closing:
            bin_morph = closing_binary_scratch(bin_morph, se)
        num_comp, labels, raw_stats = two_pass_ccl_scratch(bin_morph, connectivity=cfg.connectivity)
    else:
        import cv2
        cv_flag = cv2.THRESH_BINARY_INV if cfg.binary_method.upper() in ("BINARY_INV", "INV") else cv2.THRESH_BINARY
        bin_thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv_flag, cfg.block_size, cfg.C
        )
        kh, kw = cfg.morph_kernel_size
        cv_kshape = {"RECT": cv2.MORPH_RECT, "CROSS": cv2.MORPH_CROSS, "ELLIPSE": cv2.MORPH_ELLIPSE}.get(
            cfg.morph_kernel_shape.upper(), cv2.MORPH_RECT
        )
        kernel_cv = cv2.getStructuringElement(cv_kshape, (kw, kh))
        bin_morph = cv2.morphologyEx(bin_thresh, cv2.MORPH_OPEN, kernel_cv) if cfg.apply_opening else bin_thresh
        if cfg.apply_closing:
            bin_morph = cv2.morphologyEx(bin_morph, cv2.MORPH_CLOSE, kernel_cv)
        num_labels_cv, labels_cv, stats_cv, cents_cv = cv2.connectedComponentsWithStats(
            bin_morph, connectivity=cfg.connectivity
        )
        num_comp = max(0, int(num_labels_cv) - 1)
        labels = labels_cv.astype(np.int32)
        raw_stats = [
            {
                "label": k,
                "area": int(stats_cv[k, 4]),
                "bbox": tuple(int(v) for v in stats_cv[k, :4]),
                "centroid": (float(cents_cv[k, 0]), float(cents_cv[k, 1])),
                "aspect_ratio": float(stats_cv[k, 2] / max(1, int(stats_cv[k, 3]))),
                "extent": float(stats_cv[k, 4] / max(1, int(stats_cv[k, 2]) * int(stats_cv[k, 3]))),
            }
            for k in range(1, num_labels_cv)
        ]

    filtered_stats = raw_stats
    if cfg.filter_bubbles and num_comp > 0:
        _, filtered_stats = filter_bubble_components(
            labels,
            raw_stats,
            min_area=cfg.min_area,
            max_area=cfg.max_area,
            aspect_ratio_range=cfg.aspect_ratio_range,
            min_extent=cfg.min_extent,
        )

    centroids = [item["centroid"] for item in filtered_stats]
    grid = auto_align_grid_from_centroids(
        centroids=centroids,
        num_questions=num_questions,
        options=options,
    )
    return grid, filtered_stats


def extract_bubble_answers(
    warped_gray: np.ndarray,
    total_questions: int = 20,
    fill_threshold: float = DEFAULT_TAU_MIN,
    use_scratch: bool = True,
    question_grid: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
    config: Optional[TV3PipelineConfig] = None,
    return_details: bool = False,
) -> Union[
    Tuple[Dict[int, str], Dict[int, Dict[str, float]], np.ndarray],
    Tuple[Dict[int, str], Dict[int, Dict[str, float]], np.ndarray, Dict[str, Any]],
]:
    """
    Giao tiếp tích hợp chuẩn (Integration Interface) với Pipeline Coordinator của TV4.
    Cả hai chế độ use_scratch=True và use_scratch=False dùng chung 100% cấu hình thuật toán
    (block_size, C, morphology, inner_radius R - 3, tau_min, delta_tau_min).

    Tham số:
        warped_gray: Ảnh xám phiếu thi OMR (2D uint8) đã được TV2 nắn thẳng phối cảnh.
        total_questions: Tổng số câu hỏi (mặc định 20 câu).
        fill_threshold: Ngưỡng tỷ lệ lấp đầy tau để công nhận ô tô.
        use_scratch: True -> chạy 100% Scratch NumPy; False -> chạy OpenCV cùng tham số đối chứng.
        question_grid: Lưới tọa độ tâm ô tùy chọn. Nếu None, dùng lưới mặc định theo kích thước ảnh.
        config: Cấu hình dùng chung (TV3PipelineConfig). Nếu None, khởi tạo cấu hình chuẩn đồng nhất.
        return_details: Nếu True, trả thêm dict details chứa statuses, warnings (gồm ERASED_VALID) và omr_result.

    Trả về:
        Tuple:
            - student_answers: Dict[int, str] kết quả nhận diện (ví dụ {1: 'A', 2: 'B', ..., 5: 'BLANK', 15: 'MULTIPLE'})
            - fill_ratios_all: Dict[int, Dict[str, float]] tỷ lệ lấp đầy từng lựa chọn A/B/C/D
            - vis_warped: Ảnh BGR hiển thị kết quả khoanh tròn và chú thích trực quan
            - details (khi return_details=True): Dict chứa statuses, warnings, erased_questions, mode, omr_result
    """
    h, w = warped_gray.shape[:2]
    bubble_radius = int(min(w, h) * 0.018)

    if question_grid is not None:
        custom_grid = question_grid
    else:
        questions_per_col = (total_questions + 1) // 2
        start_y, end_y = int(h * 0.16), int(h * 0.89)
        col1_x, col2_x = int(w * 0.13), int(w * 0.57)
        row_height = (end_y - start_y) / float(questions_per_col)

        options = ["A", "B", "C", "D"]
        custom_grid = {}
        for q in range(1, total_questions + 1):
            row_idx = (q - 1) if q <= questions_per_col else (q - 1 - questions_per_col)
            base_x = col1_x if q <= questions_per_col else col2_x
            center_y = float(start_y + (row_idx + 0.5) * row_height)
            q_grid: Dict[str, Tuple[float, float]] = {}
            for opt_idx, opt in enumerate(options):
                center_x = float(base_x + (opt_idx + 1) * (bubble_radius * 2.8))
                q_grid[opt] = (center_x, center_y)
            custom_grid[q] = q_grid

    mode = "scratch" if use_scratch else "opencv"
    if config is not None:
        shared_cfg = TV3PipelineConfig(
            threshold_method=config.threshold_method,
            block_size=config.block_size,
            C=config.C,
            binary_method=config.binary_method,
            apply_opening=config.apply_opening,
            apply_closing=config.apply_closing,
            morph_kernel_shape=config.morph_kernel_shape,
            morph_kernel_size=config.morph_kernel_size,
            connectivity=config.connectivity,
            filter_bubbles=config.filter_bubbles,
            min_area=config.min_area,
            max_area=config.max_area,
            aspect_ratio_range=config.aspect_ratio_range,
            min_extent=config.min_extent,
            num_questions=total_questions,
            options=config.options,
            bubble_radius=config.bubble_radius if config.bubble_radius else float(bubble_radius),
            inner_radius_offset=config.inner_radius_offset,
            tau_min=fill_threshold,
            delta_tau_min=config.delta_tau_min,
            use_auto_grid=config.use_auto_grid,
            execution_mode=mode,
        )
    else:
        shared_cfg = TV3PipelineConfig(
            num_questions=total_questions,
            tau_min=fill_threshold,
            delta_tau_min=DEFAULT_DELTA_TAU_MIN,
            bubble_radius=float(bubble_radius),
            inner_radius_offset=3.0,
            threshold_method="adaptive",
            execution_mode=mode,
        )

    pipeline = TV3SegmentationPipeline(shared_cfg)
    res = pipeline.run(warped_gray, question_grid=custom_grid, run_opencv_benchmark=False, execution_mode=mode)

    student_answers: Dict[int, str] = {}
    fill_ratios_all: Dict[int, Dict[str, float]] = {}
    statuses: Dict[int, str] = {}
    warnings: Dict[int, str] = {}
    erased_questions: List[int] = []

    for q_id, q_res in res.omr_result.questions.items():
        fill_ratios_all[q_id] = {k: round(v, 4) for k, v in q_res.fill_ratios.items()}
        statuses[q_id] = q_res.status
        if q_res.warning:
            warnings[q_id] = q_res.warning
        if q_res.status == STATUS_ERASED_VALID:
            erased_questions.append(q_id)

        if q_res.status in (STATUS_VALID, STATUS_ERASED_VALID) and q_res.selected:
            student_answers[q_id] = q_res.selected
        elif q_res.status == STATUS_MULTIPLE_INVALID:
            student_answers[q_id] = "MULTIPLE"
        else:
            student_answers[q_id] = "BLANK"

    inner_r = int(round(shared_cfg.effective_fill_radius))
    try:
        import cv2
        vis_warped = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR) if warped_gray.ndim == 2 else warped_gray.copy()
        for q_id, q_res in res.omr_result.questions.items():
            ans = student_answers.get(q_id, "BLANK")
            for opt, (cx, cy) in custom_grid[q_id].items():
                icx, icy = int(round(cx)), int(round(cy))
                if ans == opt:
                    # Cam cho ERASED_VALID để giám thị nhận biết cảnh báo tẩy xóa, xanh lá cho VALID
                    color = (0, 140, 255) if q_res.status == STATUS_ERASED_VALID else (0, 200, 0)
                    cv2.circle(vis_warped, (icx, icy), inner_r + 2, color, 2, cv2.LINE_AA)
                elif ans == "MULTIPLE" and q_res.fill_ratios.get(opt, 0.0) >= fill_threshold:
                    cv2.circle(vis_warped, (icx, icy), inner_r + 2, (0, 0, 255), 2, cv2.LINE_AA)
    except ImportError:
        vis_warped = np.stack([warped_gray, warped_gray, warped_gray], axis=-1) if warped_gray.ndim == 2 else warped_gray.copy()

    if return_details:
        details = {
            "mode": mode,
            "statuses": statuses,
            "warnings": warnings,
            "erased_questions": erased_questions,
            "effective_fill_radius": shared_cfg.effective_fill_radius,
            "omr_result": res.omr_result,
        }
        return student_answers, fill_ratios_all, vis_warped, details

    return student_answers, fill_ratios_all, vis_warped


def extract_bubble_answers_with_warnings(
    warped_gray: np.ndarray,
    total_questions: int = 20,
    fill_threshold: float = DEFAULT_TAU_MIN,
    use_scratch: bool = True,
    question_grid: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
    config: Optional[TV3PipelineConfig] = None,
) -> Tuple[Dict[int, str], Dict[int, Dict[str, float]], np.ndarray, Dict[str, Any]]:
    """Phiên bản trả về đầy đủ cảnh báo (ERASED_VALID) và trạng thái câu hỏi cho Tab 4 của TV4."""
    return extract_bubble_answers(
        warped_gray=warped_gray,
        total_questions=total_questions,
        fill_threshold=fill_threshold,
        use_scratch=use_scratch,
        question_grid=question_grid,
        config=config,
        return_details=True,
    )


# ==============================================================================
# 5. BỘ SINH DỮ LIỆU PHIẾU OMR MÔ PHỎNG THỰC TẾ (SYNTHETIC GENERATOR)
# ==============================================================================

def generate_synthetic_omr_sheet(
    width: int = 800,
    height: int = 1000,
    num_questions: int = 20,
    options: Sequence[str] = DEFAULT_OPTIONS,
    add_shadow: bool = True,
    add_noise: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[int, Optional[str]], Dict[int, str], Dict[int, Dict[str, Tuple[float, float]]]]:
    """
    Tạo ảnh phiếu thi OMR 20 câu mô phỏng thực tế đa kịch bản (DS-01, DS-03, DS-04, DS-05):
    - Đầy đủ 4 Marker góc hình vuông màu đen (32x32 px) phục vụ TV2 và TV4.
    - Khung viền bảng câu hỏi chia 2 cột (Cột 1: Câu 1-10; Cột 2: Câu 11-20).
    - Các kịch bản câu hỏi:
      + Tô chuẩn (VALID): Tô đen đậm bút chì 2B.
      + Bỏ trống (BLANK): Ô tròn rỗng nguyên bản.
      + Tẩy xóa chọn lại (ERASED_VALID): 1 ô tẩy mờ (tau ~0.45) và 1 ô tô đậm (tau ~0.85).
      + Tô đúp phạm quy (MULTIPLE_INVALID): 2 ô tô đậm gần bằng nhau.
    - Thêm gradient bóng đổ chéo góc mô phỏng tập DS-03 (mức xám giảm cục bộ 90-120 mức).
    - Thêm nhiễu muối tiêu hoặc nhiễu Gaussian nhẹ.

    Trả về:
        Tuple:
            - sheet_gray: Ảnh mức xám phiếu thi (H, W) uint8.
            - ground_truth_answers: Dict câu hỏi -> đáp án đúng {1: 'A', 2: None, ...}
            - ground_truth_statuses: Dict câu hỏi -> trạng thái {'VALID', 'BLANK', ...}
            - grid_rois: Lưới tọa độ tâm hình tròn chuẩn {q_id: {opt: (cx, cy)}}
    """
    np.random.seed(seed)
    import cv2

    # 1. Nền giấy trắng tự nhiên (mức xám ~238-245)
    sheet = np.full((height, width), 242, dtype=np.uint8)

    # 2. Vẽ 4 Marker định vị ở 4 góc (32x32 px)
    marker_size = 32
    margin = 35
    # Top-Left, Top-Right, Bottom-Left, Bottom-Right
    sheet[margin:margin+marker_size, margin:margin+marker_size] = 20
    sheet[margin:margin+marker_size, width-margin-marker_size:width-margin] = 20
    sheet[height-margin-marker_size:height-margin, margin:margin+marker_size] = 20
    sheet[height-margin-marker_size:height-margin, width-margin-marker_size:width-margin] = 20

    # 3. Tiêu đề phiếu thi OMR
    cv2.putText(sheet, "KHAO THI OMR - PHIEU TRA LOI TRAC NGHIEM", (width // 2 - 250, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, 30, 2, cv2.LINE_AA)
    cv2.putText(sheet, "Nhom 4 - INT13146: Tu dong cham bai thi", (width // 2 - 170, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 60, 1, cv2.LINE_AA)

    # 4. Thiết kế lưới 20 câu hỏi chia làm 2 cột: Cột 1 (Q1-10), Cột 2 (Q11-20)
    col1_start_x = 110.0
    col2_start_x = 470.0
    start_y = 170.0
    step_y = 70.0
    step_x = 55.0
    bubble_radius = DEFAULT_BUBBLE_RADIUS

    grid_rois: Dict[int, Dict[str, Tuple[float, float]]] = {}
    gt_answers: Dict[int, Optional[str]] = {}
    gt_statuses: Dict[int, str] = {}

    # Kịch bản đáp án giả lập cho 20 câu hỏi:
    # Q1: A (VALID), Q2: B (VALID), Q3: C (VALID), Q4: D (VALID)
    # Q5: BLANK (Bỏ trống)
    # Q6: B (VALID), Q7: A (VALID)
    # Q8: C (ERASED_VALID - Tẩy B còn vết mờ ~0.42, tô C đậm ~0.85)
    # Q9: D (VALID), Q10: A (VALID)
    # Q11: B (VALID), Q12: BLANK (Bỏ trống)
    # Q13: C (VALID), Q14: D (VALID)
    # Q15: MULTIPLE_INVALID (Tô đúp A và B cùng đậm)
    # Q16: B (VALID), Q17: C (ERASED_VALID - Tẩy A còn vết ~0.40, tô C đậm)
    # Q18: D (VALID), Q19: A (VALID), Q20: B (VALID)
    scenarios: Dict[int, Dict[str, Any]] = {
        1: {"type": STATUS_VALID, "ans": "A"},
        2: {"type": STATUS_VALID, "ans": "B"},
        3: {"type": STATUS_VALID, "ans": "C"},
        4: {"type": STATUS_VALID, "ans": "D"},
        5: {"type": STATUS_BLANK, "ans": None},
        6: {"type": STATUS_VALID, "ans": "B"},
        7: {"type": STATUS_VALID, "ans": "A"},
        8: {"type": STATUS_ERASED_VALID, "ans": "C", "erased": "B"},
        9: {"type": STATUS_VALID, "ans": "D"},
        10: {"type": STATUS_VALID, "ans": "A"},
        11: {"type": STATUS_VALID, "ans": "B"},
        12: {"type": STATUS_BLANK, "ans": None},
        13: {"type": STATUS_VALID, "ans": "C"},
        14: {"type": STATUS_VALID, "ans": "D"},
        15: {"type": STATUS_MULTIPLE_INVALID, "ans": None, "marked": ["A", "B"]},
        16: {"type": STATUS_VALID, "ans": "B"},
        17: {"type": STATUS_ERASED_VALID, "ans": "C", "erased": "A"},
        18: {"type": STATUS_VALID, "ans": "D"},
        19: {"type": STATUS_VALID, "ans": "A"},
        20: {"type": STATUS_VALID, "ans": "B"},
    }

    # Vẽ từng câu hỏi lên tờ phiếu
    for q_idx in range(1, num_questions + 1):
        is_col2 = q_idx > 10
        row_idx = (q_idx - 1) % 10

        base_x = col2_start_x if is_col2 else col1_start_x
        base_y = start_y + row_idx * step_y

        # Nhãn số thứ tự câu hỏi
        label_text = f"Q{q_idx:02d}:"
        cv2.putText(sheet, label_text, (int(base_x - 55), int(base_y + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 40, 1, cv2.LINE_AA)

        q_rois: Dict[str, Tuple[float, float]] = {}
        sc_info = scenarios.get(q_idx, {"type": STATUS_VALID, "ans": "A"})
        q_type = sc_info["type"]
        q_ans = sc_info["ans"]

        gt_answers[q_idx] = q_ans
        gt_statuses[q_idx] = q_type

        for opt_idx, opt_name in enumerate(options):
            cx = base_x + opt_idx * step_x
            cy = base_y
            q_rois[opt_name] = (float(cx), float(cy))

            icx, icy = int(round(cx)), int(round(cy))
            ir = int(round(bubble_radius))

            # 1. Vẽ đường viền vòng tròn ô đáp án in sẵn (màu xám 70, độ dày 1)
            cv2.circle(sheet, (icx, icy), ir, 70, 1, cv2.LINE_AA)

            # 2. In chữ cái A, B, C, D mờ bên trong ô
            cv2.putText(sheet, opt_name, (icx - 4, icy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, 100, 1, cv2.LINE_AA)

            # 3. Thực hiện tô chì tùy theo kịch bản:
            if q_type == STATUS_VALID and opt_name == q_ans:
                # Tô chuẩn bằng bút chì 2B đậm (mức xám 30-45, bán kính đầy đủ)
                cv2.circle(sheet, (icx, icy), ir - 1, 35, -1, cv2.LINE_AA)

            elif q_type == STATUS_ERASED_VALID:
                if opt_name == q_ans:
                    # Ô tô thật: đậm hoàn toàn
                    cv2.circle(sheet, (icx, icy), ir - 1, 35, -1, cv2.LINE_AA)
                elif opt_name == sc_info.get("erased"):
                    # Ô bị tẩy xóa: vết chì tẩy dở còn sót lại (bán kính ~0.7*r, fill ratio ~0.45 - 0.55)
                    cv2.circle(sheet, (icx, icy), int(round(ir * 0.70)), 60, -1, cv2.LINE_AA)

            elif q_type == STATUS_MULTIPLE_INVALID:
                if opt_name in sc_info.get("marked", []):
                    # Tô đậm cả 2 ô
                    cv2.circle(sheet, (icx, icy), ir - 1, 38, -1, cv2.LINE_AA)

        grid_rois[q_idx] = q_rois

    # 5. Thêm bóng đổ không đều (DS-03 Simulation) nếu được kích hoạt
    if add_shadow:
        Y, X = np.ogrid[:height, :width]
        # Bóng đổ dạng dải chéo từ góc dưới-phải tràn sang, giảm cường độ từ 110 về 0
        shadow = ((0.4 * (X / width) + 0.6 * (Y / height)) * 105.0).astype(np.float64)
        # Giới hạn mức trừ để không làm âm ảnh
        sheet_float = sheet.astype(np.float64) - shadow
        sheet = np.clip(sheet_float, 20, 255).astype(np.uint8)

    # 6. Thêm nhiễu nhẹ
    if add_noise:
        noise = np.random.normal(0, 2.5, (height, width))
        sheet = np.clip(sheet.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return sheet, gt_answers, gt_statuses, grid_rois


# ==============================================================================
# 6. TRỰC QUAN HÓA CÁC BƯỚC TRUNG GIAN & VẼ LỚP PHỦ KẾT QUẢ (VISUALIZATION)
# ==============================================================================

def visualize_pipeline_steps(
    result: TV3PipelineResult,
    output_dir: str,
    ground_truth_answers: Optional[Dict[int, Optional[str]]] = None
) -> Dict[str, str]:
    """
    Tạo và lưu trữ toàn bộ ảnh trực quan hóa các bước xử lý trung gian vào thư mục output_dir:
        1. 1_input_warped.png: Ảnh gốc đầu vào.
        2. 2_thresholded.png: Ảnh sau phân ngưỡng nhị phân.
        3. 3_morphology.png: Ảnh sau xử lý hình thái học (Opening + Closing).
        4. 4_ccl_labels.png: Bản đồ màu phân biệt từng vùng liên thông CCL.
        5. 5_omr_annotated.png: Ảnh phiếu thi kèm vòng tròn màu và nhãn trạng thái trực quan.
        6. 6_triplet_comparison.png: Triplet View (Scratch vs OpenCV vs Difference Heatmap).

    Trả về:
        Dict[str, str]: Bảng ánh xạ tên ảnh -> đường dẫn file tuyệt đối đã lưu.
    """
    os.makedirs(output_dir, exist_ok=True)
    import cv2

    saved_files: Dict[str, str] = {}
    H, W = result.warped_gray.shape

    # 1. Lưu ảnh gốc
    p1 = os.path.join(output_dir, "1_input_warped.png")
    cv2.imwrite(p1, result.warped_gray)
    saved_files["input_warped"] = os.path.abspath(p1)

    # 2. Lưu ảnh phân ngưỡng
    p2 = os.path.join(output_dir, "2_thresholded.png")
    cv2.imwrite(p2, result.binary_thresholded)
    saved_files["thresholded"] = os.path.abspath(p2)

    # 3. Lưu ảnh Morphology
    p3 = os.path.join(output_dir, "3_morphology.png")
    cv2.imwrite(p3, result.binary_morph)
    saved_files["morphology"] = os.path.abspath(p3)

    # 4. Lưu ảnh CCL màu (mỗi label một màu ngẫu nhiên)
    p4 = os.path.join(output_dir, "4_ccl_labels.png")
    np.random.seed(123)
    num_comp = result.num_components
    # Tạo bảng màu ngẫu nhiên cho K thành phần (Label 0 là nền đen)
    color_map = np.random.randint(50, 255, size=(num_comp + 1, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # Nền đen
    colored_ccl = color_map[np.clip(result.labels, 0, num_comp)]
    cv2.imwrite(p4, cv2.cvtColor(colored_ccl, cv2.COLOR_RGB2BGR))
    saved_files["ccl_labels"] = os.path.abspath(p4)

    # 5. Lưu ảnh phiếu thi vẽ lớp phủ nhận diện (Overlay Annotation)
    p5 = os.path.join(output_dir, "5_omr_annotated.png")
    annotated = cv2.cvtColor(result.warped_gray, cv2.COLOR_GRAY2BGR)

    radius = int(round(DEFAULT_BUBBLE_RADIUS))
    for q_id, q_res in result.omr_result.questions.items():
        opt_coords = result.question_grid.get(q_id, {})
        status = q_res.status
        selected = q_res.selected

        for opt_name, center in opt_coords.items():
            cx, cy = int(round(center[0])), int(round(center[1]))
            tau_val = q_res.fill_ratios.get(opt_name, 0.0)

            # Màu sắc viền theo trạng thái
            if opt_name == selected:
                if status == STATUS_VALID:
                    color = (0, 180, 0)      # Xanh lá: Tô chuẩn hợp lệ
                    thickness = 2
                elif status == STATUS_ERASED_VALID:
                    color = (0, 140, 255)    # Cam: Tẩy xóa chọn lại
                    thickness = 2
                else:
                    color = (0, 0, 255)      # Đỏ
                    thickness = 2
                cv2.circle(annotated, (cx, cy), radius + 2, color, thickness, cv2.LINE_AA)
            elif status == STATUS_MULTIPLE_INVALID and tau_val >= DEFAULT_TAU_MIN:
                # Khoanh đỏ các ô tô tranh chấp
                cv2.circle(annotated, (cx, cy), radius + 2, (0, 0, 255), 2, cv2.LINE_AA)
            elif status == STATUS_BLANK:
                # Nét đứt màu xám cho câu bỏ trống
                pass

        # Hiển thị text kết quả cạnh câu hỏi
        if "A" in opt_coords:
            first_x = opt_coords["A"][0]
            first_y = opt_coords["A"][1]
            status_text = f"{selected or 'None'} ({status[:4]})"
            text_color = (0, 160, 0) if q_res.is_valid else ((0, 0, 255) if status == STATUS_MULTIPLE_INVALID else (100, 100, 100))
            cv2.putText(annotated, status_text, (int(first_x + 185), int(first_y + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)

    cv2.imwrite(p5, annotated)
    saved_files["omr_annotated"] = os.path.abspath(p5)

    # 6. Tạo Triplet Comparison View (Scratch vs OpenCV vs Difference Heatmap)
    p6 = os.path.join(output_dir, "6_triplet_comparison.png")
    # Sử dụng ảnh morphology để làm đối chứng chuẩn
    img_scratch = result.binary_morph

    # Tạo ảnh OpenCV đối chứng
    cv_flag = cv2.THRESH_BINARY_INV
    block_s = 25
    c_val = 10.0
    bin_cv_thresh = cv2.adaptiveThreshold(result.warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv_flag, block_s, c_val)
    kernel_cv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_cv_morph = cv2.morphologyEx(bin_cv_thresh, cv2.MORPH_OPEN, kernel_cv)
    bin_cv_morph = cv2.morphologyEx(bin_cv_morph, cv2.MORPH_CLOSE, kernel_cv)

    # Bản đồ nhiệt sai khác tuyệt đối khuếch đại x10
    diff = cv2.absdiff(img_scratch, bin_cv_morph)
    diff_boosted = np.clip(diff.astype(np.float64) * 10.0, 0, 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_boosted, cv2.COLORMAP_JET)

    # Chuyển về 3 kênh màu và ghép ngang
    scratch_bgr = cv2.cvtColor(img_scratch, cv2.COLOR_GRAY2BGR)
    cv_bgr = cv2.cvtColor(bin_cv_morph, cv2.COLOR_GRAY2BGR)

    # Thêm tiêu đề vào từng tấm ảnh
    header_h = 40
    panel_w = W
    combined_w = panel_w * 3
    triplet = np.zeros((H + header_h, combined_w, 3), dtype=np.uint8)

    # Vẽ nền header xám đậm
    triplet[:header_h, :] = (40, 40, 40)
    cv2.putText(triplet, "1. Code from Scratch (NumPy)", (20, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(triplet, "2. OpenCV Reference (cv2)", (panel_w + 20, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(triplet, "3. Difference Heatmap (|Scratch - OpenCV| x 10)", (panel_w * 2 + 20, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    triplet[header_h:, 0:panel_w] = scratch_bgr
    triplet[header_h:, panel_w:panel_w*2] = cv_bgr
    triplet[header_h:, panel_w*2:panel_w*3] = diff_heatmap

    cv2.imwrite(p6, triplet)
    saved_files["triplet_comparison"] = os.path.abspath(p6)

    return saved_files


# ==============================================================================
# 7. KHỐI THỰC THI SCRIPT ĐỘC LẬP (STANDALONE DEMO & BENCHMARK SUITE)
# ==============================================================================

def main():
    print("=" * 80)
    print("  HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG (PTIT) - KHOA CNTT 1")
    print("  DỰ ÁN XỬ LÝ ẢNH (INT13146) - HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM OMR")
    print("  BÁO CÁO THỰC NGHIỆM GIAI ĐOẠN 3: TV3_SEGMENTATION (THÀNH VIÊN 3)")
    print("=" * 80)

    # Đường dẫn xuất sản phẩm
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "tv3"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Thư mục lưu sản phẩm: {output_dir}\n")

    # 1. Khởi tạo ảnh phiếu OMR thực nghiệm (có bóng đổ DS-03, tẩy xóa, tô đúp)
    print("[1/5] Đang sinh ảnh phiếu thi OMR mô phỏng 20 câu hỏi thực nghiệm...")
    sheet_gray, gt_answers, gt_statuses, grid_rois = generate_synthetic_omr_sheet(
        width=800,
        height=1000,
        num_questions=20,
        add_shadow=True,
        add_noise=True,
        seed=42
    )
    print("      -> Sinh phiếu thành công kích thước: 800 x 1000 px.")
    print("      -> Mô phỏng: 15 câu Tô chuẩn, 2 câu Bỏ trống, 2 câu Tẩy xóa, 1 câu Tô đúp.")

    # 2. Khởi chạy Pipeline tích hợp Scratch
    print("\n[2/5] Đang thực thi TV3SegmentationPipeline (100% Scratch NumPy)...")
    pipeline = TV3SegmentationPipeline(TV3PipelineConfig(
        threshold_method="adaptive",
        block_size=25,
        C=10.0,
        bubble_radius=12.0,
        tau_min=0.38,
        delta_tau_min=0.15,
        num_questions=20
    ))
    res = pipeline.run(
        warped_gray=sheet_gray,
        question_grid=grid_rois,
        run_opencv_benchmark=True
    )

    # 3. Trực quan hóa và lưu toàn bộ ảnh trung gian
    print("\n[3/5] Đang xuất toàn bộ ảnh trung gian và bản đồ nhiệt Triplet View...")
    saved_images = visualize_pipeline_steps(res, output_dir, gt_answers)
    for name, path in saved_images.items():
        print(f"      + Đã lưu: {os.path.basename(path)}")

    # 4. Xuất file dữ liệu CSV & JSON
    print("\n[4/5] Đang kết xuất bảng dữ liệu nhận diện và benchmark...")
    csv_path = os.path.join(output_dir, "recognition_results.csv")
    json_path = os.path.join(output_dir, "recognition_results.json")
    res.save_csv(csv_path)
    res.save_json(json_path)

    # Lưu metrics benchmark
    bench = res.benchmark_opencv
    if bench:
        bench_csv = os.path.join(output_dir, "benchmark_metrics.csv")
        with open(bench_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Giai_doan", "MSE", "PSNR_dB", "IoU_percent", "Dice_percent", "Latency_Scratch_ms", "Latency_OpenCV_ms"])
            mt = bench["threshold_metrics"]
            mm = bench["morphology_metrics"]
            lat_sc = res.latency_ms
            lat_cv = bench["latency_opencv_ms"]
            w.writerow(["Thresholding", mt["mse"], mt["psnr_db"], mt["iou_percent"], mt["dice_percent"], lat_sc["thresholding_ms"], lat_cv["thresholding_ms"]])
            w.writerow(["Morphology", mm["mse"], mm["psnr_db"], mm["iou_percent"], mm["dice_percent"], lat_sc["morphology_ms"], lat_cv["morphology_ms"]])
            w.writerow(["Two_Pass_CCL", "-", "-", "-", "-", lat_sc["ccl_ms"], lat_cv["ccl_ms"]])
            w.writerow(["Total_Pipeline", "-", "-", "-", "-", lat_sc["total_pipeline_scratch_ms"], lat_cv["total_opencv_ms"]])

        bench_json = os.path.join(output_dir, "benchmark_metrics.json")
        with open(bench_json, "w", encoding="utf-8") as f:
            json.dump(bench, f, indent=4, ensure_ascii=False)
        print("      + Đã xuất: benchmark_metrics.csv và benchmark_metrics.json")

    # 5. So khớp độ chính xác và in bảng tổng hợp
    print("\n[5/5] TỔNG HỢP KẾT QUẢ ĐỐI CHỨNG VÀ NGHIỆM THU:")
    print("-" * 80)
    print(f" {'Q#':<4} | {'Dap An Nhan Dien':<18} | {'Ground Truth':<14} | {'Trang Thai':<18} | {'Ket Qua':<8}")
    print("-" * 80)

    correct_count = 0
    ans_vec = res.get_answer_vector()
    for q_id in range(1, 21):
        pred = ans_vec.get(q_id)
        gt = gt_answers.get(q_id)
        status = res.omr_result.questions[q_id].status
        is_match = (pred == gt)
        if is_match:
            correct_count += 1
        res_str = "MATCH" if is_match else "MISMATCH"
        print(f" Q{q_id:02d} | {str(pred):<18} | {str(gt):<14} | {status:<18} | {res_str}")

    accuracy = (correct_count / 20) * 100.0
    print("-" * 80)
    print(f" -> DO CHINH XAC NHAN DIEN DAP AN: {accuracy:.2f}% ({correct_count}/20 cau)")
    print(f" -> THONG KE TRANG THAI: {res.get_summary()}")

    if bench:
        print("\n--- CHỈ SỐ ĐỊNH LƯỢNG ĐỐI CHỨNG VỚI OPENCV ---")
        mt = bench["threshold_metrics"]
        mm = bench["morphology_metrics"]
        print(f" [Thresholding] IoU: {mt['iou_percent']}% | Dice: {mt['dice_percent']}% | PSNR: {mt['psnr_db']} dB")
        print(f" [Morphology]   IoU: {mm['iou_percent']}% | Dice: {mm['dice_percent']}% | PSNR: {mm['psnr_db']} dB")
        print(f" [CCL Count]    Scratch: {bench['ccl_scratch_count']} | OpenCV: {bench['ccl_opencv_count']} | Match: {bench['ccl_count_match']}")
        print(f" [Thời gian]    Scratch Total: {res.latency_ms['total_pipeline_scratch_ms']:.2f} ms | OpenCV: {bench['latency_opencv_ms']['total_opencv_ms']:.2f} ms")
        print(f" [Nghiệm thu]   IoU >= 98.5%: {'DAT' if bench['acceptance_criteria']['iou_morph_pass'] else 'KHONG DAT'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
