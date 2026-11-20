"""Module TV4 - Metrics: Quantitative Benchmarks, Error Analytics & Ablation Studies
Thành viên 4: Bộ đo lường định lượng, Kiểm định toán học & Khảo sát độ nhạy tham số

Các phép đo định lượng & Kiểm chuẩn:
1. Độ tương đồng tín hiệu ảnh số:
   - MSE (Mean Squared Error)
   - PSNR (Peak Signal-to-Noise Ratio)
   - Difference Map (Bản đồ sai khác nhân hệ số khuếch đại)
2. Độ chính xác mặt nạ nhị phân:
   - IoU (Intersection over Union)
   - Dice Similarity Coefficient
3. Chỉ số kiểm định OMR:
   - Độ lệch ngưỡng Otsu Delta T = |T*_scratch - T*_opencv| (Mục tiêu <= 1 mức xám)
   - Luật phân định vết tẩy xóa Delta_tau = tau_(1) - tau_(2) >= 0.15
   - Sai số định vị 4 góc phiếu Euclidean Corner Error
   - Độ chính xác nhận diện tô đáp án (Mark Detection Accuracy)
4. Đánh giá toàn diện Pipeline:
   - benchmark_all_modules: Kiểm thử định lượng liên phân hệ TV1 -> TV4
   - run_ablation_study: Khảo sát độ nhạy phân đoạn (Otsu vs Adaptive) và ngưỡng tô tau_min
"""

from __future__ import annotations
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np


# ==============================================================================
# 1. PHÉP ĐO SAI SỐ TÍN HIỆU ẢNH SỐ (MSE, PSNR, DIFF MAP)
# ==============================================================================

def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Squared Error (MSE) between two 2D images.
    
    Formula: MSE = (1 / (H * W)) * sum( (I1(x, y) - I2(x, y))^2 )
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    
    Formula: PSNR = 10 * log10(MAX_I^2 / MSE)
    """
    mse = calculate_mse(img1, img2)
    if mse <= 1e-10:
        return 100.0  # Cap at 100 dB for exact mathematical equivalence
    psnr = float(10.0 * np.log10((max_val ** 2) / mse))
    return min(psnr, 100.0)


def calculate_difference_map(img1: np.ndarray, img2: np.ndarray, amplify: float = 10.0) -> np.ndarray:
    """Absolute difference map scaled by amplification factor for visual inspection."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    diff_amplified = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    return diff_amplified


# ==============================================================================
# 2. PHÉP ĐO NHỊ PHÂN VÀ HÌNH HỌC (IOU, DICE, DELTA T, CORNER ERROR)
# ==============================================================================

def calculate_binary_iou_dice(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, float]:
    """Calculate Intersection over Union (IoU) and Dice Score between two binary masks.
    
    Formulas:
      IoU = |A cap B| / |A cup B|
      Dice = 2 * |A cap B| / (|A| + |B|)
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"Shape mismatch: {mask1.shape} vs {mask2.shape}")
    b1 = mask1 > 0
    b2 = mask2 > 0
    intersection = float(np.logical_and(b1, b2).sum())
    union = float(np.logical_or(b1, b2).sum())
    total_area = float(b1.sum() + b2.sum())

    iou = 1.0 if union == 0.0 else intersection / union
    dice = 1.0 if total_area == 0.0 else (2.0 * intersection) / total_area
    return round(iou, 4), round(dice, 4)


def calculate_otsu_delta_t(t_scratch: float, t_opencv: float) -> float:
    """Absolute difference between Scratch and OpenCV Otsu threshold values (Delta T = |T*_sc - T*_cv|)."""
    return round(abs(float(t_scratch) - float(t_opencv)), 4)


def calculate_corner_error(corners_pred: np.ndarray, corners_gt: np.ndarray) -> float:
    """Mean Euclidean localization error (in pixels) across the 4 detected sheet corners."""
    c1 = np.asarray(corners_pred, dtype=np.float64).reshape(4, 2)
    c2 = np.asarray(corners_gt, dtype=np.float64).reshape(4, 2)
    dists = np.sqrt(np.sum((c1 - c2) ** 2, axis=1))
    return round(float(np.mean(dists)), 4)


# ==============================================================================
# 3. CHUẨN HÓA VÀ ĐÁNH GIÁ ĐỘ CHÍNH XÁC NHẬN DIỆN ĐÁP ÁN (ACCURACY)
# ==============================================================================

VALID_OMR_OPTIONS = {"A", "B", "C", "D"}
BLANK_TOKENS = {"", "BLANK", "NONE", "NAN", "NA", "N/A", "NULL"}


def _normalize_mark_value(raw_ans: Any) -> str:
    """Normalize any single/multi answer representation into 'A'|'B'|'C'|'D'|'BLANK'|'MULTIPLE'."""
    if raw_ans is None:
        return "BLANK"
    if isinstance(raw_ans, float) and np.isnan(raw_ans):
        return "BLANK"

    if isinstance(raw_ans, (list, tuple, set)):
        cleaned_opts: List[str] = []
        for x in raw_ans:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                continue
            token = str(x).strip().upper()
            if token and token not in BLANK_TOKENS and token not in cleaned_opts:
                cleaned_opts.append(token)
        if len(cleaned_opts) == 0:
            return "BLANK"
        if len(cleaned_opts) == 1:
            single = cleaned_opts[0]
            return single if single in VALID_OMR_OPTIONS else ("BLANK" if single in BLANK_TOKENS else "MULTIPLE")
        return "MULTIPLE"

    student_ans = str(raw_ans).strip().upper()
    if student_ans in BLANK_TOKENS:
        return "BLANK"
    if "," in student_ans or "/" in student_ans or (len(student_ans) >= 2 and all(c in VALID_OMR_OPTIONS for c in student_ans)):
        return "MULTIPLE"
    if student_ans in ("MULTIPLE", "INVALID", "DOUBLE") or student_ans not in VALID_OMR_OPTIONS:
        return "MULTIPLE"
    return student_ans


def calculate_mark_accuracy(
    pred_answers: Dict[Any, Any],
    ground_truth_answers: Dict[Any, Any]
) -> float:
    """Calculate Mark Detection Accuracy (%) against ground-truth marks on the sheet."""
    if not ground_truth_answers:
        return 0.0
    matched = 0
    total = len(ground_truth_answers)
    for q, gt in ground_truth_answers.items():
        if q in pred_answers:
            p = pred_answers[q]
        else:
            p = pred_answers.get(str(q), "BLANK")
        if _normalize_mark_value(p) == _normalize_mark_value(gt):
            matched += 1
    return round((matched / float(total)) * 100.0, 2)


# ==============================================================================
# 4. BENCHMARK TOÀN DIỆN LIÊN PHÂN HỆ (TV1 - TV4)
# ==============================================================================

def benchmark_all_modules(image: np.ndarray) -> List[Dict[str, Any]]:
    """Run comprehensive quantitative unit benchmarks across TV4 algorithms (and TV1-TV3 if present)."""
    import cv2

    has_tv1 = False
    try:
        from .tv1_preprocessing import (
            rgb_to_gray_scratch,
            gaussian_blur_scratch,
            canny_edge_scratch,
            detect_sheet_corners
        )
        has_tv1 = True
    except ImportError:
        try:
            import sys
            from pathlib import Path
            tv1_dir = str(Path(__file__).resolve().parent.parent.parent / "OMR_System_TV1" / "src")
            if tv1_dir not in sys.path:
                sys.path.insert(0, tv1_dir)
            from tv1_preprocessing import (
                rgb_to_gray_scratch,
                gaussian_blur_scratch,
                canny_edge_scratch,
                detect_sheet_corners
            )
            has_tv1 = True
        except ImportError:
            pass

    has_tv2 = False
    match_template_ncc = None
    try:
        from tv2_perspective import warp_perspective_sheet, match_template_ncc
        has_tv2 = True
    except ImportError:
        try:
            from .tv2_perspective import warp_perspective_sheet, match_template_ncc
            has_tv2 = True
        except ImportError:
            try:
                import sys
                from pathlib import Path
                tv2_dir = str(Path(__file__).resolve().parent.parent.parent / "OMR_System_TV2")
                if tv2_dir not in sys.path:
                    sys.path.insert(0, tv2_dir)
                from tv2_perspective import warp_perspective_sheet, match_template_ncc
                has_tv2 = True
            except ImportError:
                pass

    has_tv3 = False
    try:
        from .tv3_segmentation import (
            otsu_threshold_scratch,
            adaptive_mean_threshold_scratch,
            morphology_open_scratch,
            morphology_close_scratch,
        )
        has_tv3 = True
    except ImportError:
        try:
            import sys
            from pathlib import Path
            tv3_dir = str(Path(__file__).resolve().parent.parent.parent / "OMR_System_TV3" / "src")
            if tv3_dir not in sys.path:
                sys.path.insert(0, tv3_dir)
            from tv3_segmentation import (
                otsu_threshold_scratch,
                adaptive_mean_threshold_scratch,
                opening_binary_scratch as morphology_open_scratch,
                closing_binary_scratch as morphology_close_scratch,
            )
            has_tv3 = True
        except ImportError:
            pass

    if image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    results: List[Dict[str, Any]] = []

    # 1-3. TV1 Benchmarks (if TV1 module present)
    if has_tv1:
        if image.ndim == 3 and image.shape[2] >= 3:
            t0 = time.perf_counter()
            try:
                g_sc = rgb_to_gray_scratch(image, is_bgr=True)
            except TypeError:
                g_sc = rgb_to_gray_scratch(image[:, :, :3][:, :, ::-1])
            t_sc = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            g_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            t_cv = (time.perf_counter() - t0) * 1000.0
            results.append({
                "module": "Phân hệ 1 (Tiền xử lý)",
                "algorithm": "RGB to Grayscale (BT.601)",
                "metric_type": "intensity",
                "mse": round(calculate_mse(g_sc, g_cv), 4),
                "psnr_db": round(calculate_psnr(g_sc, g_cv), 2),
                "iou": None,
                "dice": None,
                "delta_t": None,
                "time_scratch_ms": round(t_sc, 3),
                "time_opencv_ms": round(t_cv, 3),
            })

        t0 = time.perf_counter()
        b_sc = gaussian_blur_scratch(gray, kernel_size=5, sigma=1.4)
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        b_cv = cv2.GaussianBlur(gray, (5, 5), 1.4, borderType=cv2.BORDER_REFLECT_101)
        t_cv = (time.perf_counter() - t0) * 1000.0
        results.append({
            "module": "Phân hệ 1 (Tiền xử lý)",
            "algorithm": "Gaussian Blur 2D (5x5)",
            "metric_type": "intensity",
            "mse": round(calculate_mse(b_sc, b_cv), 4),
            "psnr_db": round(calculate_psnr(b_sc, b_cv), 2),
            "iou": None,
            "dice": None,
            "delta_t": None,
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

        t0 = time.perf_counter()
        e_sc = canny_edge_scratch(gray, 50.0, 150.0)
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        e_cv = cv2.Canny(b_cv, 50, 150, L2gradient=True)
        t_cv = (time.perf_counter() - t0) * 1000.0
        iou_e, dice_e = calculate_binary_iou_dice(e_sc, e_cv)
        results.append({
            "module": "Phân hệ 1 (Tiền xử lý)",
            "algorithm": "Canny Edge Detection",
            "metric_type": "binary",
            "mse": round(calculate_mse(e_sc, e_cv), 4),
            "psnr_db": round(calculate_psnr(e_sc, e_cv), 2),
            "iou": iou_e,
            "dice": dice_e,
            "delta_t": None,
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

        t0 = time.perf_counter()
        c_sc = detect_sheet_corners(gray, use_scratch=True)
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        c_cv = detect_sheet_corners(gray, use_scratch=False)
        t_cv = (time.perf_counter() - t0) * 1000.0
        c_err = calculate_corner_error(c_sc, c_cv)
        results.append({
            "module": "Phân hệ 1 (Tiền xử lý)",
            "algorithm": "Hough & Quad Corner Detection",
            "metric_type": "corner",
            "mse": round(calculate_mse(c_sc, c_cv), 4),
            "psnr_db": round(calculate_psnr(c_sc, c_cv), 2),
            "iou": None,
            "dice": None,
            "delta_t": c_err,
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

    # 4. TV2: Homography SVD & Bilinear Warp + NCC Template Matching (if TV2 module present)
    if has_tv2:
        h, w = gray.shape
        quad = np.array([
            [0.02 * w, 0.02 * h],
            [0.97 * w, 0.01 * h],
            [0.98 * w, 0.98 * h],
            [0.01 * w, 0.97 * h],
        ], dtype=np.float32)
        t0 = time.perf_counter()
        w_sc = warp_perspective_sheet(gray, quad, 800, 1000, use_scratch=True)
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        w_cv = warp_perspective_sheet(gray, quad, 800, 1000, use_scratch=False)
        t_cv = (time.perf_counter() - t0) * 1000.0
        results.append({
            "module": "Phân hệ 2 (Hình học SVD)",
            "algorithm": "SVD Homography + Bilinear Warp",
            "metric_type": "intensity",
            "mse": round(calculate_mse(w_sc, w_cv), 4),
            "psnr_db": round(calculate_psnr(w_sc, w_cv), 2),
            "iou": None,
            "dice": None,
            "delta_t": None,
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

        if match_template_ncc is not None and h >= 30 and w >= 30:
            anchor = np.zeros((30, 30), dtype=np.uint8)
            anchor[5:25, 5:25] = 245
            anchor[11:19, 11:19] = 0
            mh = max(30, int(h * 0.25))
            mw = max(30, int(w * 0.25))
            roi_tr = gray[0:mh, max(0, w - mw):w]
            t0 = time.perf_counter()
            ncc_sc = match_template_ncc(roi_tr, anchor)
            t_sc = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            ncc_cv = cv2.matchTemplate(roi_tr, anchor, cv2.TM_CCOEFF_NORMED)
            t_cv = (time.perf_counter() - t0) * 1000.0
            results.append({
                "module": "Phân hệ 2 (Hình học SVD)",
                "algorithm": "Template Matching NCC (30x30)",
                "metric_type": "intensity",
                "mse": round(calculate_mse(ncc_sc, ncc_cv), 4),
                "psnr_db": round(calculate_psnr(ncc_sc, ncc_cv, max_val=1.0), 2),
                "iou": None,
                "dice": None,
                "delta_t": None,
                "time_scratch_ms": round(t_sc, 3),
                "time_opencv_ms": round(t_cv, 3),
            })

    # 5-7. TV3 Benchmarks (if TV3 module present)
    if has_tv3:
        t0 = time.perf_counter()
        t_val_sc, otsu_sc = otsu_threshold_scratch(gray, method="BINARY_INV")
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        t_val_cv, otsu_cv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        t_cv = (time.perf_counter() - t0) * 1000.0
        iou_o, dice_o = calculate_binary_iou_dice(otsu_sc, otsu_cv)
        results.append({
            "module": "Phân hệ 3 (Phân đoạn)",
            "algorithm": "Otsu Global Thresholding",
            "metric_type": "binary",
            "mse": round(calculate_mse(otsu_sc, otsu_cv), 4),
            "psnr_db": round(calculate_psnr(otsu_sc, otsu_cv), 2),
            "iou": iou_o,
            "dice": dice_o,
            "delta_t": calculate_otsu_delta_t(t_val_sc, t_val_cv),
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

        t0 = time.perf_counter()
        adap_sc = adaptive_mean_threshold_scratch(gray, block_size=51, C=42.0)
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        adap_cv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42
        )
        t_cv = (time.perf_counter() - t0) * 1000.0
        iou_a, dice_a = calculate_binary_iou_dice(adap_sc, adap_cv)
        results.append({
            "module": "Phân hệ 3 (Phân đoạn)",
            "algorithm": "Adaptive Mean Thresholding",
            "metric_type": "binary",
            "mse": round(calculate_mse(adap_sc, adap_cv), 4),
            "psnr_db": round(calculate_psnr(adap_sc, adap_cv), 2),
            "iou": iou_a,
            "dice": dice_a,
            "delta_t": None,
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

        kernel3 = np.ones((3, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        open_sc = morphology_open_scratch(adap_sc, kernel3)
        close_sc = morphology_close_scratch(open_sc, kernel3)
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        open_cv = cv2.morphologyEx(adap_cv, cv2.MORPH_OPEN, kernel3)
        close_cv = cv2.morphologyEx(open_cv, cv2.MORPH_CLOSE, kernel3)
        t_cv = (time.perf_counter() - t0) * 1000.0
        iou_m, dice_m = calculate_binary_iou_dice(close_sc, close_cv)
        results.append({
            "module": "Phân hệ 3 (Phân đoạn)",
            "algorithm": "Morphological Open + Close (3x3)",
            "metric_type": "binary",
            "mse": round(calculate_mse(close_sc, close_cv), 4),
            "psnr_db": round(calculate_psnr(close_sc, close_cv), 2),
            "iou": iou_m,
            "dice": dice_m,
            "delta_t": None,
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

    # 8. TV4: Histogram Equalization & Median Filter (Always tested)
    try:
        from .tv4_filters import benchmark_hist_equalization, benchmark_median_filter
    except ImportError:
        from tv4_filters import benchmark_hist_equalization, benchmark_median_filter

    bm_he = benchmark_hist_equalization(gray, iterations=3)
    results.append({
        "module": "Phân hệ 4 (Cải thiện ảnh)",
        "algorithm": bm_he["algorithm"],
        "metric_type": "intensity",
        "mse": bm_he["mse"],
        "psnr_db": bm_he["psnr_db"],
        "iou": None,
        "dice": None,
        "delta_t": None,
        "time_scratch_ms": bm_he["time_scratch_ms"],
        "time_opencv_ms": bm_he["time_opencv_ms"],
    })
    bm_mf = benchmark_median_filter(gray, kernel_size=3, iterations=3)
    results.append({
        "module": "Phân hệ 4 (Khử nhiễu)",
        "algorithm": bm_mf["algorithm"],
        "metric_type": "intensity",
        "mse": bm_mf["mse"],
        "psnr_db": bm_mf["psnr_db"],
        "iou": None,
        "dice": None,
        "delta_t": None,
        "time_scratch_ms": bm_mf["time_scratch_ms"],
        "time_opencv_ms": bm_mf["time_opencv_ms"],
    })

    return results


# ==============================================================================
# 5. ABLATION STUDY (NGHIÊN CỨU ĐỘ NHẠY THAM SỐ)
# ==============================================================================

def _classify_question_from_ratios(
    q_ratios: Dict[str, float],
    fill_threshold: float,
    delta_tau_min: float = 0.15,
) -> str:
    """Classify a single question ('A'|'B'|'C'|'D'|'BLANK'|'MULTIPLE') using tau_min and delta_tau_min."""
    cands = [opt for opt, r in q_ratios.items() if r >= fill_threshold]
    if len(cands) == 0:
        return "BLANK"
    if len(cands) == 1:
        return cands[0]
    sorted_cands = sorted(q_ratios.items(), key=lambda x: x[1], reverse=True)
    if (sorted_cands[0][1] - sorted_cands[1][1]) >= delta_tau_min:
        return sorted_cands[0][0]
    return "MULTIPLE"


def _compute_bubble_ratios_from_binary(
    binary_img: np.ndarray,
    total_questions: int = 20,
    question_grid: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
    bubble_radius: Optional[int] = None,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """Compute unrounded and rounded bubble fill ratios on a binary mask."""
    h, w = binary_img.shape[:2]
    questions_per_col = (total_questions + 1) // 2
    if bubble_radius is None:
        bubble_radius = int(min(w, h) * 0.018)
    inner_radius = max(bubble_radius - 2, 4)
    yy, xx = np.ogrid[-inner_radius:inner_radius + 1, -inner_radius:inner_radius + 1]
    circle_mask = (xx ** 2 + yy ** 2) <= (inner_radius ** 2)
    mask_area = float(np.sum(circle_mask))

    def_start_y, def_end_y = int(h * 0.16), int(h * 0.89)
    def_col1_x, def_col2_x = int(w * 0.13), int(w * 0.57)
    def_row_height = (def_end_y - def_start_y) / float(questions_per_col)

    raw_ratios_dict: Dict[int, Dict[str, float]] = {}
    rounded_ratios_dict: Dict[int, Dict[str, float]] = {}
    for q in range(1, total_questions + 1):
        q_raw: Dict[str, float] = {}
        q_round: Dict[str, float] = {}
        for opt_idx, opt in enumerate(["A", "B", "C", "D"]):
            if question_grid is not None and q in question_grid and opt in question_grid[q]:
                center_x, center_y = question_grid[q][opt]
                center_x, center_y = int(round(center_x)), int(round(center_y))
            else:
                row_idx = (q - 1) if q <= questions_per_col else (q - 1 - questions_per_col)
                base_x = def_col1_x if q <= questions_per_col else def_col2_x
                center_y = int(def_start_y + (row_idx + 0.5) * def_row_height)
                center_x = int(base_x + (opt_idx + 1) * (bubble_radius * 2.8))

            y1, y2 = center_y - inner_radius, center_y + inner_radius + 1
            x1, x2 = center_x - inner_radius, center_x + inner_radius + 1
            if y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
                roi = binary_img[y1:y2, x1:x2]
                ratio = float(np.sum((roi == 255) & circle_mask)) / mask_area
            else:
                ratio = 0.0
            q_raw[opt] = ratio
            q_round[opt] = round(ratio, 4)
        raw_ratios_dict[q] = q_raw
        rounded_ratios_dict[q] = q_round

    return raw_ratios_dict, rounded_ratios_dict


def _extract_bubble_answers_from_binary(
    binary_img: np.ndarray,
    total_questions: int = 20,
    fill_threshold: float = 0.38,
    delta_tau_min: float = 0.15,
    question_grid: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
    bubble_radius: Optional[int] = None
) -> Tuple[Dict[int, str], Dict[int, Dict[str, float]]]:
    """Helper for evaluating bubble fills on any binary image.
    Supports dynamically resolved question_grid and applies delta_tau rule for erasure parity with main pipeline.
    """
    raw_ratios, rounded_ratios = _compute_bubble_ratios_from_binary(
        binary_img,
        total_questions=total_questions,
        question_grid=question_grid,
        bubble_radius=bubble_radius,
    )
    ans_dict: Dict[int, str] = {
        q: _classify_question_from_ratios(q_raw, fill_threshold, delta_tau_min)
        for q, q_raw in raw_ratios.items()
    }
    return ans_dict, rounded_ratios


def run_ablation_study(
    warped_gray: np.ndarray,
    ground_truth_answers: Dict[int, Any],
    thresholds: Optional[List[float]] = None,
    question_grid: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
    bubble_radius: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Ablation Study (Section 6.2):
    1. Compare Global Otsu vs Adaptive Mean Thresholding on the sheet (using unified pipeline params).
    2. Survey fill_threshold tau_min in {0.25, 0.30, 0.35, 0.38, 0.45, 0.50}.
    """
    import cv2

    has_tv3 = False
    try:
        from tv3_thresholding import adaptive_mean_threshold_scratch, otsu_threshold_scratch
        from tv3_morphology_ccl import opening_binary_scratch as morphology_open_scratch
        has_tv3 = True
    except ImportError:
        try:
            from .tv3_segmentation import (
                otsu_threshold_scratch,
                adaptive_mean_threshold_scratch,
                morphology_open_scratch,
            )
            has_tv3 = True
        except ImportError:
            try:
                import sys
                from pathlib import Path
                tv3_dir = str(Path(__file__).resolve().parent.parent.parent / "OMR_System_TV3" / "src")
                if tv3_dir not in sys.path:
                    sys.path.insert(0, tv3_dir)
                from tv3_thresholding import adaptive_mean_threshold_scratch, otsu_threshold_scratch
                from tv3_morphology_ccl import opening_binary_scratch as morphology_open_scratch
                has_tv3 = True
            except ImportError:
                pass

    if warped_gray.ndim == 3 and warped_gray.shape[2] == 1:
        warped_gray = warped_gray[:, :, 0]
    elif warped_gray.ndim == 3:
        warped_gray = cv2.cvtColor(warped_gray, cv2.COLOR_BGR2GRAY)

    if thresholds is None:
        thresholds = [0.25, 0.30, 0.35, 0.38, 0.45, 0.50]

    if not ground_truth_answers:
        return {
            "error": "Không có dữ liệu đáp án chuẩn (Ground Truth) để tính độ chính xác kiểm định.",
            "otsu_accuracy_pct": 0.0,
            "adaptive_accuracy_pct": 0.0,
            "threshold_survey": [],
        }

    total_q = len(ground_truth_answers)

    # Auto resolve grid & bubble radius if not provided to match pipeline behavior
    if question_grid is None or bubble_radius is None:
        try:
            try:
                from .pipeline_coordinator import _resolve_bubble_grid
            except ImportError:
                from pipeline_coordinator import _resolve_bubble_grid
            resolved_grid, resolved_radius = _resolve_bubble_grid(warped_gray, total_q, use_scratch=True)
            if question_grid is None:
                question_grid = resolved_grid
            if bubble_radius is None:
                bubble_radius = resolved_radius
        except Exception:
            pass

    kernel3 = np.ones((3, 3), np.uint8)
    if has_tv3:
        # Use unified pipeline configuration: block_size=51, C=42.0
        adap_bin = adaptive_mean_threshold_scratch(warped_gray, block_size=51, C=42.0)
        adap_clean = morphology_open_scratch(adap_bin, kernel3)

        _, otsu_bin = otsu_threshold_scratch(warped_gray, method="BINARY_INV")
        otsu_clean = morphology_open_scratch(otsu_bin, kernel3)
    else:
        adap_bin = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42)
        adap_clean = cv2.morphologyEx(adap_bin, cv2.MORPH_OPEN, kernel3)

        _, otsu_bin = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_clean = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, kernel3)

    raw_ratios_adap, _ = _compute_bubble_ratios_from_binary(
        adap_clean, total_questions=total_q, question_grid=question_grid, bubble_radius=bubble_radius
    )
    ans_adap = {
        q: _classify_question_from_ratios(q_raw, 0.38, 0.15)
        for q, q_raw in raw_ratios_adap.items()
    }
    acc_adap = calculate_mark_accuracy(ans_adap, ground_truth_answers)

    ans_otsu, _ = _extract_bubble_answers_from_binary(
        otsu_clean, total_q, fill_threshold=0.38, delta_tau_min=0.15,
        question_grid=question_grid, bubble_radius=bubble_radius
    )
    acc_otsu = calculate_mark_accuracy(ans_otsu, ground_truth_answers)

    # Part B: Survey fill_threshold tau_min with delta_tau erasure rule on unrounded ratios
    tau_survey = []
    for tau in thresholds:
        ans_t = {
            q: _classify_question_from_ratios(q_raw, tau, 0.15)
            for q, q_raw in raw_ratios_adap.items()
        }
        acc_t = calculate_mark_accuracy(ans_t, ground_truth_answers)
        tau_survey.append({
            "fill_threshold": tau,
            "mark_accuracy_pct": acc_t,
            "blank_count": sum(1 for v in ans_t.values() if v == "BLANK"),
            "multiple_count": sum(1 for v in ans_t.values() if v == "MULTIPLE"),
        })

    return {
        "otsu_accuracy_pct": acc_otsu,
        "adaptive_accuracy_pct": acc_adap,
        "threshold_survey": tau_survey,
    }
