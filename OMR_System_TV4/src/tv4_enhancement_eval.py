"""Module TV4: Image Enhancement, Benchmarking, CSV Matrix I/O, and Grading Logic
Author: Thanh Vien 4 - OMR Project Team 4
Course: Digital Image Processing (INT13146) - PTIT

Algorithms implemented from scratch using NumPy:
1. Histogram Equalization from scratch (using CDF mapping, robust to uniform & edge-case images)
2. Median Filter 2D from scratch (vectorized insertion sort across neighborhood planes, pure NumPy without np.sort/np.partition)
3. Robust CSV Matrix I/O for grayscale matrices (according to instructor's verification slides 10-12)
4. Comprehensive Quantitative Benchmarks (MSE, PSNR, Difference Map, Execution Time)
5. Grading System, Result Validation, and Professional Excel Reporting with OpenPyXL styling
"""

from __future__ import annotations
import os
import io
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import cv2
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# ==============================================================================
# 1. CORE ALGORITHMS FROM SCRATCH (NUMPY ONLY)
# ==============================================================================

def histogram_equalization_scratch(image: np.ndarray, use_min_cdf: bool = True) -> np.ndarray:
    """Enhance image contrast using Histogram Equalization implemented from scratch.

    Mathematical formulation:
      1. Compute discrete histogram h(k) for k in [0, 255] by counting pixel occurrences.
      2. If all pixels share the same intensity (single-level/uniform), return unchanged copy.
      3. Compute Cumulative Distribution Function (CDF) iteratively:
             cdf(0) = h(0)
             cdf(k) = cdf(k - 1) + h(k)   for k = 1..255
      4. Map input level k to output level s_k:
         - Standard min-CDF normalization (use_min_cdf=True, matches OpenCV):
             cdf_min = min { cdf(k) | cdf(k) > 0 }
             s_k = round( (cdf(k) - cdf_min) / (H * W - cdf_min) * 255 )
         - Textbook global CDF normalization (use_min_cdf=False, Gonzalez & Woods):
             s_k = round( cdf(k) / (H * W) * 255 )

    Args:
        image: 2D numpy array of uint8 (grayscale) or shape (H, W, 1).
        use_min_cdf: Whether to normalize by cdf_min (True matches OpenCV; False uses textbook formula).

    Returns:
        Equalized 2D numpy array of uint8.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy ndarray")

    # Squeeze trailing singleton dimension if present (e.g. from cv2)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim != 2:
        raise ValueError(f"Histogram Equalization requires 2D grayscale image, got shape {image.shape}")

    if image.dtype != np.uint8:
        image = np.clip(np.round(image), 0, 255).astype(np.uint8)

    h, w = image.shape
    total_pixels = h * w
    if total_pixels == 0:
        return image.copy()

    # Step 1: Compute discrete 256-bin histogram from scratch (without np.bincount / np.histogram)
    flat = image.ravel()
    hist = np.zeros(256, dtype=np.int64)
    np.add.at(hist, flat, 1)

    # Find minimum gray level p0 with non-zero count
    p0 = -1
    for k in range(256):
        if hist[k] > 0:
            p0 = k
            break

    if p0 == -1 or total_pixels == int(hist[p0]):
        # Edge Case: Empty or uniform single-level image -> return unchanged copy
        return image.copy()

    # Step 2: Compute Cumulative Distribution Function (CDF) iteratively from scratch
    cdf = np.zeros(256, dtype=np.int64)
    running_sum = 0
    for k in range(256):
        running_sum += int(hist[k])
        cdf[k] = running_sum

    # Step 3: Construct Lookup Table (LUT) s_k in [0, 255]
    lut = np.zeros(256, dtype=np.uint8)
    if use_min_cdf:
        cdf_min = int(cdf[p0])
        denom = float(total_pixels - cdf_min)
        scale = 255.0 / denom
        lut[p0] = 0
        for k in range(p0 + 1, 256):
            # Python's round() uses round-half-to-even (matching OpenCV cvRound / C++)
            val = int(round(float(cdf[k] - cdf_min) * scale))
            lut[k] = 0 if val < 0 else (255 if val > 255 else val)
    else:
        scale = 255.0 / float(total_pixels)
        for k in range(256):
            val = int(round(float(cdf[k]) * scale))
            lut[k] = 0 if val < 0 else (255 if val > 255 else val)

    # Step 4: Apply LUT transformation to every pixel
    equalized = lut[image]
    return equalized


def _reflect_index(pos: int, length: int) -> int:
    """Map out-of-bounds index `pos` into [0, length - 1] via boundary reflection (matches np.pad reflect)."""
    if length <= 1:
        return 0
    period = 2 * (length - 1)
    idx = pos % period
    if idx >= length:
        idx = period - idx
    return idx


def _pad_2d_scratch(image: np.ndarray, pad: int, mode: str = "edge") -> np.ndarray:
    """Manual 2D boundary padding implemented from scratch without np.pad."""
    h, w = image.shape
    if pad <= 0 or h == 0 or w == 0:
        return image.copy()

    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=image.dtype)
    # Copy center region
    padded[pad:pad + h, pad:pad + w] = image

    if mode == "constant":
        return padded

    if mode == "reflect":
        for i in range(pad):
            # Top row i corresponds to coordinate (i - pad) < 0; bottom row (pad + h + i) -> (h + i)
            src_top = _reflect_index(i - pad, h)
            src_bot = _reflect_index(h + i, h)
            padded[i, pad:pad + w] = image[src_top, :]
            padded[pad + h + i, pad:pad + w] = image[src_bot, :]
        for j in range(pad):
            src_left = _reflect_index(j - pad, w)
            src_right = _reflect_index(w + j, w)
            padded[:, j] = padded[:, pad + src_left]
            padded[:, pad + w + j] = padded[:, pad + src_right]
        return padded

    if mode != "edge":
        raise ValueError(f"Unsupported padding_mode '{mode}'. Choose 'edge', 'reflect', or 'constant'.")

    # Default: 'edge' (replicate boundary pixels, matching OpenCV BORDER_REPLICATE)
    for i in range(pad):
        padded[i, pad:pad + w] = image[0, :]
        padded[pad + h + i, pad:pad + w] = image[-1, :]
    for j in range(pad):
        padded[:, j] = padded[:, pad]
        padded[:, pad + w + j] = padded[:, pad + w - 1]
    return padded


def median_filter_scratch(
    image: np.ndarray,
    kernel_size: int = 3,
    padding_mode: str = "edge"
) -> np.ndarray:
    """2D Median Filter implemented 100% from scratch using basic NumPy slicing & insertion sort.

    Mathematical principle:
      For each pixel (x, y), extract local neighborhood W of size k x k:
          W(x, y) = { I(x + u, y + v) | u, v in [-r, r] },  r = (k - 1) // 2
      Sort the N = k^2 neighborhood intensities in ascending order:
          W_sorted[0] <= W_sorted[1] <= ... <= W_sorted[N - 1]
      Select the middle rank (median) at index m = (k^2) // 2:
          I_filtered(x, y) = W_sorted[m]

    Implementation highlights (100% pure from-scratch without np.pad, sliding_window_view, or np.partition):
      1. Manual 2D boundary padding via `_pad_2d_scratch`.
      2. Explicit collection of the k*k shifted 2D neighborhood slices `padded[dy:dy+h, dx:dx+w]`.
      3. Explicit vectorized insertion sort across the k*k neighborhood channels to find the exact median.

    Args:
        image: 2D numpy array of uint8 (grayscale).
        kernel_size: Odd integer >= 1 (e.g. 3, 5, 7).
        padding_mode: Padding mode ('edge', 'reflect', 'constant').

    Returns:
        Filtered 2D numpy array of uint8.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy ndarray")

    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim != 2:
        raise ValueError(f"Median filter requires 2D grayscale image, got shape {image.shape}")

    if not isinstance(kernel_size, (int, np.integer)) or isinstance(kernel_size, bool) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size}")
    kernel_size = int(kernel_size)

    if kernel_size == 1 or image.size == 0:
        return image.copy()

    if image.dtype != np.uint8:
        image = np.clip(np.round(image), 0, 255).astype(np.uint8)

    h, w = image.shape
    pad = kernel_size // 2
    num_elements = kernel_size * kernel_size
    median_idx = num_elements // 2

    # Step 1: Manual boundary padding (no np.pad)
    padded = _pad_2d_scratch(image, pad=pad, mode=padding_mode)

    # Step 2: Collect all k*k shifted neighborhood planes (no sliding_window_view)
    neighbors = np.empty((num_elements, h, w), dtype=np.uint8)
    idx = 0
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            neighbors[idx] = padded[dy:dy + h, dx:dx + w]
            idx += 1

    # Step 3: Explicit vectorized insertion sort across the k*k neighborhood planes (no np.partition/np.sort)
    for i in range(1, num_elements):
        for j in range(i, 0, -1):
            a = neighbors[j - 1]
            b = neighbors[j]
            low = np.minimum(a, b)
            high = np.maximum(a, b)
            neighbors[j - 1] = low
            neighbors[j] = high

    # Step 4: Extract the exact middle element at index (k*k) // 2
    return neighbors[median_idx].copy()


# ==============================================================================
# 2. MATRIX CSV I/O (FOR INSTRUCTOR VERIFICATION - SLIDES 10, 11, 12)
# ==============================================================================

def save_matrix_to_csv(matrix: np.ndarray, file_path: str) -> None:
    """Save 2D uint8 image matrix to CSV file.

    Format: comma-separated integer values (0-255).
    """
    if matrix.ndim != 2:
        raise ValueError(f"Only 2D matrices can be saved to CSV, got shape {matrix.shape}")
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    np.savetxt(file_path, matrix.astype(int), fmt="%d", delimiter=",")


def load_matrix_from_csv(file_path_or_content: Union[str, io.StringIO]) -> np.ndarray:
    """Robustly load 2D uint8 image matrix from CSV file or StringIO buffer.

    Tolerates:
      - Trailing commas at line ends
      - Semicolons or commas as delimiters
      - Extra whitespaces and empty lines
      - Comment or header lines with non-numeric text
    """
    lines = []
    if isinstance(file_path_or_content, str):
        if not os.path.exists(file_path_or_content):
            raise FileNotFoundError(f"CSV matrix file not found: {file_path_or_content}")
        with open(file_path_or_content, "r", encoding="utf-8-sig") as f:
            raw_lines = f.readlines()
    else:
        raw_lines = file_path_or_content.getvalue().splitlines()

    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Replace semicolons with commas and split
        parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
        if not parts:
            continue
        try:
            row_vals = [float(p) for p in parts]
            lines.append(row_vals)
        except ValueError:
            # Skip non-numeric title/header row if present
            continue

    if not lines:
        raise ValueError("No valid numeric matrix rows found in CSV")

    # Ensure all rows have equal column count
    num_cols = len(lines[0])
    for idx, row in enumerate(lines):
        if len(row) != num_cols:
            raise ValueError(f"Inconsistent column count at row {idx+1}: expected {num_cols}, got {len(row)}")

    data = np.array(lines, dtype=np.float64)
    clipped = np.clip(np.round(data), 0, 255).astype(np.uint8)
    return clipped


# ==============================================================================
# 3. BENCHMARK & QUANTITATIVE EVALUATION METRICS
# ==============================================================================

def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Squared Error (MSE) between two 2D images."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
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


def calculate_binary_iou_dice(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, float]:
    """Calculate Intersection over Union (IoU) and Dice Score between two binary masks."""
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
    """Absolute difference between Scratch and OpenCV Otsu threshold values (Delta T)."""
    return round(abs(float(t_scratch) - float(t_opencv)), 4)


def calculate_corner_error(corners_pred: np.ndarray, corners_gt: np.ndarray) -> float:
    """Mean Euclidean localization error (in pixels) across the 4 detected sheet corners."""
    c1 = np.asarray(corners_pred, dtype=np.float64).reshape(4, 2)
    c2 = np.asarray(corners_gt, dtype=np.float64).reshape(4, 2)
    dists = np.sqrt(np.sum((c1 - c2) ** 2, axis=1))
    return round(float(np.mean(dists)), 4)


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


def benchmark_all_modules(image: np.ndarray) -> List[Dict[str, Any]]:
    """Run comprehensive quantitative unit benchmarks across TV4 algorithms (and TV1-TV3 if present)."""
    has_tv1 = False
    try:
        from .tv1_preprocessing import rgb_to_gray_scratch, gaussian_blur_scratch, canny_edge_scratch
        has_tv1 = True
    except ImportError:
        pass

    has_tv2 = False
    try:
        from .tv2_perspective import warp_perspective_sheet
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
            rgb = image[:, :, :3][:, :, ::-1]
            t0 = time.perf_counter()
            g_sc = rgb_to_gray_scratch(rgb)
            t_sc = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            g_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            t_cv = (time.perf_counter() - t0) * 1000.0
            results.append({
                "module": "TV1",
                "algorithm": "RGB to Grayscale",
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
            "module": "TV1",
            "algorithm": "Gaussian Blur (5x5)",
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
            "module": "TV1",
            "algorithm": "Canny Edge Detection",
            "mse": round(calculate_mse(e_sc, e_cv), 4),
            "psnr_db": round(calculate_psnr(e_sc, e_cv), 2),
            "iou": iou_e,
            "dice": dice_e,
            "delta_t": None,
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

    # 4. TV2: Homography SVD & Bilinear Warp (if TV2 module present)
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
            "module": "TV2",
            "algorithm": "SVD Homography + Bilinear Warp",
            "mse": round(calculate_mse(w_sc, w_cv), 4),
            "psnr_db": round(calculate_psnr(w_sc, w_cv), 2),
            "iou": None,
            "dice": None,
            "delta_t": None,
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

    # 5-7. TV3 Benchmarks (if TV3 module present)
    if has_tv3:
        t0 = time.perf_counter()
        t_val_sc, otsu_sc = otsu_threshold_scratch(gray)
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        t_val_cv, otsu_cv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        t_cv = (time.perf_counter() - t0) * 1000.0
        iou_o, dice_o = calculate_binary_iou_dice(otsu_sc, otsu_cv)
        results.append({
            "module": "TV3",
            "algorithm": "Otsu Global Thresholding",
            "mse": round(calculate_mse(otsu_sc, otsu_cv), 4),
            "psnr_db": round(calculate_psnr(otsu_sc, otsu_cv), 2),
            "iou": iou_o,
            "dice": dice_o,
            "delta_t": calculate_otsu_delta_t(t_val_sc, t_val_cv),
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

        t0 = time.perf_counter()
        adap_sc = adaptive_mean_threshold_scratch(gray, block_size=51, c_offset=42.0)
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        adap_cv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42
        )
        t_cv = (time.perf_counter() - t0) * 1000.0
        iou_a, dice_a = calculate_binary_iou_dice(adap_sc, adap_cv)
        results.append({
            "module": "TV3",
            "algorithm": "Adaptive Mean Thresholding",
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
        open_sc = morphology_open_scratch(adap_sc, kernel_size=3)
        close_sc = morphology_close_scratch(open_sc, kernel_size=3)
        t_sc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        open_cv = cv2.morphologyEx(adap_cv, cv2.MORPH_OPEN, kernel3)
        close_cv = cv2.morphologyEx(open_cv, cv2.MORPH_CLOSE, kernel3)
        t_cv = (time.perf_counter() - t0) * 1000.0
        iou_m, dice_m = calculate_binary_iou_dice(close_sc, close_cv)
        results.append({
            "module": "TV3",
            "algorithm": "Morphological Open + Close (3x3)",
            "mse": round(calculate_mse(close_sc, close_cv), 4),
            "psnr_db": round(calculate_psnr(close_sc, close_cv), 2),
            "iou": iou_m,
            "dice": dice_m,
            "delta_t": None,
            "time_scratch_ms": round(t_sc, 3),
            "time_opencv_ms": round(t_cv, 3),
        })

    # 8. TV4: Histogram Equalization & Median Filter (Always tested)
    bm_he = benchmark_hist_equalization(gray, iterations=3)
    results.append({
        "module": "TV4",
        "algorithm": bm_he["algorithm"],
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
        "module": "TV4",
        "algorithm": bm_mf["algorithm"],
        "mse": bm_mf["mse"],
        "psnr_db": bm_mf["psnr_db"],
        "iou": None,
        "dice": None,
        "delta_t": None,
        "time_scratch_ms": bm_mf["time_scratch_ms"],
        "time_opencv_ms": bm_mf["time_opencv_ms"],
    })

    return results


def _extract_bubble_answers_from_binary(
    binary_img: np.ndarray,
    total_questions: int = 20,
    fill_threshold: float = 0.38
) -> Tuple[Dict[int, str], Dict[int, Dict[str, float]]]:
    """Helper for evaluating bubble fills on any binary image without external dependencies."""
    h, w = binary_img.shape[:2]
    questions_per_col = (total_questions + 1) // 2
    start_y, end_y = int(h * 0.16), int(h * 0.89)
    col1_x, col2_x = int(w * 0.13), int(w * 0.57)
    row_height = (end_y - start_y) / float(questions_per_col)
    bubble_radius = int(min(w, h) * 0.018)
    inner_radius = max(bubble_radius - 3, 4)
    yy, xx = np.ogrid[-inner_radius:inner_radius + 1, -inner_radius:inner_radius + 1]
    circle_mask = (xx ** 2 + yy ** 2) <= (inner_radius ** 2)
    mask_area = float(np.sum(circle_mask))

    ans_dict: Dict[int, str] = {}
    ratios_dict: Dict[int, Dict[str, float]] = {}
    for q in range(1, total_questions + 1):
        row_idx = (q - 1) if q <= questions_per_col else (q - 1 - questions_per_col)
        base_x = col1_x if q <= questions_per_col else col2_x
        center_y = int(start_y + (row_idx + 0.5) * row_height)
        q_rat: Dict[str, float] = {}
        for opt_idx, opt in enumerate(["A", "B", "C", "D"]):
            center_x = int(base_x + (opt_idx + 1) * (bubble_radius * 2.8))
            y1, y2 = center_y - inner_radius, center_y + inner_radius + 1
            x1, x2 = center_x - inner_radius, center_x + inner_radius + 1
            if y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
                roi = binary_img[y1:y2, x1:x2]
                ratio = float(np.sum((roi == 255) & circle_mask)) / mask_area
            else:
                ratio = 0.0
            q_rat[opt] = round(ratio, 4)
        ratios_dict[q] = q_rat
        cands = [opt for opt, r in q_rat.items() if r >= fill_threshold]
        ans_dict[q] = "BLANK" if len(cands) == 0 else (cands[0] if len(cands) == 1 else "MULTIPLE")
    return ans_dict, ratios_dict


def run_ablation_study(
    warped_gray: np.ndarray,
    ground_truth_answers: Dict[int, Any],
    thresholds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Run Ablation Study (Section 6.2):
    1. Compare Global Otsu vs Adaptive Mean Thresholding on the sheet.
    2. Survey fill_threshold tau_min in {0.25, 0.30, 0.35, 0.38, 0.45, 0.50}.
    """
    has_tv3 = False
    try:
        from .tv3_segmentation import (
            otsu_threshold_scratch,
            morphology_open_scratch,
            extract_bubble_answers,
        )
        has_tv3 = True
    except ImportError:
        pass

    if warped_gray.ndim == 3 and warped_gray.shape[2] == 1:
        warped_gray = warped_gray[:, :, 0]
    elif warped_gray.ndim == 3:
        warped_gray = cv2.cvtColor(warped_gray, cv2.COLOR_BGR2GRAY)

    if thresholds is None:
        thresholds = [0.25, 0.30, 0.35, 0.38, 0.45, 0.50]

    total_q = len(ground_truth_answers) if ground_truth_answers else 20

    if has_tv3:
        ans_adap, ratios_adap, _ = extract_bubble_answers(
            warped_gray, total_questions=total_q, fill_threshold=0.38, use_scratch=True
        )
        _, otsu_bin = otsu_threshold_scratch(warped_gray)
        otsu_clean = morphology_open_scratch(otsu_bin, kernel_size=3)
    else:
        adap_bin = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42)
        kernel3 = np.ones((3, 3), np.uint8)
        adap_clean = cv2.morphologyEx(adap_bin, cv2.MORPH_OPEN, kernel3)
        ans_adap, ratios_adap = _extract_bubble_answers_from_binary(adap_clean, total_q, fill_threshold=0.38)

        _, otsu_bin = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_clean = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, kernel3)

    acc_adap = calculate_mark_accuracy(ans_adap, ground_truth_answers)
    ans_otsu, _ = _extract_bubble_answers_from_binary(otsu_clean, total_q, fill_threshold=0.38)
    acc_otsu = calculate_mark_accuracy(ans_otsu, ground_truth_answers)

    # Part B: Survey fill_threshold tau_min
    tau_survey = []
    for tau in thresholds:
        ans_t: Dict[int, str] = {}
        for q, q_rat in ratios_adap.items():
            cands = [opt for opt, r in q_rat.items() if r >= tau]
            ans_t[q] = "BLANK" if len(cands) == 0 else (cands[0] if len(cands) == 1 else "MULTIPLE")
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


def benchmark_hist_equalization(image: np.ndarray, iterations: int = 5) -> Dict[str, Any]:
    """Benchmark Histogram Equalization: Scratch vs OpenCV."""
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OpenCV execution
    t0 = time.perf_counter()
    res_cv = None
    for _ in range(iterations):
        res_cv = cv2.equalizeHist(image)
    t_cv = (time.perf_counter() - t0) / iterations * 1000.0

    # Scratch execution
    t0 = time.perf_counter()
    res_scratch = None
    for _ in range(iterations):
        res_scratch = histogram_equalization_scratch(image)
    t_scratch = (time.perf_counter() - t0) / iterations * 1000.0

    mse = calculate_mse(res_scratch, res_cv)
    psnr = calculate_psnr(res_scratch, res_cv)
    amplify = 15.0
    diff_map = calculate_difference_map(res_scratch, res_cv, amplify=amplify)

    return {
        "algorithm": "Histogram Equalization",
        "time_scratch_ms": round(t_scratch, 3),
        "time_opencv_ms": round(t_cv, 3),
        "speedup_ratio": round(t_scratch / max(t_cv, 1e-4), 2),
        "mse": round(mse, 4),
        "psnr_db": round(psnr, 2),
        "img_scratch": res_scratch,
        "img_opencv": res_cv,
        "diff_map": diff_map,
        "amplify": amplify,
    }


def benchmark_median_filter(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark Median Filter: Scratch vs OpenCV."""
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OpenCV execution
    t0 = time.perf_counter()
    res_cv = None
    for _ in range(iterations):
        res_cv = cv2.medianBlur(image, kernel_size)
    t_cv = (time.perf_counter() - t0) / iterations * 1000.0

    # Scratch execution
    t0 = time.perf_counter()
    res_scratch = None
    for _ in range(iterations):
        res_scratch = median_filter_scratch(image, kernel_size=kernel_size, padding_mode="edge")
    t_scratch = (time.perf_counter() - t0) / iterations * 1000.0

    mse = calculate_mse(res_scratch, res_cv)
    psnr = calculate_psnr(res_scratch, res_cv)
    amplify = 10.0
    diff_map = calculate_difference_map(res_scratch, res_cv, amplify=amplify)

    return {
        "algorithm": f"Median Filter ({kernel_size}x{kernel_size})",
        "time_scratch_ms": round(t_scratch, 3),
        "time_opencv_ms": round(t_cv, 3),
        "speedup_ratio": round(t_scratch / max(t_cv, 1e-4), 2),
        "mse": round(mse, 4),
        "psnr_db": round(psnr, 2),
        "img_scratch": res_scratch,
        "img_opencv": res_cv,
        "diff_map": diff_map,
        "amplify": amplify,
    }


# ==============================================================================
# 4. GRADING, STATISTICAL ANALYTICS & PROFESSIONAL EXCEL REPORTING
# ==============================================================================

class OMRGrader:
    """Manages Answer Key, grading logic, and report generation."""

    VALID_OPTIONS = {"A", "B", "C", "D"}

    def __init__(self, answer_key_path: Optional[str] = None):
        self.answer_key: Dict[int, str] = {}
        if answer_key_path and os.path.exists(answer_key_path):
            self.load_answer_key(answer_key_path)

    def load_answer_key(self, csv_path: str) -> Dict[int, str]:
        """Robustly load answer key from CSV file.

        Handles:
          - Semicolon or Comma delimiters
          - UTF-8 with BOM from Windows Excel
          - Headers with extra spaces
        """
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # Strip whitespace from column names
        df.columns = [str(c).strip().lower() for c in df.columns]
        if len(df.columns) < 2:
            raise ValueError(f"Answer key CSV must contain at least 2 columns: question_id and answer, found {df.columns}")

        q_col = df.columns[0]
        ans_col = df.columns[1]

        key_dict = {}
        for _, row in df.iterrows():
            try:
                q_num = int(row[q_col])
                raw_opt = row[ans_col]
                if raw_opt is None or (isinstance(raw_opt, float) and np.isnan(raw_opt)):
                    continue
                ans = str(raw_opt).strip().upper()
                if ans in self.VALID_OPTIONS:
                    key_dict[q_num] = ans
            except (ValueError, TypeError):
                continue

        if not key_dict:
            raise ValueError("No valid question-answer pairs found in answer key CSV")

        self.answer_key = key_dict
        return self.answer_key

    def set_answer_key_dict(self, key_dict: Dict[int, str]) -> None:
        """Directly set answer key from dictionary."""
        self.answer_key = {int(k): str(v).strip().upper() for k, v in key_dict.items()}

    def grade_student(
        self,
        student_id: str,
        student_answers: Dict[Any, Any]
    ) -> Dict[str, Any]:
        """Grade single student submission against answer key."""
        if not self.answer_key:
            raise ValueError("Answer key has not been loaded yet!")

        total_questions = len(self.answer_key)
        correct_count = 0
        incorrect_count = 0
        blank_count = 0
        multiple_count = 0

        question_details = []

        for q_num in sorted(self.answer_key.keys()):
            correct_ans = self.answer_key[q_num]
            if q_num in student_answers:
                raw_ans = student_answers[q_num]
            else:
                raw_ans = student_answers.get(str(q_num), "BLANK")

            student_ans = _normalize_mark_value(raw_ans)

            if student_ans == "BLANK":
                status = "BLANK"
                blank_count += 1
                is_correct = False
            elif student_ans == "MULTIPLE":
                status = "MULTIPLE"
                multiple_count += 1
                is_correct = False
            elif student_ans == correct_ans:
                status = "CORRECT"
                correct_count += 1
                is_correct = True
            else:
                status = "INCORRECT"
                incorrect_count += 1
                is_correct = False

            question_details.append({
                "question": q_num,
                "student_ans": student_ans,
                "correct_ans": correct_ans,
                "status": status,
                "is_correct": is_correct
            })

        # Score on 10-point scale
        score_10 = round((correct_count / float(total_questions)) * 10.0, 2)

        return {
            "student_id": student_id,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "blank_count": blank_count,
            "multiple_count": multiple_count,
            "score_10": score_10,
            "details": question_details
        }

    def grade_batch(
        self,
        batch_submissions: Dict[str, Dict[int, Any]]
    ) -> List[Dict[str, Any]]:
        """Grade a batch of submissions."""
        results = []
        for sid, answers in batch_submissions.items():
            res = self.grade_student(sid, answers)
            results.append(res)
        return results

    @staticmethod
    def generate_excel_report(batch_results: List[Dict[str, Any]], output_excel_path: str) -> str:
        """Export grading report to Excel (.xlsx) with professional OpenPyXL styling.

        Includes 2 styled sheets:
          1. Bang_Diem_Tong_Hop (Summary score table with colors).
          2. Thong_Ke_Cau_Hoi (Per-question statistics and difficulty analysis).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_excel_path)), exist_ok=True)

        wb = openpyxl.Workbook()

        # Styles
        font_header = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        fill_header = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        fill_sub_header = PatternFill(start_color="2E86C1", end_color="2E86C1", fill_type="solid")
        align_center = Alignment(horizontal="center", vertical="center")
        thin_border = Border(
            left=Side(style="thin", color="D3D3D3"),
            right=Side(style="thin", color="D3D3D3"),
            top=Side(style="thin", color="D3D3D3"),
            bottom=Side(style="thin", color="D3D3D3")
        )

        fill_correct = PatternFill(start_color="E8F8F5", end_color="E8F8F5", fill_type="solid")
        fill_wrong = PatternFill(start_color="FDEDEC", end_color="FDEDEC", fill_type="solid")
        fill_warn = PatternFill(start_color="FEF9E7", end_color="FEF9E7", fill_type="solid")

        # -------------------------------------------------------------
        # SHEET 1: BẢNG ĐIỂM TỔNG HỢP
        # -------------------------------------------------------------
        ws1 = wb.active
        ws1.title = "Bang_Diem_Tong_Hop"

        headers1 = [
            "STT", "Mã Sinh Viên", "Điểm Số (Hệ 10)", "Số Câu Đúng",
            "Số Câu Sai", "Số Câu Bỏ Trống", "Số Câu Phạm Quy", "Tổng Số Câu"
        ]
        q_keys = []
        if batch_results:
            q_keys = sorted({d["question"] for r in batch_results for d in r["details"]})
            for qk in q_keys:
                headers1.append(f"Q{qk:02d}")

        ws1.append(headers1)
        for cell in ws1[1]:
            cell.font = font_header
            cell.fill = fill_header
            cell.alignment = align_center

        for stt, r in enumerate(batch_results, start=1):
            row_data = [
                stt,
                r["student_id"],
                r["score_10"],
                r["correct_count"],
                r["incorrect_count"],
                r["blank_count"],
                r["multiple_count"],
                r["total_questions"]
            ]
            det_dict = {d["question"]: d for d in r["details"]}
            for qk in q_keys:
                item = det_dict.get(qk, {"student_ans": "N/A"})
                row_data.append(item["student_ans"])

            ws1.append(row_data)
            row_idx = ws1.max_row
            # Format row cells
            for col_idx, val in enumerate(row_data, start=1):
                c = ws1.cell(row=row_idx, column=col_idx)
                c.border = thin_border
                c.alignment = align_center
                if col_idx > 8:
                    qk = q_keys[col_idx - 9]
                    status = det_dict.get(qk, {}).get("status", "")
                    if status == "CORRECT":
                        c.fill = fill_correct
                    elif status == "INCORRECT":
                        c.fill = fill_wrong
                    elif status in ("BLANK", "MULTIPLE"):
                        c.fill = fill_warn

        # Auto-adjust column widths for Sheet 1
        for col in ws1.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws1.column_dimensions[col_letter].width = max(max_len + 3, 10)

        # -------------------------------------------------------------
        # SHEET 2: THỐNG KÊ CÂU HỎI
        # -------------------------------------------------------------
        ws2 = wb.create_sheet(title="Thong_Ke_Cau_Hoi")
        headers2 = [
            "Câu Hỏi", "Đáp Án Đúng", "Số SV Đúng", "Số SV Sai",
            "Số SV Bỏ Trống", "Số SV Phạm Quy", "Tỷ Lệ Đúng (%)", "Tỷ Lệ Sai (%)"
        ]
        ws2.append(headers2)
        for cell in ws2[1]:
            cell.font = font_header
            cell.fill = fill_sub_header
            cell.alignment = align_center

        if batch_results:
            total_students = len(batch_results)
            first_det_map = {
                d["question"]: d["correct_ans"]
                for r in batch_results
                for d in r["details"]
            }
            for qk in q_keys:
                correct = sum(1 for r in batch_results if any(d["question"] == qk and d["is_correct"] for d in r["details"]))
                blank = sum(1 for r in batch_results if any(d["question"] == qk and d["status"] == "BLANK" for d in r["details"]))
                mult = sum(1 for r in batch_results if any(d["question"] == qk and d["status"] == "MULTIPLE" for d in r["details"]))
                wrong = total_students - correct - blank - mult

                acc_rate = round((correct / max(total_students, 1)) * 100.0, 1)
                err_rate = round(100.0 - acc_rate, 1)

                key_ans = first_det_map.get(qk, "")

                ws2.append([
                    f"Câu {qk:02d}", key_ans, correct, wrong, blank, mult, acc_rate, err_rate
                ])
                row_idx = ws2.max_row
                for col_idx in range(1, len(headers2) + 1):
                    c = ws2.cell(row=row_idx, column=col_idx)
                    c.border = thin_border
                    c.alignment = align_center

        for col in ws2.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws2.column_dimensions[col_letter].width = max(max_len + 3, 12)

        wb.save(output_excel_path)
        return output_excel_path
