"""Module TV4 - Filters: Image Enhancement & Noise Removal Algorithms from Scratch
Thành viên 4: Thuật toán Cải thiện chất lượng ảnh số & Khử nhiễu không gian (Scratch NumPy)

Các thuật toán cài đặt từ đầu (Pure NumPy, không dùng hàm OpenCV cv2.equalizeHist, cv2.medianBlur):
1. Cân bằng lược đồ mức xám (Histogram Equalization):
   - Đếm tần suất mức xám h(k) và tính hàm phân phối tích lũy CDF(k).
   - Ánh xạ chuẩn hóa mức xám s_k theo min-CDF (chuẩn OpenCV) hoặc Textbook CDF.
   - Xử lý an toàn cho ảnh đồng nhất (all-black, all-white, single-tone).
2. Bộ lọc trung vị 2D (Median Filter):
   - Đệm biên ảnh tự xây dựng _pad_2d_scratch (edge, reflect, constant).
   - Trích xuất lân cận cửa sổ trượt k x k thành các mặt phẳng lân cận 3D.
   - Sắp xếp chèn vector hóa (Vectorized Insertion Sort) và trích xuất giá trị trung vị.
3. Hàm Benchmark định lượng so sánh Scratch vs OpenCV (MSE, PSNR, Difference Map, Execution Time).
"""

from __future__ import annotations
import time
from typing import Dict, Any, Union
import numpy as np

try:
    from .tv4_metrics import calculate_mse, calculate_psnr, calculate_difference_map
except ImportError:
    try:
        from tv4_metrics import calculate_mse, calculate_psnr, calculate_difference_map
    except ImportError:
        # Fallback inline metrics if imported in isolation
        def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
            if img1.shape != img2.shape:
                raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
            diff = img1.astype(np.float64) - img2.astype(np.float64)
            return float(np.mean(diff ** 2))

        def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
            mse = calculate_mse(img1, img2)
            if mse <= 1e-10:
                return 100.0
            psnr = float(10.0 * np.log10((max_val ** 2) / mse))
            return min(psnr, 100.0)

        def calculate_difference_map(img1: np.ndarray, img2: np.ndarray, amplify: float = 10.0) -> np.ndarray:
            if img1.shape != img2.shape:
                raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
            diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
            return np.clip(diff * amplify, 0, 255).astype(np.uint8)


# ==============================================================================
# 1. HISTOGRAM EQUALIZATION (CÂN BẰNG LƯỢC ĐỒ MỨC XÁM - SCRATCH)
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


# ==============================================================================
# 2. PADDING & MEDIAN FILTER 2D (BỘ LỌC TRUNG VỊ - SCRATCH)
# ==============================================================================

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

    Implementation highlights (100% pure NumPy without np.pad, sliding_window_view, or OpenCV):
      1. Manual 2D boundary padding via `_pad_2d_scratch`.
      2. Explicit collection of the k*k shifted 2D neighborhood slices `padded[dy:dy+h, dx:dx+w]`.
      3. Rank selection: vectorized compare-exchange insertion sort for k <= 3, and O(N) `np.partition`
         selection for k >= 5 to avoid O(k^4) sorting cost while guaranteeing exact parity with OpenCV.

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

    # Step 3: Fast median rank selection
    # For small kernels (k <= 3, 9 elements), vectorized compare-exchange/insertion sort runs in ~7 ms.
    # For larger kernels (k >= 5), using np.partition along the channel axis (axis=0) provides O(N) selection,
    # eliminating O(k^4) full sorting overhead while guaranteeing 100% exact numerical match with OpenCV.
    if kernel_size <= 3:
        for i in range(1, num_elements):
            for j in range(i, 0, -1):
                a = neighbors[j - 1]
                b = neighbors[j]
                low = np.minimum(a, b)
                high = np.maximum(a, b)
                neighbors[j - 1] = low
                neighbors[j] = high
        return neighbors[median_idx].copy()
    else:
        return np.partition(neighbors, median_idx, axis=0)[median_idx].copy()


# ==============================================================================
# 3. BENCHMARK HOÀN CHỈNH CHO PHÂN HỆ 4
# ==============================================================================

def benchmark_hist_equalization(image: np.ndarray, iterations: int = 5) -> Dict[str, Any]:
    """Benchmark Histogram Equalization: Scratch vs OpenCV."""
    import cv2

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
    import cv2

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
