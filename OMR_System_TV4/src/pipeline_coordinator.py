"""Pipeline Coordinator: End-to-End OMR System Pipeline
Production Computer Vision Engine

Integrates all modules into a seamless high-performance pipeline.
Supports both 100% Scratch NumPy mode and OpenCV Baseline mode with detailed latency profiling.
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

try:
    from .tv4_enhancement_eval import (
        histogram_equalization_scratch,
        median_filter_scratch,
        OMRGrader
    )
except (ImportError, ValueError):
    from tv4_enhancement_eval import (
        histogram_equalization_scratch,
        median_filter_scratch,
        OMRGrader
    )

# Optional dynamic imports (integrating TV1, TV2, and TV3 modules)
try:
    from .tv1_preprocessing import rgb_to_gray_scratch, detect_sheet_corners
except (ImportError, ValueError):
    try:
        import sys
        from pathlib import Path
        tv1_dir = str(Path(__file__).resolve().parent.parent.parent / "OMR_System_TV1" / "src")
        if tv1_dir not in sys.path:
            sys.path.insert(0, tv1_dir)
        from tv1_preprocessing import rgb_to_gray_scratch, detect_sheet_corners
    except ImportError:
        rgb_to_gray_scratch = None
        detect_sheet_corners = None

try:
    from tv2_perspective import warp_perspective_sheet
except ImportError:
    try:
        from .tv2_perspective import warp_perspective_sheet
    except (ImportError, ValueError):
        try:
            import sys
            from pathlib import Path
            tv2_dir = str(Path(__file__).resolve().parent.parent.parent / "OMR_System_TV2")
            if tv2_dir not in sys.path:
                sys.path.insert(0, tv2_dir)
            from tv2_perspective import warp_perspective_sheet
        except ImportError:
            warp_perspective_sheet = None

try:
    from tv3_thresholding import adaptive_mean_threshold_scratch
    from tv3_morphology_ccl import two_pass_ccl_scratch
    from tv3_segmentation import TV3PipelineConfig, TV3SegmentationPipeline, extract_bubble_answers
except ImportError:
    try:
        import sys
        from pathlib import Path
        tv3_dir = str(Path(__file__).resolve().parent.parent.parent / "OMR_System_TV3" / "src")
        if tv3_dir not in sys.path:
            sys.path.insert(0, tv3_dir)
        from tv3_thresholding import adaptive_mean_threshold_scratch
        from tv3_morphology_ccl import two_pass_ccl_scratch
        from tv3_segmentation import TV3PipelineConfig, TV3SegmentationPipeline, extract_bubble_answers
    except ImportError:
        adaptive_mean_threshold_scratch = None
        two_pass_ccl_scratch = None
        TV3PipelineConfig = None
        TV3SegmentationPipeline = None
        extract_bubble_answers = None


def _resize_bilinear_scratch(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize 2D or 3D image using vectorized bilinear interpolation (100% NumPy from scratch).

    Implements OpenCV-compatible half-pixel center aligned coordinate mapping:
      src_coord = (dst_coord + 0.5) * scale - 0.5
    """
    dst_w, dst_h = target_size
    src_h, src_w = image.shape[:2]
    if src_h == dst_h and src_w == dst_w:
        return image.copy()

    scale_y = float(src_h) / float(dst_h)
    scale_x = float(src_w) / float(dst_w)

    y = (np.arange(dst_h, dtype=np.float32) + 0.5) * scale_y - 0.5
    x = (np.arange(dst_w, dtype=np.float32) + 0.5) * scale_x - 0.5

    y = np.clip(y, 0, src_h - 1)
    x = np.clip(x, 0, src_w - 1)

    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, src_w - 1)

    dy = (y - y0).reshape(-1, 1)
    dx = (x - x0).reshape(1, -1)

    if image.ndim == 2:
        top = (1.0 - dx) * image[y0[:, None], x0[None, :]] + dx * image[y0[:, None], x1[None, :]]
        bot = (1.0 - dx) * image[y1[:, None], x0[None, :]] + dx * image[y1[:, None], x1[None, :]]
        out = (1.0 - dy) * top + dy * bot
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    else:
        channels = []
        for c in range(image.shape[2]):
            ch = image[:, :, c]
            top = (1.0 - dx) * ch[y0[:, None], x0[None, :]] + dx * ch[y0[:, None], x1[None, :]]
            bot = (1.0 - dx) * ch[y1[:, None], x0[None, :]] + dx * ch[y1[:, None], x1[None, :]]
            out = (1.0 - dy) * top + dy * bot
            channels.append(np.clip(np.round(out), 0, 255).astype(np.uint8))
        return np.stack(channels, axis=2)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners: top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]  # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _detect_sheet_corners_cv(image: np.ndarray) -> np.ndarray:
    """Robust sheet boundary corner detection using OpenCV."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # 1. Check if the image has a dark desk/background surrounding the sheet
    corner_pixels = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
    if np.mean(corner_pixels) < 120:
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 0.25 * (w * h):
                peri = cv2.arcLength(c, True)
                for eps in [0.02, 0.03, 0.015, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(c, eps * peri, True)
                    if len(approx) == 4:
                        return _order_corners(approx.reshape(4, 2))
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                return _order_corners(box)

    # 2. General edge-based quad detection on desk
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 120)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(c) > 0.25 * (w * h):
                peri = cv2.arcLength(c, True)
                found_pts = None
                for eps in [0.02, 0.03, 0.015, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(c, eps * peri, True)
                    if len(approx) == 4:
                        found_pts = approx.reshape(4, 2).astype(np.float32)
                        break
                if found_pts is None:
                    rect = cv2.minAreaRect(c)
                    found_pts = cv2.boxPoints(rect).astype(np.float32)

                ordered = _order_corners(found_pts)
                x_min, y_min = np.min(ordered, axis=0)
                x_max, y_max = np.max(ordered, axis=0)
                if (x_min > 0.02 * w and x_min < 0.08 * w and
                    y_min > 0.03 * h and y_min < 0.10 * h and
                    x_max > 0.92 * w and y_max > 0.90 * h):
                    # Detected contour is the inner fiducial border; sheet is already the full image
                    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

                return ordered

    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _warp_perspective_cv(image: np.ndarray, corners: np.ndarray, target_width: int = 800, target_height: int = 1000) -> np.ndarray:
    """Baseline perspective warping using OpenCV."""
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    y_sorted = pts[np.argsort(pts[:, 1])]
    tl, tr = y_sorted[:2][np.argsort(y_sorted[:2, 0])]
    bl, br = y_sorted[2:][np.argsort(y_sorted[2:, 0])]
    rect = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (target_width, target_height))


def _resolve_bubble_grid(
    warped_gray: np.ndarray,
    total_questions: int = 20,
    use_scratch: bool = True
) -> Tuple[Dict[int, Dict[str, Tuple[float, float]]], int]:
    """Dynamically resolve bubble grid coordinates for any exam sheet.

    If the sheet matches the standard PTIT template within tolerance, standard reference
    coordinates are preserved for 100% mathematical consistency.
    If the sheet is a cropped upload, phone photo, or non-standard scan, the grid parameters
    (start_y, row_height, col1_x, col2_x, step_x) adaptively adjust to the true bubble marks.
    In Scratch mode, candidate bubble extraction runs 100% from scratch via TV3's adaptive threshold & CCL.
    """
    h, w = warped_gray.shape[:2]
    questions_per_col = (total_questions + 1) // 2

    # Standard template baseline parameters
    def_start_y = int(h * 0.16)
    def_end_y = int(h * 0.89)
    def_row_height = (def_end_y - def_start_y) / float(questions_per_col)
    def_col1_x = int(w * 0.13)
    def_col2_x = int(w * 0.57)
    bubble_radius = int(min(w, h) * 0.018)
    def_dx = bubble_radius * 2.8

    bubbles = []
    if use_scratch and adaptive_mean_threshold_scratch is not None and two_pass_ccl_scratch is not None:
        thresh = adaptive_mean_threshold_scratch(warped_gray, block_size=31, C=15.0)
        _, _, raw_stats = two_pass_ccl_scratch(thresh, connectivity=8)
        for s in raw_stats:
            bw, bh = s["bbox"][2], s["bbox"][3]
            if (80 <= s["area"] <= 1500 and 0.65 <= s["aspect_ratio"] <= 1.55 
                and 14 <= bw <= 50 and 14 <= bh <= 50 and s["extent"] >= 0.20):
                bubbles.append(s["centroid"])
    else:
        # Extract bubble contour candidates via OpenCV baseline
        thresh = cv2.adaptiveThreshold(
            warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42
        )
        cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_area = int(np.pi * (bubble_radius * 0.4) ** 2)
        max_area = int(np.pi * (bubble_radius * 2.8) ** 2)
        for c in cnts:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = float(bw) / float(bh)
                if 0.65 <= aspect <= 1.5:
                    peri = cv2.arcLength(c, True)
                    if 4 * np.pi * area / (peri * peri + 1e-5) > 0.30:
                        bubbles.append((bx + bw / 2.0, by + bh / 2.0))

    inner = [b for b in bubbles if 0.05 * h <= b[1] <= 0.95 * h]

    start_y = float(def_start_y)
    row_height = float(def_row_height)
    col1_x = float(def_col1_x)
    col2_x = float(def_col2_x)
    dx = float(def_dx)
    row_exact_ys = None

    if len(inner) >= 16:
        # Cluster bubble centers into distinct rows
        cluster_tol = 18 if questions_per_col > 8 else 30
        rows = []
        for b in sorted(inner, key=lambda p: p[1]):
            placed = False
            for r in rows:
                if abs(np.mean([p[1] for p in r]) - b[1]) < cluster_tol:
                    r.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])

        valid_rows = [r for r in rows if len(r) >= 4 and np.mean([p[1] for p in r]) >= 0.14 * h]
        valid_rows.sort(key=lambda r: np.mean([p[1] for p in r]))
        if len(valid_rows) == questions_per_col:
            row_exact_ys = [float(np.mean([p[1] for p in r])) for r in valid_rows]

        q_bubbles = [b for b in inner if b[1] >= 0.14 * h]
        y_arr = np.array([b[1] for b in (q_bubbles if len(q_bubbles) >= 12 else inner)])
        y_p5, y_p95 = np.percentile(y_arr, 5), np.percentile(y_arr, 95)
        if questions_per_col > 1:
            est_row_height = (y_p95 - y_p5) / float(questions_per_col - 1)
        else:
            est_row_height = def_row_height
        est_start_y = y_p5 - 0.5 * est_row_height

        # Calculate horizontal step between adjacent bubbles in same row
        dxs = []
        for i in range(len(inner)):
            for j in range(i + 1, min(i + 15, len(inner))):
                if abs(inner[i][1] - inner[j][1]) < 18:
                    dx_val = abs(inner[i][0] - inner[j][0])
                    if 28 <= dx_val <= 75:
                        dxs.append(dx_val)
        est_dx = float(np.median(dxs)) if len(dxs) >= 8 else def_dx

        mid_x = w * 0.5
        left_xs = [b[0] for b in inner if b[0] < mid_x]
        right_xs = [b[0] for b in inner if b[0] >= mid_x]

        if len(left_xs) >= 8 and len(right_xs) >= 8:
            # Option D is the rightmost bubble of the 4 options (A, B, C, D) in each column.
            d1 = float(np.percentile(left_xs, 97))
            d2 = float(np.percentile(right_xs, 97))
            c1 = d1 - 4.0 * est_dx
            c2 = d2 - 4.0 * est_dx
            if 0.01 * w <= c1 <= 0.40 * w and 0.35 * w <= c2 <= 0.85 * w:
                est_col1_x = c1
                est_col2_x = c2
            else:
                est_col1_x = float(def_col1_x)
                est_col2_x = float(def_col2_x)
        else:
            est_col1_x = float(def_col1_x)
            est_col2_x = float(def_col2_x)

        # Deviation check: if within tolerance of standard 20-question template, preserve baseline
        if total_questions == 20 and abs(est_start_y - def_start_y) <= 15 and abs(est_row_height - def_row_height) <= 6:
            row_exact_ys = None
            start_y = def_start_y
            row_height = def_row_height
            col1_x = def_col1_x
            col2_x = def_col2_x
            dx = def_dx
        else:
            start_y = est_start_y
            row_height = est_row_height
            dx = est_dx
            col1_x = est_col1_x
            col2_x = est_col2_x
            if est_dx > 45:
                bubble_radius = int(round(est_dx * 0.35))

    options = ["A", "B", "C", "D"]
    grid: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for q in range(1, total_questions + 1):
        row_idx = (q - 1) if q <= questions_per_col else (q - 1 - questions_per_col)
        base_x = col1_x if q <= questions_per_col else col2_x
        if row_exact_ys is not None:
            cy = row_exact_ys[row_idx]
        else:
            cy = float(start_y + (row_idx + 0.5) * row_height)
        q_dict: Dict[str, Tuple[float, float]] = {}
        for opt_idx, opt in enumerate(options):
            cx = float(base_x + (opt_idx + 1) * dx)
            q_dict[opt] = (cx, cy)
        grid[q] = q_dict

    return grid, bubble_radius


def detect_sheet_questions(warped_gray: np.ndarray, use_scratch: bool = True) -> int:
    """Detects the approximate number of question bubbles/rows on the rectified exam sheet.

    Returns estimated total questions (e.g. 10, 20, 30), or 0 if indeterminate.
    Assumes standard 2-column layout (each question row contains options across both columns).
    In Scratch mode, candidate bubble extraction runs 100% from scratch via TV3's adaptive threshold & CCL.
    """
    if warped_gray is None or warped_gray.size == 0:
        return 0
    h, w = warped_gray.shape[:2]
    bubble_radius = int(min(w, h) * 0.018)

    bubbles = []
    if use_scratch and adaptive_mean_threshold_scratch is not None and two_pass_ccl_scratch is not None:
        thresh = adaptive_mean_threshold_scratch(warped_gray, block_size=31, C=15.0)
        _, _, raw_stats = two_pass_ccl_scratch(thresh, connectivity=8)
        for s in raw_stats:
            bw, bh = s["bbox"][2], s["bbox"][3]
            if (80 <= s["area"] <= 1500 and 0.65 <= s["aspect_ratio"] <= 1.55 
                and 14 <= bw <= 50 and 14 <= bh <= 50 and s["extent"] >= 0.20):
                bubbles.append(s["centroid"])
    else:
        thresh = cv2.adaptiveThreshold(
            warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42
        )
        cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_area = int(np.pi * (bubble_radius * 0.4) ** 2)
        max_area = int(np.pi * (bubble_radius * 2.8) ** 2)
        for c in cnts:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = float(bw) / float(bh)
                if 0.65 <= aspect <= 1.5:
                    peri = cv2.arcLength(c, True)
                    if 4 * np.pi * area / (peri * peri + 1e-5) > 0.30:
                        bubbles.append((bx + bw / 2.0, by + bh / 2.0))

    inner = [b for b in bubbles if 0.05 * h <= b[1] <= 0.95 * h]
    if len(inner) < 12:
        return 0

    # Cluster bubbles vertically into question rows
    rows = []
    for b in sorted(inner, key=lambda p: p[1]):
        placed = False
        for r in rows:
            if abs(np.mean([p[1] for p in r]) - b[1]) < 22:
                r.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    # Filter rows that correspond to question rows (have at least 4 bubbles and in question area)
    q_rows = [r for r in rows if len(r) >= 4 and 0.14 * h <= np.mean([p[1] for p in r]) <= 0.92 * h]
    num_rows = len(q_rows)
    if num_rows == 0:
        return 0

    return num_rows * 2


def _extract_bubble_answers_cv(
    warped_gray: np.ndarray,
    total_questions: int = 20,
    fill_threshold: float = 0.38,
    delta_tau_min: float = 0.15,
    question_grid: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
    bubble_radius: Optional[int] = None
) -> Tuple[Dict[int, str], Dict[int, Dict[str, float]], np.ndarray, Dict[int, str]]:
    """Baseline bubble mark extraction using OpenCV with unified parameters and Δτ rule."""
    h, w = warped_gray.shape[:2]
    binary_adaptive = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42
    )
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
    vis_img = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

    options = ["A", "B", "C", "D"]
    student_answers: Dict[int, str] = {}
    fill_ratios_all: Dict[int, Dict[str, float]] = {}
    erased_questions: Dict[int, str] = {}

    if bubble_radius is None:
        bubble_radius = int(min(w, h) * 0.018)
    inner_radius = max(bubble_radius - 2, 4)

    yy, xx = np.ogrid[-inner_radius:inner_radius + 1, -inner_radius:inner_radius + 1]
    circle_mask = (xx ** 2 + yy ** 2) <= (inner_radius ** 2)
    mask_area = float(np.sum(circle_mask))

    if question_grid is None:
        question_grid, _ = _resolve_bubble_grid(warped_gray, total_questions, use_scratch=False)

    for q in range(1, total_questions + 1):
        q_ratios: Dict[str, float] = {}
        for opt in options:
            center_x, center_y = question_grid[q][opt]
            cx, cy = int(round(center_x)), int(round(center_y))
            y1, y2 = cy - inner_radius, cy + inner_radius + 1
            x1, x2 = cx - inner_radius, cx + inner_radius + 1
            if y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
                roi = binary[y1:y2, x1:x2]
                filled = float(np.sum((roi == 255) & circle_mask))
                ratio = filled / mask_area
            else:
                ratio = 0.0
            q_ratios[opt] = round(ratio, 4)

        fill_ratios_all[q] = q_ratios

        # Decision rule with Δτ (identical to TV3 classify_question_bubbles)
        sorted_opts = sorted(q_ratios.keys(), key=lambda o: q_ratios[o], reverse=True)
        opt1, opt2 = sorted_opts[0], sorted_opts[1]
        tau1, tau2 = q_ratios[opt1], q_ratios[opt2]

        if tau1 < fill_threshold:
            student_answers[q] = "BLANK"
        elif tau1 >= fill_threshold and tau2 < fill_threshold:
            student_answers[q] = opt1
        elif tau1 >= fill_threshold and tau2 >= fill_threshold:
            if (tau1 - tau2) >= delta_tau_min:
                student_answers[q] = opt1
                erased_questions[q] = opt1
            else:
                student_answers[q] = "MULTIPLE"

    return student_answers, fill_ratios_all, vis_img, erased_questions


class OMRPipeline:
    """End-to-End Pipeline Manager for Automatic OMR Processing."""

    def __init__(self, answer_key_path: Optional[str] = None):
        self.grader = OMRGrader(answer_key_path)

    def process_sheet(
        self,
        image_bgr: np.ndarray,
        student_id: str = "SV001",
        use_scratch_mode: bool = True,
        apply_enhancement: bool = True,
        fill_threshold: float = 0.38
    ) -> Dict[str, Any]:
        """Execute the complete 4-stage pipeline on input exam sheet image.

        Stages:
          1. Preprocessing (Grayscale conversion + Median Filter + Adaptive Contrast Enhancement)
          2. Perspective detection and rectification (SVD Homography + Bilinear Warping)
          3. Segmentation and bubble mark recognition (Adaptive Threshold + Morphology + Circular Mask)
          4. Grading against answer key and color-coded visual annotation (Green/Red/Orange)
        """
        latencies = {}
        t_start = time.perf_counter()

        # --- Stage 0: Input Sanitization & Resolution Normalization (Report Section 3.1) ---
        # Normalize high-resolution smartphone captures (e.g. 3000x4000) to standard height H=1000 px.
        # This reduces 2D convolution computation by ~12x to ~43x while preserving crisp 18x18 px bubble ROIs.
        img_work = image_bgr
        if img_work.ndim == 3 and img_work.shape[2] >= 4:
            img_work = img_work[:, :, :3]

        cur_h, cur_w = img_work.shape[:2]
        if cur_h > 1250:
            scale = 1000.0 / float(cur_h)
            new_w = int(round(cur_w * scale))
            if use_scratch_mode:
                img_work = _resize_bilinear_scratch(img_work, (new_w, 1000))
            else:
                img_work = cv2.resize(img_work, (new_w, 1000), interpolation=cv2.INTER_LINEAR)

        # --- Stage 1: Preprocessing (Grayscale & Impulse Noise Removal) ---
        t0 = time.perf_counter()
        if use_scratch_mode and rgb_to_gray_scratch is not None:
            gray = rgb_to_gray_scratch(img_work, is_bgr=True) if img_work.ndim == 3 else img_work.copy()
        else:
            gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY) if img_work.ndim == 3 else img_work.copy()

        if apply_enhancement:
            gray = median_filter_scratch(gray, kernel_size=3) if use_scratch_mode else cv2.medianBlur(gray, 3)
        latencies["stage1_preprocess_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # --- Stage 2: Perspective Detection, Warping & Low-Light Contrast Enhancement ---
        t0 = time.perf_counter()
        if use_scratch_mode and detect_sheet_corners is not None:
            corners = detect_sheet_corners(gray, use_scratch=True)
        else:
            corners = _detect_sheet_corners_cv(img_work)

        if use_scratch_mode and warp_perspective_sheet is not None:
            try:
                warped_gray = warp_perspective_sheet(gray, corners, target_width=800, target_height=1000, use_scratch=True)
            except ValueError:
                # TV2 báo ValueError khi 4 góc suy biến (thẳng hàng / trùng nhau):
                # quay về giả định phiếu chiếm toàn khung hình, giống fallback của TV1.
                gh, gw = gray.shape[:2]
                corners = np.array([[0, 0], [gw - 1, 0], [gw - 1, gh - 1], [0, gh - 1]], dtype=np.float32)
                warped_gray = warp_perspective_sheet(gray, corners, target_width=800, target_height=1000, use_scratch=True)
        else:
            warped_gray = _warp_perspective_cv(gray, corners, target_width=800, target_height=1000)

        # Apply TV4 Histogram Equalization on rectified sheet when overall highlight is low-light
        # (use 95th percentile so uneven cast-shadow sheets with bright white regions aren't over-darkened)
        if apply_enhancement and float(np.percentile(warped_gray, 95)) < 185.0:
            if use_scratch_mode:
                warped_gray = histogram_equalization_scratch(warped_gray)
            else:
                warped_gray = cv2.equalizeHist(warped_gray)
        latencies["stage2_perspective_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # --- Stage 3: Mark Extraction ---
        t0 = time.perf_counter()
        total_q = len(self.grader.answer_key) if self.grader.answer_key else 20
        answers = None
        ratios = None
        vis_warped = None
        erased_questions = {}
        actual_mode = "Scratch (NumPy)" if use_scratch_mode else "OpenCV Baseline"
        fallback_warning = None

        # Detect physical sheet question count and check structure mismatch
        detected_q = detect_sheet_questions(warped_gray, use_scratch=use_scratch_mode)
        sheet_warning = None
        if detected_q > 0 and abs(detected_q - total_q) >= 4:
            sheet_warning = (
                f"Cảnh báo cấu trúc: Phiếu bài thi thực tế có khoảng {detected_q} câu hỏi "
                f"({detected_q // 2} hàng), nhưng file đáp án chuẩn đang cấu hình {total_q} câu. "
                f"Có thể bạn đã nạp nhầm ảnh bài thi hoặc chọn sai file đáp án!"
            )

        # Dynamically resolve question bubble grid (adaptive to custom uploads or standard template)
        grid, b_radius = _resolve_bubble_grid(warped_gray, total_q, use_scratch=use_scratch_mode)

        if use_scratch_mode and TV3PipelineConfig is not None and TV3SegmentationPipeline is not None:
            # Direct invocation of TV3 core segmentation pipeline with unified parameters
            try:
                inner_r = max(b_radius - 2, 5)
                config_tv3 = TV3PipelineConfig(
                    num_questions=total_q,
                    tau_min=fill_threshold,
                    delta_tau_min=0.15,
                    bubble_radius=inner_r,
                    block_size=51,
                    C=42.0,
                    apply_opening=True,
                    apply_closing=False,
                    threshold_method="adaptive"
                )
                pipeline_tv3 = TV3SegmentationPipeline(config_tv3)
                res_tv3 = pipeline_tv3.run(warped_gray, question_grid=grid, run_opencv_benchmark=False)

                answers = {}
                ratios = {}
                for q_id, q_res in res_tv3.omr_result.questions.items():
                    ratios[q_id] = {k: round(v, 4) for k, v in q_res.fill_ratios.items()}
                    if q_res.status == "ERASED_VALID" and q_res.selected:
                        answers[q_id] = q_res.selected
                        erased_questions[q_id] = q_res.selected
                    elif q_res.status == "VALID" and q_res.selected:
                        answers[q_id] = q_res.selected
                    elif q_res.status == "MULTIPLE_INVALID":
                        answers[q_id] = "MULTIPLE"
                    else:
                        answers[q_id] = "BLANK"

                vis_warped = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR) if warped_gray.ndim == 2 else warped_gray.copy()
            except Exception as exc:
                actual_mode = "Fallback (OpenCV Baseline)"
                fallback_warning = f"Lỗi nhánh Scratch: {exc}. Hệ thống đã tự động kích hoạt Fallback sang OpenCV Baseline."
                answers = None

        if answers is None:
            if use_scratch_mode and not fallback_warning:
                actual_mode = "Fallback (OpenCV Baseline)"
                fallback_warning = "Không tìm thấy module TV3 Scratch. Đã tự động kích hoạt Fallback sang OpenCV Baseline."
            answers, ratios, vis_warped, cv_erased = _extract_bubble_answers_cv(
                warped_gray,
                total_questions=total_q,
                fill_threshold=fill_threshold,
                delta_tau_min=0.15,
                question_grid=grid,
                bubble_radius=b_radius
            )
            erased_questions.update(cv_erased)
        latencies["stage3_segmentation_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # --- Stage 4: Grading & Final Color-Coded Annotation ---
        t0 = time.perf_counter()
        grading_result = self.grader.grade_student(student_id, answers, erased_questions=erased_questions)
        combined_warning = sheet_warning
        if fallback_warning:
            combined_warning = f"{fallback_warning} | {sheet_warning}" if sheet_warning else fallback_warning

        if combined_warning:
            grading_result["warning"] = combined_warning
            grading_result["detected_sheet_questions"] = detected_q

        annotated_image = self._annotate_sheet_results(
            vis_warped,
            grading_result,
            ratios,
            fill_threshold,
            question_grid=grid,
            bubble_radius=b_radius,
            warning=combined_warning,
            detected_questions=detected_q
        )
        latencies["stage4_grading_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        latencies["total_pipeline_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

        return {
            "student_id": student_id,
            "mode": actual_mode,
            "grading_result": grading_result,
            "corners": corners,
            "warped_gray": warped_gray,
            "annotated_image": annotated_image,
            "fill_ratios": ratios,
            "latencies": latencies,
            "detected_questions": detected_q,
            "warning": combined_warning,
            "erased_questions": erased_questions
        }

    def _annotate_sheet_results(
        self,
        vis_image: np.ndarray,
        grading_result: Dict[str, Any],
        fill_ratios: Optional[Dict[int, Dict[str, float]]] = None,
        fill_threshold: float = 0.38,
        question_grid: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
        bubble_radius: Optional[int] = None,
        warning: Optional[str] = None,
        detected_questions: Optional[int] = None
    ) -> np.ndarray:
        """Draw color-coded circles (Green=Correct, Red=Wrong, Orange=Blank/Multiple) and score banners."""
        img = vis_image.copy()
        h, w = img.shape[:2]
        total_questions = grading_result["total_questions"]

        if bubble_radius is None:
            bubble_radius = int(min(w, h) * 0.018)
        options = ["A", "B", "C", "D"]

        # Color palette (BGR format for OpenCV)
        COLOR_CORRECT = (40, 180, 40)     # Green
        COLOR_WRONG = (30, 30, 230)       # Red
        COLOR_WARNING = (0, 150, 255)     # Orange / Amber

        if question_grid is None:
            gray_for_grid = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            question_grid, _ = _resolve_bubble_grid(gray_for_grid, total_questions)

        for item in grading_result["details"]:
            q = item["question"]
            stu_ans = item["student_ans"]
            cor_ans = item["correct_ans"]
            status = item["status"]

            for opt in options:
                center_x, center_y = question_grid[q][opt]
                cx, cy = int(round(center_x)), int(round(center_y))

                if status == "CORRECT" and opt == stu_ans:
                    # Student answered correctly -> Bold Green Circle
                    cv2.circle(img, (cx, cy), bubble_radius + 3, COLOR_CORRECT, 3)
                elif status == "INCORRECT":
                    if opt == stu_ans:
                        # Student's wrong choice -> Bold Red Circle
                        cv2.circle(img, (cx, cy), bubble_radius + 3, COLOR_WRONG, 3)
                    elif opt == cor_ans:
                        # Show true answer in Green Outline
                        cv2.circle(img, (cx, cy), bubble_radius + 3, COLOR_CORRECT, 2)
                elif status == "BLANK":
                    if opt == cor_ans:
                        # Highlight missed correct answer in Orange/Amber
                        cv2.circle(img, (cx, cy), bubble_radius + 3, COLOR_WARNING, 2)
                elif status == "MULTIPLE":
                    q_rat = fill_ratios.get(q, {}) if fill_ratios else {}
                    if q_rat.get(opt, 0.0) >= fill_threshold:
                        # Highlight all multiple-filled bubbles in Orange/Amber
                        cv2.circle(img, (cx, cy), bubble_radius + 3, COLOR_WARNING, 3)
                    if opt == cor_ans:
                        cv2.circle(img, (cx, cy), bubble_radius + 6, COLOR_CORRECT, 2)

            # Draw short status tag on the right side of the question row
            d_x, d_y = question_grid[q]["D"]
            last_x = int(round(d_x + bubble_radius * 1.5 + 8))
            row_cy = int(round(d_y))
            if item.get("is_erased"):
                cv2.putText(img, f"TAY XOA ({stu_ans})", (last_x, row_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WARNING, 1)
            elif status == "CORRECT":
                cv2.putText(img, "DUNG", (last_x, row_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_CORRECT, 1)
            elif status == "INCORRECT":
                cv2.putText(img, f"SAI ({cor_ans})", (last_x, row_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WRONG, 1)
            elif status == "BLANK":
                cv2.putText(img, f"TRONG ({cor_ans})", (last_x, row_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WARNING, 1)
            elif status == "MULTIPLE":
                cv2.putText(img, f"LOI >1 ({cor_ans})", (last_x, row_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WARNING, 1)

        # Draw top banner (above fiducial border at y=62)
        if warning and ("Cảnh báo cấu trúc" in warning or "LECH DE THI" in warning):
            cv2.rectangle(img, (0, 0), (w, 54), (25, 35, 185), -1)
            det_q = detected_questions if detected_questions is not None else grading_result.get("detected_sheet_questions", "?")
            warn_header = f"CANH BAO: ANH ~{det_q} CAU != DAP AN {total_questions} CAU (LECH DE THI!)"
            cv2.putText(img, warn_header, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 235, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (0, 0), (w, 54), (30, 35, 45), -1)
            score_text = f"MSV: {grading_result['student_id']} | DIEM: {grading_result['score_10']:.1f}/10.0 ({grading_result['correct_count']}/{total_questions})"
            cv2.putText(img, score_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (80, 255, 120), 2, cv2.LINE_AA)

        # Draw bottom legend & counts banner (below fiducial border at y=942)
        cv2.rectangle(img, (0, h - 48), (w, h), (30, 35, 45), -1)
        sub_text = (
            f"Xanh: Dung ({grading_result['correct_count']}) | "
            f"Do: Sai ({grading_result['incorrect_count']}) | "
            f"Cam: Bo trong ({grading_result['blank_count']}) / Pham quy ({grading_result['multiple_count']})"
        )
        cv2.putText(img, sub_text, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)

        return img
