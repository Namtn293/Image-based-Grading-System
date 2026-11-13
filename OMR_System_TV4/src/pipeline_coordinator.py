"""Pipeline Coordinator: End-to-End OMR System Pipeline
Author: Thanh Vien 4 - OMR Project Team 4
Course: Digital Image Processing (INT13146) - PTIT

Integrates all modules from TV1, TV2, TV3, and TV4 into a seamless pipeline.
Supports both 100% Scratch NumPy mode and OpenCV Baseline mode with detailed latency profiling.
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

from .tv4_enhancement_eval import (
    histogram_equalization_scratch,
    median_filter_scratch,
    OMRGrader
)

# Optional dynamic imports (if teammates add their modules in the future)
try:
    from .tv1_preprocessing import rgb_to_gray_scratch, detect_sheet_corners
except ImportError:
    rgb_to_gray_scratch = None
    detect_sheet_corners = None

try:
    from .tv2_perspective import warp_perspective_sheet
except ImportError:
    warp_perspective_sheet = None

try:
    from .tv3_segmentation import extract_bubble_answers
except ImportError:
    extract_bubble_answers = None


def _detect_sheet_corners_cv(image: np.ndarray) -> np.ndarray:
    """Baseline sheet boundary corner detection using OpenCV."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # 1. Check if the image has a dark desk/background surrounding the sheet
    corner_pixels = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
    if np.mean(corner_pixels) < 120:
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 0.25 * (w * h):
                pts = approx.reshape(4, 2).astype(np.float32)
                y_sorted = pts[np.argsort(pts[:, 1])]
                tl, tr = y_sorted[:2][np.argsort(y_sorted[:2, 0])]
                bl, br = y_sorted[2:][np.argsort(y_sorted[2:, 0])]
                return np.array([tl, tr, br, bl], dtype=np.float32)

    # 2. General edge-based quad detection on desk
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 120)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 0.25 * (w * h):
                pts = approx.reshape(4, 2).astype(np.float32)
                y_sorted = pts[np.argsort(pts[:, 1])]
                tl, tr = y_sorted[:2][np.argsort(y_sorted[:2, 0])]
                bl, br = y_sorted[2:][np.argsort(y_sorted[2:, 0])]

                # Distinguish outer sheet boundary from inner fiducial border of an already flat sheet
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                if (x_min > 0.02 * w and x_min < 0.08 * w and
                    y_min > 0.03 * h and y_min < 0.10 * h and
                    x_max > 0.92 * w and y_max > 0.90 * h):
                    # Detected contour is the inner fiducial border; the sheet is already the full image
                    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

                return np.array([tl, tr, br, bl], dtype=np.float32)

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


def _extract_bubble_answers_cv(
    warped_gray: np.ndarray,
    total_questions: int = 20,
    fill_threshold: float = 0.38
) -> Tuple[Dict[int, str], Dict[int, Dict[str, float]], np.ndarray]:
    """Baseline bubble mark extraction using OpenCV."""
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

    questions_per_col = (total_questions + 1) // 2
    start_y, end_y = int(h * 0.16), int(h * 0.89)
    col1_x, col2_x = int(w * 0.13), int(w * 0.57)
    row_height = (end_y - start_y) / float(questions_per_col)
    bubble_radius = int(min(w, h) * 0.018)
    inner_radius = max(bubble_radius - 3, 4)

    yy, xx = np.ogrid[-inner_radius:inner_radius + 1, -inner_radius:inner_radius + 1]
    circle_mask = (xx ** 2 + yy ** 2) <= (inner_radius ** 2)
    mask_area = float(np.sum(circle_mask))

    for q in range(1, total_questions + 1):
        row_idx = (q - 1) if q <= questions_per_col else (q - 1 - questions_per_col)
        base_x = col1_x if q <= questions_per_col else col2_x
        center_y = int(start_y + (row_idx + 0.5) * row_height)
        q_ratios: Dict[str, float] = {}

        for opt_idx, opt in enumerate(options):
            center_x = int(base_x + (opt_idx + 1) * (bubble_radius * 2.8))
            y1, y2 = center_y - inner_radius, center_y + inner_radius + 1
            x1, x2 = center_x - inner_radius, center_x + inner_radius + 1
            if y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
                roi = binary[y1:y2, x1:x2]
                filled = float(np.sum((roi == 255) & circle_mask))
                ratio = filled / mask_area
            else:
                ratio = 0.0
            q_ratios[opt] = round(ratio, 4)

        fill_ratios_all[q] = q_ratios
        cands = [opt for opt, r in q_ratios.items() if r >= fill_threshold]
        if len(cands) == 1:
            student_answers[q] = cands[0]
        elif len(cands) > 1:
            student_answers[q] = "MULTIPLE"
        else:
            student_answers[q] = "BLANK"

    return student_answers, fill_ratios_all, vis_img


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
            img_work = cv2.resize(img_work, (new_w, 1000), interpolation=cv2.INTER_LINEAR)

        # --- Stage 1: Preprocessing (Grayscale & Impulse Noise Removal) ---
        t0 = time.perf_counter()
        if use_scratch_mode and rgb_to_gray_scratch is not None:
            gray = rgb_to_gray_scratch(img_work[:, :, :3][:, :, ::-1]) if img_work.ndim == 3 else img_work.copy()
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
        if use_scratch_mode and extract_bubble_answers is not None:
            answers, ratios, vis_warped = extract_bubble_answers(
                warped_gray,
                total_questions=total_q,
                fill_threshold=fill_threshold,
                use_scratch=True
            )
        else:
            answers, ratios, vis_warped = _extract_bubble_answers_cv(
                warped_gray,
                total_questions=total_q,
                fill_threshold=fill_threshold
            )
        latencies["stage3_segmentation_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # --- Stage 4: Grading & Final Color-Coded Annotation ---
        t0 = time.perf_counter()
        grading_result = self.grader.grade_student(student_id, answers)
        annotated_image = self._annotate_sheet_results(
            vis_warped, grading_result, ratios, fill_threshold
        )
        latencies["stage4_grading_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        latencies["total_pipeline_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

        return {
            "student_id": student_id,
            "mode": "Scratch (NumPy)" if use_scratch_mode else "OpenCV Baseline",
            "grading_result": grading_result,
            "corners": corners,
            "warped_gray": warped_gray,
            "annotated_image": annotated_image,
            "fill_ratios": ratios,
            "latencies": latencies
        }

    def _annotate_sheet_results(
        self,
        vis_image: np.ndarray,
        grading_result: Dict[str, Any],
        fill_ratios: Optional[Dict[int, Dict[str, float]]] = None,
        fill_threshold: float = 0.38
    ) -> np.ndarray:
        """Draw color-coded circles (Green=Correct, Red=Wrong, Orange=Blank/Multiple) and score banners."""
        img = vis_image.copy()
        h, w = img.shape[:2]
        total_questions = grading_result["total_questions"]
        questions_per_col = (total_questions + 1) // 2

        start_y = int(h * 0.16)
        end_y = int(h * 0.89)
        col1_x = int(w * 0.13)
        col2_x = int(w * 0.57)
        row_height = (end_y - start_y) / float(questions_per_col)
        bubble_radius = int(min(w, h) * 0.018)
        options = ["A", "B", "C", "D"]

        # Color palette (BGR format for OpenCV)
        COLOR_CORRECT = (40, 180, 40)     # Green
        COLOR_WRONG = (30, 30, 230)       # Red
        COLOR_WARNING = (0, 150, 255)     # Orange / Amber

        for item in grading_result["details"]:
            q = item["question"]
            stu_ans = item["student_ans"]
            cor_ans = item["correct_ans"]
            status = item["status"]

            if q <= questions_per_col:
                row_idx = q - 1
                base_x = col1_x
            else:
                row_idx = q - 1 - questions_per_col
                base_x = col2_x

            center_y = int(start_y + (row_idx + 0.5) * row_height)

            for opt_idx, opt in enumerate(options):
                center_x = int(base_x + (opt_idx + 1) * (bubble_radius * 2.8))

                if status == "CORRECT" and opt == stu_ans:
                    # Student answered correctly -> Bold Green Circle + checkmark badge
                    cv2.circle(img, (center_x, center_y), bubble_radius + 3, COLOR_CORRECT, 3)
                elif status == "INCORRECT":
                    if opt == stu_ans:
                        # Student's wrong choice -> Bold Red Circle
                        cv2.circle(img, (center_x, center_y), bubble_radius + 3, COLOR_WRONG, 3)
                    elif opt == cor_ans:
                        # Show true answer in Green Outline
                        cv2.circle(img, (center_x, center_y), bubble_radius + 3, COLOR_CORRECT, 2)
                elif status == "BLANK":
                    if opt == cor_ans:
                        # Highlight missed correct answer in Orange/Amber
                        cv2.circle(img, (center_x, center_y), bubble_radius + 3, COLOR_WARNING, 2)
                elif status == "MULTIPLE":
                    q_rat = fill_ratios.get(q, {}) if fill_ratios else {}
                    if q_rat.get(opt, 0.0) >= fill_threshold:
                        # Highlight all multiple-filled bubbles in Orange/Amber
                        cv2.circle(img, (center_x, center_y), bubble_radius + 3, COLOR_WARNING, 3)
                    if opt == cor_ans:
                        cv2.circle(img, (center_x, center_y), bubble_radius + 6, COLOR_CORRECT, 2)

            # Draw short status tag on the right side of the question row
            last_x = int(base_x + 4 * (bubble_radius * 2.8) + 24)
            if status == "CORRECT":
                cv2.putText(img, "DUNG", (last_x, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_CORRECT, 1)
            elif status == "INCORRECT":
                cv2.putText(img, f"SAI ({cor_ans})", (last_x, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WRONG, 1)
            elif status == "BLANK":
                cv2.putText(img, f"TRONG ({cor_ans})", (last_x, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WARNING, 1)
            elif status == "MULTIPLE":
                cv2.putText(img, f"LOI >1 ({cor_ans})", (last_x, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WARNING, 1)

        # Draw top banner (above fiducial border at y=62)
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
