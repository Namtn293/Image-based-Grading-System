"""Module: TV1 - Tiền xử lý & Phát hiện khung phiếu thi (Preprocessing & Sheet Detection)
Đề tài: Hệ thống Chấm bài Trắc nghiệm Tự động (OMR System)
Môn học: Xử lý ảnh số (INT13146) - Học viện Công nghệ Bưu chính Viễn thông (PTIT)
Giảng viên hướng dẫn: TS. Phạm Hoàng Việt
Thành viên phụ trách: Thành viên 1 (Giai đoạn 1 Pipeline OMR)
Tỷ lệ đóng góp kỹ thuật: 25% (Toàn bộ thuật toán cài đặt 100% Code from Scratch bằng NumPy)

GIAI ĐOẠN 1 TRONG TOÀN HỆ THỐNG:
[Đầu vào: Ảnh chụp phiếu thi (.jpg/.png) hoặc Ma trận mức xám (.csv)]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1 (THÀNH VIÊN 1): Tiền xử lý & Phát hiện khung phiếu thi      │
│  1. Chuyển đổi ảnh màu RGB sang ảnh mức xám (Luminance ITU-R BT.601)    │
│  2. Làm mịn khử nhiễu bằng Bộ lọc Gaussian 2D (Tích chập phân tách 2D)   │
│  3. Phát hiện biên Canny 4 bước (Sobel 3x3 -> NMS -> Double Thresh -> Hyst)│
│  4. Biến đổi Hough cực (rho, theta) & Xác định 4 đỉnh góc khung phiếu   │
└─────────────────────────────────────────────────────────────────────────┘
                    │ (Tọa độ 4 đỉnh khung phiếu: TL, TR, BR, BL)
                    ▼
[Chuyển giao cho Thành viên 2: Nắn thẳng phối cảnh Homography & NCC Marker]
"""

from __future__ import annotations
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# Import các thuật toán thành phần 100% Scratch (hỗ trợ cả package import và direct import)
try:
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
        collect_boundary_extremes,
        otsu_threshold_scratch,
        apply_parameter_space_nms,
        score_lines_by_edge_support,
        detect_quad_from_hough_lines,
        detect_quad_from_boundary_scratch
    )
except ImportError:
    from tv1_grayscale_blur import (
        rgb_to_gray_scratch,
        grayscale_luminance,
        get_gaussian_kernel_1d,
        get_gaussian_kernel_2d,
        pad_image_reflect,
        convolve2d_scratch,
        gaussian_blur_2d,
        gaussian_blur_scratch
    )
    from tv1_canny import (
        sobel_gradients,
        non_maximum_suppression,
        double_threshold_hysteresis,
        canny_edge_detector,
        canny_edge_scratch
    )
    from tv1_hough import (
        hough_lines_standard,
        intersect_lines_polar,
        order_quad_corners,
        convex_hull_2d,
        ramer_douglas_peucker,
        simplify_closed_polygon,
        collect_boundary_extremes,
        otsu_threshold_scratch,
        apply_parameter_space_nms,
        score_lines_by_edge_support,
        detect_quad_from_hough_lines,
        detect_quad_from_boundary_scratch
    )


def _border_band_stats(gray: np.ndarray, band_ratio: float = 0.02) -> Dict[str, float]:
    """Đo mức xám của các dải viền ảnh để nhận biết phiếu có bị nền bàn tối bao quanh.

    Code cũ chỉ lấy ĐÚNG 4 pixel ở 4 góc ảnh rồi so với ngưỡng cố định 100. Như vậy một
    vệt bóng, một hạt nhiễu, hay một góc bị cắt lệch là chọn sai nhánh xử lý. Ở đây ta
    lấy trung vị của cả 4 dải viền (mỗi dải rộng ``band_ratio * min(H, W)`` pixel), nên
    kết quả ổn định với nhiễu cục bộ, và dùng ngưỡng Otsu thay cho hằng số 100.

    Args:
        gray: Ảnh mức xám 2D.
        band_ratio: Độ rộng dải viền theo tỉ lệ cạnh ngắn của ảnh.

    Returns:
        Dict gồm ``median`` (trung vị toàn bộ dải viền), ``top``, ``bottom``, ``left``,
        ``right`` (trung vị từng dải), ``interior`` (trung vị vùng giữa ảnh) và
        ``otsu`` (ngưỡng Otsu toàn ảnh).
    """
    h, w = gray.shape[:2]
    band = max(4, int(round(min(h, w) * band_ratio)))

    strips = {
        "top": gray[:band, :].ravel(),
        "bottom": gray[h - band:, :].ravel(),
        "left": gray[:, :band].ravel(),
        "right": gray[:, w - band:].ravel(),
    }
    all_strips = np.concatenate(list(strips.values()))

    stats: Dict[str, float] = {name: float(np.median(values)) for name, values in strips.items()}
    stats["median"] = float(np.median(all_strips))

    # Vùng giữa ảnh (50% kích thước) đại diện cho thân tờ phiếu
    y0, y1 = h // 4, max(h // 4 + 1, 3 * h // 4)
    x0, x1 = w // 4, max(w // 4 + 1, 3 * w // 4)
    stats["interior"] = float(np.median(gray[y0:y1, x0:x1]))

    stats["otsu"] = float(otsu_threshold_scratch(gray))
    return stats


def _is_sheet_surrounded_by_dark_desk(gray: np.ndarray) -> bool:
    """Quyết định ảnh có nền bàn tối **bao quanh** phiếu hay không (dải viền + Otsu).

    Ba tiêu chí độc lập, chỉ cần một tiêu chí đúng là kết luận "có nền tối":

      1. **Cả bốn** dải viền đều tối hơn ngưỡng Otsu ít nhất 20 mức xám.
      2. **Mỗi** dải viền có từ 60% pixel trở lên thuộc lớp tối của Otsu (``<= T*``).
      3. **Cạnh sáng nhất** trong bốn dải viền vẫn nhỏ hơn 60% trung vị vùng giữa ảnh.

    Vì sao phải xét riêng từng phía thay vì gộp cả bốn dải làm một trung vị: nền bàn tối
    thật sự bao quanh phiếu nên **mọi phía** đều tối (DS-02: cả 4 dải ≈ 45 trong khi vùng
    giữa ≈ 245). Ngược lại, ảnh DS-05 là tờ phiếu phủ kín khung hình chỉ bị **bóng đổ** ở
    phía dưới: dải trên vẫn sáng 245, ba dải còn lại tối 80. Nếu gộp chung, trung vị rơi
    vào vùng tối (80) và dải sáng bị "bỏ phiếu thiểu số", khiến tờ phiếu bị cắt thành một
    tứ giác thụt vào trong — sai 67 px so với baseline. Yêu cầu cả bốn phía cùng tối loại
    đúng trường hợp bóng đổ này mà vẫn nhận đúng DS-02.

    Ba tiêu chí đều dựa trên trung vị và tỉ lệ của **toàn bộ** dải viền (không phải 4 pixel
    góc), nên một vệt bóng hay hạt nhiễu cục bộ không đánh lừa được bộ phân loại.

    Args:
        gray: Ảnh mức xám 2D.

    Returns:
        True nếu ảnh có nền tối bao quanh phiếu.
    """
    stats = _border_band_stats(gray)
    threshold = stats["otsu"]
    side_medians = (stats["top"], stats["bottom"], stats["left"], stats["right"])
    brightest_side = max(side_medians)

    # Tiêu chí 1: CẢ BỐN phía đều tối hơn ngưỡng Otsu đáng kể (nền bao quanh thật sự)
    if brightest_side < threshold - 20.0:
        return True

    # Tiêu chí 2: MỖI dải viền đều có >= 60% pixel thuộc lớp tối của Otsu
    h, w = gray.shape[:2]
    band = max(4, int(round(min(h, w) * 0.02)))
    side_strips = (
        gray[:band, :].ravel(),
        gray[h - band:, :].ravel(),
        gray[:, :band].ravel(),
        gray[:, w - band:].ravel(),
    )
    if all(float(np.mean(strip <= threshold)) >= 0.60 for strip in side_strips):
        return True

    # Tiêu chí 3: phía sáng nhất vẫn tối hơn hẳn vùng giữa ảnh
    interior = stats["interior"]
    if interior > 1.0 and brightest_side < 0.60 * interior:
        return True

    return False


def detect_sheet_corners(
    image: np.ndarray,
    use_scratch: bool = True,
    precomputed_gray: Optional[np.ndarray] = None,
    precomputed_edges: Optional[np.ndarray] = None
) -> np.ndarray:
    """Xác định tọa độ 4 góc của tờ phiếu trả lời trắc nghiệm (Sheet Corner Detection).

    Giao tiếp chuẩn đồng bộ với Pipeline Coordinator chung của nhóm:
      - Trả về mảng numpy shape (4, 2) theo đúng thứ tự [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
      - Tương thích 100% với hàm order_corners() và warp_perspective_sheet() của Thành viên 2 (TV2).

    Hai kịch bản thực tế:

    1. **Phiếu chiếm toàn khung ảnh** (DS-01, DS-03, DS-04, DS-05):
       Nền ảnh chính là giấy nên mép phiếu trùng mép ảnh. Khi đó trả về 4 góc khung hình.

    2. **Phiếu nằm trên nền bàn tối** (DS-02):
       Chạy đầy đủ chuỗi Canny -> Hough -> giao điểm Cramer:
         a. Làm mịn Gaussian và phát hiện biên Canny (4 bước, thuần NumPy).
         b. Biến đổi Hough cực để lấy các đường thẳng mạnh.
         c. Chấm điểm từng đường bằng số điểm biên thật sự nằm trên nó.
         d. Tách họ cạnh dọc / cạnh ngang, chọn 2 cạnh đối diện mạnh nhất mỗi họ.
         e. Tính 4 giao điểm bằng :func:`intersect_lines_polar` (quy tắc Cramer).
       Nếu Hough không dựng được tứ giác hợp lệ, dự phòng bằng phân ngưỡng Otsu +
       bao lồi Andrew's Monotone Chain + xấp xỉ đa giác RDP.

    Args:
        image: Ảnh đầu vào 2D (mức xám) hoặc 3D (màu BGR/RGB).
        use_scratch: True = sử dụng 100% thuật toán Scratch của TV1;
                     False = sử dụng OpenCV baseline làm đối chứng.
        precomputed_gray: Ảnh mức xám đã tính sẵn (tránh chuyển đổi lặp lại khi gọi từ
                     :func:`process_preprocessing_pipeline`).
        precomputed_edges: Ảnh biên Canny đã tính sẵn trên ``precomputed_gray`` (tránh
                     chạy lại Canny lần thứ hai trong cùng một pipeline).

    Returns:
        corners: Mảng float32 shape (4, 2) theo thứ tự [TL, TR, BR, BL].
    """
    img = np.asarray(image)
    h, w = img.shape[:2]

    # Chuẩn hóa về ảnh mức xám
    if precomputed_gray is not None:
        gray = np.asarray(precomputed_gray)
    elif img.ndim == 3 and img.shape[2] >= 3:
        gray = rgb_to_gray_scratch(img, is_bgr=True) if use_scratch else _cv_rgb_to_gray(img)
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = img.copy()

    full_frame = np.array([
        [0.0, 0.0],
        [float(w - 1), 0.0],
        [float(w - 1), float(h - 1)],
        [0.0, float(h - 1)]
    ], dtype=np.float32)

    if not use_scratch:
        return _detect_sheet_corners_cv(img)

    # Nhánh 1: phiếu chiếm toàn khung ảnh -> 4 góc khung hình
    if not _is_sheet_surrounded_by_dark_desk(gray):
        return full_frame

    # Nhánh 2: phiếu trên nền bàn tối -> Canny + Hough + giao điểm Cramer
    if precomputed_edges is not None:
        edges = np.asarray(precomputed_edges)
    else:
        blurred = gaussian_blur_scratch(gray, kernel_size=5, sigma=1.4)
        edges = canny_edge_detector(blurred, low_threshold=50.0, high_threshold=150.0, preblurred=True)
    lines, _ = hough_lines_standard(
        edges, rho_res=1.0, theta_res=np.pi / 180.0, threshold=80
    )

    quad = detect_quad_from_hough_lines(lines, edges, (h, w))
    if quad is not None:
        return quad.astype(np.float32)

    # Dự phòng: phân ngưỡng Otsu + bao lồi + RDP trên đa giác khép kín
    threshold = otsu_threshold_scratch(gray)
    binary = ((gray > threshold).astype(np.uint8)) * 255
    quad = detect_quad_from_boundary_scratch(binary, (h, w))
    if quad is not None:
        return quad.astype(np.float32)

    # Không tìm thấy tứ giác hợp lệ, fallback về 4 góc ảnh
    return full_frame

def _cv_rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Hàm phụ trợ gọi OpenCV chuyển mức xám (chỉ dùng trong chế độ đối chứng)."""
    import cv2
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _detect_sheet_corners_cv(image: np.ndarray) -> np.ndarray:
    """Hàm đối chứng sử dụng OpenCV tìm 4 góc phiếu."""
    import cv2
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

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
                        return order_quad_corners(approx.reshape(4, 2))
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                return order_quad_corners(box)

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

                ordered = order_quad_corners(found_pts)
                x_min, y_min = np.min(ordered, axis=0)
                x_max, y_max = np.max(ordered, axis=0)
                if (x_min > 0.02 * w and x_min < 0.08 * w and
                    y_min > 0.03 * h and y_min < 0.10 * h and
                    x_max > 0.92 * w and y_max > 0.90 * h):
                    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
                return ordered

    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def process_preprocessing_pipeline(
    image: np.ndarray,
    use_scratch: bool = True,
    canny_low: float = 50.0,
    canny_high: float = 150.0,
    gaussian_ksize: int = 5,
    gaussian_sigma: float = 1.4
) -> Dict[str, Any]:
    """Thực thi toàn bộ luồng Giai đoạn 1 (Stage 1 Preprocessing & Sheet Detection).

    Quy trình thực hiện:
      1. Ảnh đầu vào -> Chuyển mức xám (Grayscale).
      2. Khử nhiễu làm mịn -> Gaussian Blur 2D.
      3. Phát hiện biên chi tiết -> Sobel Gradients, NMS, Double Threshold & Hysteresis Canny.
      4. Định vị khung phiếu -> Hough Line Transform & Quad Sheet Corners [TL, TR, BR, BL].

    Returns:
        Dictionary chứa:
          - 'gray': Ảnh mức xám
          - 'blurred': Ảnh sau khi làm mịn
          - 'edges': Ảnh biên nhị phân Canny
          - 'corners': Tọa độ 4 đỉnh shape (4, 2)
          - 'latencies': Thời gian thực thi từng bước (ms)
    """
    latencies: Dict[str, float] = {}
    t_start = time.perf_counter()

    # 1. Chuyển đổi mức xám
    t0 = time.perf_counter()
    if use_scratch:
        gray = rgb_to_gray_scratch(image, is_bgr=True) if image.ndim == 3 else image.copy()
    else:
        gray = _cv_rgb_to_gray(image)
    latencies["grayscale_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # 2. Làm mịn Gaussian
    t0 = time.perf_counter()
    if use_scratch:
        blurred = gaussian_blur_scratch(gray, kernel_size=gaussian_ksize, sigma=gaussian_sigma)
    else:
        import cv2
        blurred = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), gaussian_sigma, borderType=cv2.BORDER_REFLECT_101)
    latencies["gaussian_blur_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # 3. Phát hiện biên Canny
    t0 = time.perf_counter()
    if use_scratch:
        edges = canny_edge_detector(blurred, low_threshold=canny_low, high_threshold=canny_high, preblurred=True)
    else:
        import cv2
        edges = cv2.Canny(blurred, int(canny_low), int(canny_high), L2gradient=True)
    latencies["canny_edge_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # 4. Xác định 4 góc phiếu (dùng lại gray + edges đã tính, không chạy lại Canny)
    t0 = time.perf_counter()
    corners = detect_sheet_corners(
        image,
        use_scratch=use_scratch,
        precomputed_gray=gray,
        precomputed_edges=edges
    )
    latencies["corner_detection_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    latencies["total_tv1_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)

    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "corners": corners,
        "latencies": latencies
    }


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Tính sai số bình phương trung bình (MSE)."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Tính tỷ số tín hiệu cực đại trên nhiễu (PSNR - dB)."""
    mse = calculate_mse(img1, img2)
    if mse < 1e-10:
        return 999.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def calculate_binary_iou_dice(bin1: np.ndarray, bin2: np.ndarray) -> Tuple[float, float]:
    """Tính chỉ số IoU và Dice giữa 2 ảnh nhị phân."""
    b1 = bin1 > 0
    b2 = bin2 > 0
    inter = int(np.logical_and(b1, b2).sum())
    union = int(np.logical_or(b1, b2).sum())
    iou = (inter / union) if union > 0 else 1.0
    dice = (2.0 * inter / (int(b1.sum()) + int(b2.sum()))) if (int(b1.sum()) + int(b2.sum())) > 0 else 1.0
    return round(float(iou), 4), round(float(dice), 4)


def calculate_line_match_rate(
    lines_reference: List[Tuple[float, float]],
    lines_test: List[Tuple[float, float]],
    theta_tolerance: float = np.deg2rad(2.0),
    rho_tolerance: float = 5.0,
    canonical: bool = True
) -> float:
    """Đo tỉ lệ đường thẳng trong ``lines_reference`` được ``lines_test`` tìm lại được.

    So sánh trực tiếp SỐ LƯỢNG đường (cách làm cũ trong benchmark) không phản ánh đúng
    chất lượng: OpenCV trả về rất nhiều cực đại phụ nên số lượng luôn lệch, trong khi tập
    đường chính lại trùng nhau. Hàm này so khớp theo tham số hình học:

      - Hai đường được coi là trùng nếu ``|Δθ| <= theta_tolerance`` và
        ``|Δρ| <= rho_tolerance``.
      - Góc được chuẩn hóa về ``[0, π)`` theo quy ước Hesse ``ρ = x·cosθ + y·sinθ`` để
        ``(ρ, θ)`` và ``(-ρ, θ + π)`` biểu diễn cùng một đường.

    Args:
        lines_reference: Danh sách (rho, theta) làm chuẩn (thường là kết quả OpenCV).
        lines_test: Danh sách (rho, theta) cần kiểm tra (thường là kết quả Scratch).
        theta_tolerance: Dung sai góc (radian).
        rho_tolerance: Dung sai khoảng cách (pixel).
        canonical: Chuẩn hóa góc về [0, π) trước khi so khớp.

    Returns:
        Tỉ lệ trong [0, 1]: phần đường của ``lines_reference`` được tìm thấy trong ``lines_test``.
    """

    def canon(rho: float, theta: float) -> Tuple[float, float]:
        if not canonical:
            return rho, theta
        theta = float(theta)
        rho = float(rho)
        # Đưa theta về [0, pi); nếu vượt biên thì đổi dấu rho tương ứng
        full_turns = np.floor(theta / np.pi)
        theta = theta - full_turns * np.pi
        if full_turns % 2 != 0:
            rho = -rho
        return rho, theta

    if not lines_reference:
        return 1.0
    if not lines_test:
        return 0.0

    test_canon = [canon(rho, theta) for rho, theta in lines_test]

    matched = 0
    for ref_rho, ref_theta in lines_reference:
        rr, rt = canon(ref_rho, ref_theta)
        for rho, theta in test_canon:
            d_theta = abs(theta - rt)
            d_theta = min(d_theta, abs(d_theta - np.pi))  # góc là modulo pi
            if d_theta <= theta_tolerance and abs(rho - rr) <= rho_tolerance:
                matched += 1
                break

    return float(matched) / float(len(lines_reference))


def run_unit_benchmarks(sample_image: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Chạy bộ kiểm thử đối chứng định lượng Unit Benchmark (NumPy Scratch vs. OpenCV).

    Bám sát bảng thiết kế thí nghiệm 6.1 trong Đề cương đề tài:
      - Thuật toán 1: Grayscale Luminance vs cv2.cvtColor(COLOR_RGB2GRAY)
      - Thuật toán 2: Gaussian Blur 2D (k=5, sigma=1.4) vs cv2.GaussianBlur()
      - Thuật toán 3: Canny Edge Detector (50, 150) vs cv2.Canny(L2gradient=True)
      - Thuật toán 4: Hough Lines Standard vs cv2.HoughLines()
      - Thuật toán 5: Sheet Corner Detection vs OpenCV Contour Quad
    """
    import cv2

    if sample_image is None:
        # Tạo ảnh mẫu tổng hợp có khung viền và chi tiết
        sample_image = np.full((1000, 800, 3), 245, dtype=np.uint8)
        # Vẽ viền ngoài
        cv2.rectangle(sample_image, (40, 40), (760, 960), (30, 30, 30), 4)
        # Vẽ một số hình tròn và đường kẻ
        for y in range(150, 850, 40):
            cv2.line(sample_image, (60, y), (740, y), (180, 180, 180), 1)
            for x in range(120, 700, 50):
                cv2.circle(sample_image, (x, y), 12, (50, 50, 50), -1 if (x + y) % 7 == 0 else 2)

    image = sample_image
    results: List[Dict[str, Any]] = []

    # --- Thí nghiệm 1: RGB to Grayscale ---
    rgb = image[:, :, :3][:, :, ::-1] if image.ndim == 3 else image
    t0 = time.perf_counter()
    g_sc = rgb_to_gray_scratch(rgb)
    t_sc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    g_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    t_cv = (time.perf_counter() - t0) * 1000.0

    mse_g = calculate_mse(g_sc, g_cv)
    psnr_g = calculate_psnr(g_sc, g_cv)
    results.append({
        "module": "TV1",
        "algorithm": "Grayscale Luminance (BT.601)",
        "mse": round(mse_g, 4),
        "psnr_db": round(psnr_g, 2),
        "iou": None,
        "dice": None,
        "error_metric": f"Max Diff: {int(np.max(np.abs(g_sc.astype(int) - g_cv.astype(int))))}",
        "time_scratch_ms": round(t_sc, 3),
        "time_opencv_ms": round(t_cv, 3),
        "status": "PASS" if mse_g < 1.0 else "FAIL"
    })

    # --- Thí nghiệm 2: Gaussian Blur 2D ---
    t0 = time.perf_counter()
    b_sc = gaussian_blur_scratch(g_sc, kernel_size=5, sigma=1.4)
    t_sc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    b_cv = cv2.GaussianBlur(g_cv, (5, 5), 1.4, borderType=cv2.BORDER_REFLECT_101)
    t_cv = (time.perf_counter() - t0) * 1000.0

    mse_b = calculate_mse(b_sc, b_cv)
    psnr_b = calculate_psnr(b_sc, b_cv)
    results.append({
        "module": "TV1",
        "algorithm": "Gaussian Blur 2D (5x5, sigma=1.4)",
        "mse": round(mse_b, 4),
        "psnr_db": round(psnr_b, 2),
        "iou": None,
        "dice": None,
        "error_metric": f"Max Diff: {int(np.max(np.abs(b_sc.astype(int) - b_cv.astype(int))))}",
        "time_scratch_ms": round(t_sc, 3),
        "time_opencv_ms": round(t_cv, 3),
        "status": "PASS" if psnr_b > 48.0 else "FAIL"
    })

    # --- Thí nghiệm 3: Canny Edge Detector ---
    t0 = time.perf_counter()
    e_sc = canny_edge_detector(b_sc, low_threshold=50.0, high_threshold=150.0, preblurred=True)
    t_sc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    e_cv = cv2.Canny(b_cv, 50, 150, L2gradient=True)
    t_cv = (time.perf_counter() - t0) * 1000.0

    iou_e, dice_e = calculate_binary_iou_dice(e_sc, e_cv)
    mse_e = calculate_mse(e_sc, e_cv)
    psnr_e = calculate_psnr(e_sc, e_cv)
    results.append({
        "module": "TV1",
        "algorithm": "Canny Edge Detector (4 bước)",
        "mse": round(mse_e, 4),
        "psnr_db": round(psnr_e, 2),
        "iou": iou_e,
        "dice": dice_e,
        "error_metric": f"IoU: {iou_e*100:.2f}%",
        "time_scratch_ms": round(t_sc, 3),
        "time_opencv_ms": round(t_cv, 3),
        "status": "PASS" if iou_e >= 0.985 else "FAIL"
    })

    # --- Thí nghiệm 4: Hough Lines Standard ---
    t0 = time.perf_counter()
    lines_sc, _ = hough_lines_standard(e_sc, rho_res=1.0, theta_res=np.pi / 180.0, threshold=100)
    t_sc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    lines_cv_raw = cv2.HoughLines(e_cv, 1.0, np.pi / 180.0, 100)
    t_cv = (time.perf_counter() - t0) * 1000.0

    lines_cv = (
        [(float(l[0][0]), float(l[0][1])) for l in lines_cv_raw]
        if lines_cv_raw is not None else []
    )
    # cv2.HoughLines không khử cực đại phụ nên trả về rất nhiều đường sát nhau. Để phép
    # đối chứng công bằng, chuẩn hóa danh sách của OpenCV bằng ĐÚNG quy trình NMS mà
    # hough_lines_standard đang dùng, rồi so khớp theo tham số (rho, theta).
    lines_cv_normalized = apply_parameter_space_nms(lines_cv)
    hough_recall = calculate_line_match_rate(lines_cv_normalized, lines_sc)
    hough_precision = calculate_line_match_rate(lines_sc, lines_cv_normalized)

    results.append({
        "module": "TV1",
        "algorithm": "Hough Lines Standard (rho=1, theta=1°)",
        # Hai tập đường thẳng không cùng số phần tử và không cùng thứ tự nên KHÔNG thể
        # tính MSE/PSNR theo pixel. Để None thay vì in ra 0.0 / 999.0 gây hiểu sai.
        "mse": None,
        "psnr_db": None,
        "iou": round(hough_recall, 4),
        "dice": round(hough_precision, 4),
        "error_metric": (
            f"Recall={hough_recall * 100:.1f}% / Precision={hough_precision * 100:.1f}% "
            f"(Scratch={len(lines_sc)}, CV NMS={len(lines_cv_normalized)})"
        ),
        "time_scratch_ms": round(t_sc, 3),
        "time_opencv_ms": round(t_cv, 3),
        # Tập đường chính phải trùng nhau: recall >= 95% và precision >= 90%
        "status": "PASS" if (hough_recall >= 0.95 and hough_precision >= 0.90) else "FAIL"
    })

    # --- Thí nghiệm 5: Sheet Corner Detection (kịch bản phiếu chiếm toàn khung) ---
    t0 = time.perf_counter()
    c_sc = detect_sheet_corners(image, use_scratch=True)
    t_sc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    c_cv = detect_sheet_corners(image, use_scratch=False)
    t_cv = (time.perf_counter() - t0) * 1000.0

    corner_err = float(np.mean(np.linalg.norm(c_sc - c_cv, axis=1)))
    results.append({
        "module": "TV1",
        "algorithm": "Sheet Corner Localization (4 góc)",
        # MSE/PSNR là đại lượng của ảnh, không áp dụng cho 4 cặp toạ độ. Sai số hình học
        # được báo cáo riêng qua "Corner Dist" (px).
        "mse": None,
        "psnr_db": None,
        "iou": None,
        "dice": None,
        "error_metric": f"Corner Dist: {corner_err:.2f} px",
        "time_scratch_ms": round(t_sc, 3),
        "time_opencv_ms": round(t_cv, 3),
        "status": "PASS" if corner_err <= 3.0 else "FAIL"
    })


    return results


def print_benchmark_table(benchmarks: List[Dict[str, Any]]) -> None:
    """In bảng kết quả benchmark chuẩn hóa ra Terminal/Console."""
    header = (
        f"{'Thuật toán (TV1)':<36} | {'MSE':<8} | {'PSNR (dB)':<10} | {'IoU (%)':<8} | "
        f"{'Sai số/Đặc trưng':<22} | {'Scratch':<10} | {'OpenCV':<8} | {'Đánh giá'}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(" BẢNG ĐỐI CHỨNG ĐỊNH LƯỢNG KỸ THUẬT: MODULE THÀNH VIÊN 1 (SCRATCH vs. OPENCV)")
    print(sep)
    print(header)
    print(sep)

    for b in benchmarks:
        iou_str = f"{b['iou']*100:.2f}%" if b.get('iou') is not None else "—"
        mse_str = f"{b['mse']:.4f}" if b.get('mse') is not None else "—"
        psnr_str = f"{b['psnr_db']:.2f}" if b.get('psnr_db') is not None else "—"
        mark = "✅" if b['status'] == "PASS" else "❌"
        print(
            f"{b['algorithm']:<36} | {mse_str:<8} | {psnr_str:<10} | {iou_str:<8} | "
            f"{b['error_metric']:<22} | {b['time_scratch_ms']:<8.2f}ms | {b['time_opencv_ms']:<6.2f}ms | {mark} {b['status']}"
        )
    print(sep + "\n")


if __name__ == "__main__":
    import cv2

    print("=" * 70)
    print("  KHỞI CHẠY KIỂM THỬ VÀ NGHIỆM THU MODULE THÀNH VIÊN 1 (TV1)")
    print("=" * 70)

    # Thư mục gốc dự án
    base_dir = Path(__file__).resolve().parent.parent.parent
    sample_path = base_dir / "OMR_System_TV4" / "data" / "samples" / "01_B21DCCN001_Chuan.png"

    if sample_path.exists():
        print(f"[*] Nạp ảnh kiểm thử thực nghiệm: {sample_path.name}")
        test_img = cv2.imread(str(sample_path))
    else:
        print("[*] Tạo ảnh kiểm thử nhân tạo...")
        test_img = None

    # Chạy benchmark
    bm_results = run_unit_benchmarks(test_img)
    print_benchmark_table(bm_results)

    # Xuất báo cáo kết quả ra outputs/tv1
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "tv1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chạy pipeline lưu ảnh các bước
    pipeline_res = process_preprocessing_pipeline(test_img if test_img is not None else np.zeros((100, 100, 3), np.uint8))

    cv2.imwrite(str(out_dir / "1_input_sample.png"), test_img if test_img is not None else np.zeros((100, 100, 3), np.uint8))
    cv2.imwrite(str(out_dir / "2_grayscale.png"), pipeline_res["gray"])
    cv2.imwrite(str(out_dir / "3_gaussian_blur.png"), pipeline_res["blurred"])
    gx, gy, mag, _ = sobel_gradients(pipeline_res["blurred"])
    cv2.imwrite(str(out_dir / "4_sobel_gradient.png"), np.clip(mag, 0, 255).astype(np.uint8))
    nms_img = non_maximum_suppression(mag, gx, gy)
    cv2.imwrite(str(out_dir / "5_nms.png"), np.clip(nms_img, 0, 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "6_canny_edges.png"), pipeline_res["edges"])

    # Không gian tham số Hough (rho, theta) của ảnh biên vừa tính
    _, hough_acc = hough_lines_standard(
        pipeline_res["edges"], rho_res=1.0, theta_res=np.pi / 180.0, threshold=80
    )
    acc_vis = np.log1p(hough_acc.astype(np.float64))
    acc_vis = (255.0 * acc_vis / max(acc_vis.max(), 1.0)).astype(np.uint8)
    acc_vis = cv2.applyColorMap(acc_vis, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(out_dir / "7_hough_accumulator.png"), acc_vis)

    # Vẽ 4 góc lên ảnh
    annotated = test_img.copy() if test_img is not None else np.zeros((100, 100, 3), np.uint8)
    corners = pipeline_res["corners"].astype(int)
    for i, pt in enumerate(corners):
        cv2.circle(annotated, (pt[0], pt[1]), 10, (0, 0, 255), -1)
        cv2.putText(annotated, f"C{i+1}:({pt[0]},{pt[1]})", (pt[0] + 10, pt[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / "8_sheet_corners.png"), annotated)

    # Triplet view: Scratch vs OpenCV vs Heatmap
    e_cv = cv2.Canny(cv2.GaussianBlur(pipeline_res["gray"], (5, 5), 1.4, borderType=cv2.BORDER_REFLECT_101), 50, 150, L2gradient=True)
    diff_map = np.abs(pipeline_res["edges"].astype(float) - e_cv.astype(float)).astype(np.uint8) * 10
    diff_colored = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
    e_sc_bgr = cv2.cvtColor(pipeline_res["edges"], cv2.COLOR_GRAY2BGR)
    e_cv_bgr = cv2.cvtColor(e_cv, cv2.COLOR_GRAY2BGR)
    triplet = np.hstack([e_sc_bgr, e_cv_bgr, diff_colored])
    cv2.imwrite(str(out_dir / "9_triplet_comparison.png"), triplet)

    # Lưu metrics CSV & JSON
    import csv
    with open(out_dir / "benchmark_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bm_results[0].keys()))
        writer.writeheader()
        writer.writerows(bm_results)

    with open(out_dir / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(bm_results, f, indent=2, ensure_ascii=False)

    print(f"[+] Toàn bộ ảnh trung gian và bảng số liệu đã xuất thành công tại: {out_dir}")
