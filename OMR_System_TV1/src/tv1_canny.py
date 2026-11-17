"""Module: TV1 - Phát hiện biên Canny (Canny Edge Detector from Scratch)
Đề tài: Hệ thống Chấm bài Trắc nghiệm Tự động (OMR System)
Môn học: Xử lý ảnh số (INT13146) - Học viện Công nghệ Bưu chính Viễn thông (PTIT)
Giảng viên hướng dẫn: TS. Phạm Hoàng Việt
Thành viên phụ trách: Thành viên 1 (Tiền xử lý & Phát hiện khung phiếu)

YÊU CẦU BẮT BUỘC:
- 100% Cài đặt từ đầu (Code from Scratch) bằng NumPy qua 4 bước chuẩn giáo trình:
    Bước 1: Tính đạo hàm bằng toán tử Sobel 3x3 (Gx, Gy) & độ lớn Gradient Euclidean L2
    Bước 2: Triệt tiêu không cực đại (Non-Maximum Suppression - NMS) 4 hướng góc
    Bước 3: Phân ngưỡng kép (Double Thresholding: T_high và T_low)
    Bước 4: Nối biên trễ bằng liên thông 8 hướng (Hysteresis Edge Tracking via BFS)
- Đạt độ trùng khớp IoU >= 98.5% so với hàm cv2.Canny(L2gradient=True) của OpenCV.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    from .tv1_grayscale_blur import (
        rgb_to_gray_scratch,
        gaussian_blur_2d,
        pad_image_reflect
    )
except ImportError:
    from tv1_grayscale_blur import (
        rgb_to_gray_scratch,
        gaussian_blur_2d,
        pad_image_reflect
    )


def sobel_gradients(
    image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bước 1: Tính đạo hàm bậc nhất không gian bằng toán tử Sobel 3x3 và độ lớn Gradient L2.

    Cơ sở toán học:
      Toán tử Sobel xấp xỉ đạo hàm riêng theo trục x (ngang) và y (dọc):
        K_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        K_y = [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]]   (Quy ước trục y hướng xuống)

      Độ lớn Gradient Euclidean L2:
        M(x, y) = sqrt( G_x(x, y)^2 + G_y(x, y)^2 )

      Hướng Gradient:
        theta(x, y) = atan2( G_y(x, y), G_x(x, y) )

    Args:
        image: Ảnh mức xám 2D (H, W) đã làm mịn Gaussian.

    Returns:
        gx: Mảng float64 đạo hàm theo trục X (H, W).
        gy: Mảng float64 đạo hàm theo trục Y (H, W).
        magnitude: Mảng float64 độ lớn Gradient L2 (H, W).
        theta: Mảng float64 hướng gradient (radian) (H, W).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"sobel_gradients yêu cầu ảnh 2D, nhận được shape {img.shape}")

    # Đệm biên đối xứng (Reflect 101) 1 pixel: fedcba | abcdefgh | hgfedc
    padded = pad_image_reflect(img, 1)

    # Trích xuất 9 lát (slice) của cửa sổ 3x3. Cách này cho kết quả trùng khít
    # với sliding_window_view nhưng không tạo mảng ảo (h, w, 3, 3) nên nhanh hơn ~2.4 lần.
    a = padded[0:-2, 0:-2]   # hàng trên, cột trái
    b = padded[0:-2, 1:-1]   # hàng trên, cột giữa
    c = padded[0:-2, 2:]     # hàng trên, cột phải
    d = padded[1:-1, 0:-2]   # hàng giữa, cột trái
    f = padded[1:-1, 2:]     # hàng giữa, cột phải
    g = padded[2:, 0:-2]     # hàng dưới, cột trái
    h_row = padded[2:, 1:-1] # hàng dưới, cột giữa
    i = padded[2:, 2:]       # hàng dưới, cột phải

    # K_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  (hệ số ±2 nằm ở hàng giữa)
    # K_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]  (quy ước trục y hướng xuống)
    gx = (c + 2.0 * f + i) - (a + 2.0 * d + g)
    gy = (g + 2.0 * h_row + i) - (a + 2.0 * b + c)

    # Độ lớn Gradient L2 chuẩn
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)

    return gx, gy, magnitude, theta


def non_maximum_suppression(
    magnitude: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    low_threshold: float = 50.0
) -> np.ndarray:
    """Bước 2: Triệt tiêu không cực đại (Non-Maximum Suppression - NMS).

    Cơ sở toán học:
      Để làm mảnh biên về độ dày 1 pixel, tại mỗi điểm ảnh (r, c), ta so sánh độ lớn
      gradient M(r, c) với 2 điểm lân cận nằm dọc theo phương trực giao với đường biên
      (tức phương của vector Gradient).
      Góc theta được lượng tử hóa thành 4 khu vực chính:
        1. Sector 0 (Hướng ngang, biên dọc): |dy| <= tan(22.5°) * |dx| ≈ 0.4142 * |dx|
           -> So sánh với 2 điểm lân cận bên Trái (r, c-1) và bên Phải (r, c+1).
        2. Sector 2 (Hướng dọc, biên ngang): |dy| >= tan(67.5°) * |dx| ≈ 2.4142 * |dx|
           -> So sánh với 2 điểm lân cận phía Trên (r-1, c) và phía Dưới (r+1, c).
        3. Sector 1 & 3 (Hướng đường chéo):
           - Nếu gx * gy > 0: Góc chéo chính (135° / 45° cùng dấu)
             -> So sánh với góc Trên-Trái (r-1, c-1) và Dưới-Phải (r+1, c+1).
           - Nếu gx * gy < 0: Góc chéo phụ
             -> So sánh với góc Trên-Phải (r-1, c+1) và Dưới-Trái (r+1, c-1).

      Quy ước bất đẳng thức (Strict/Non-strict) chuẩn Canny của OpenCV:
        Sử dụng một bên nghiêm ngặt (m > n1) và một bên không nghiêm ngặt (m >= n2)
        để triệt tiêu hoàn toàn trường hợp sườn dốc bằng phẳng sinh biên kép.

    Args:
        magnitude: Ma trận độ lớn gradient M(r, c) float64.
        gx: Ma trận đạo hàm trục X float64.
        gy: Ma trận đạo hàm trục Y float64.
        low_threshold: Ngưỡng lọc sơ bộ (điểm ảnh < low_threshold bị loại ngay).

    Returns:
        Ma trận nms float64 cùng kích thước, chỉ giữ lại các điểm cực đại cục bộ.
    """
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float64)

    # Phân loại 4 khu vực góc
    abs_dx = np.abs(gx)
    abs_dy = np.abs(gy)

    sec0 = (abs_dy <= 0.41421356 * abs_dx)
    sec2 = (abs_dy >= 2.41421356 * abs_dx)
    diag = ~(sec0 | sec2)
    sec_pos = diag & ((gx * gy) > 0)
    sec_neg = diag & ((gx * gy) < 0)

    # Vùng lõi (bỏ viền 1 pixel để so sánh 8 lân cận)
    m = magnitude[1:-1, 1:-1]
    s0 = sec0[1:-1, 1:-1]
    s2 = sec2[1:-1, 1:-1]
    sp = sec_pos[1:-1, 1:-1]
    sn = sec_neg[1:-1, 1:-1]

    # Trích xuất 8 lát lân cận vector hóa
    left   = magnitude[1:-1, :-2]
    right  = magnitude[1:-1, 2:]
    top    = magnitude[:-2, 1:-1]
    bottom = magnitude[2:, 1:-1]
    tl     = magnitude[:-2, :-2]
    br     = magnitude[2:, 2:]
    tr     = magnitude[:-2, 2:]
    bl     = magnitude[2:, :-2]

    # Kiểm tra điều kiện cực đại cục bộ
    keep0 = s0 & (m > left)  & (m >= right)
    keep2 = s2 & (m > top)   & (m >= bottom)
    keepp = sp & (m > tl)    & (m >= br)
    keepn = sn & (m > tr)    & (m >= bl)

    keep = (keep0 | keep2 | keepp | keepn) & (m >= low_threshold)

    nms_core = np.zeros_like(m)
    nms_core[keep] = m[keep]
    nms[1:-1, 1:-1] = nms_core

    return nms


def double_threshold_hysteresis(
    nms_image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0
) -> np.ndarray:
    """Bước 3 & 4: Phân ngưỡng kép (Double Thresholding) và Nối biên trễ (Hysteresis Tracking).

    Cơ sở toán học:
      1. Phân loại 3 mức:
         - Điểm biên mạnh (Strong Edge): M_nms >= high_threshold (chắc chắn là biên) -> gán 255.
         - Điểm biên yếu (Weak Edge): low_threshold <= M_nms < high_threshold (tiềm năng là biên).
         - Điểm loại bỏ (Non-edge): M_nms < low_threshold -> gán 0.
      2. Nối biên trễ (Hysteresis) tương đương tìm kiếm lan truyền 8-lân cận:
         - Xuất phát từ toàn bộ điểm biên mạnh.
         - Lặp: mọi điểm biên yếu có ít nhất 1 lân cận 8 hướng đã được kết nạp sẽ được kết nạp.
         - Dừng khi không còn điểm nào thay đổi; điểm biên yếu cô lập bị xóa về 0.
         Phép lặp này chính là phép tái tạo hình thái học (morphological reconstruction)
         trên mặt nạ weak_mask, cho kết quả trùng khít với duyệt BFS từng điểm nhưng
         vector hóa hoàn toàn nên không có chi phí vòng lặp Python theo từng pixel.

    Args:
        nms_image: Ma trận NMS float64 (H, W).
        low_threshold: Ngưỡng biên yếu T_low.
        high_threshold: Ngưỡng biên mạnh T_high.

    Returns:
        Ma trận ảnh nhị phân uint8 (H, W) với các điểm biên mang giá trị 255, nền 0.
    """
    h, w = nms_image.shape
    edges = np.zeros((h, w), dtype=np.uint8)

    # Phân loại 3 mức trạng thái bằng mặt nạ boolean
    strong_mask = (nms_image >= high_threshold)
    weak_mask = (nms_image >= low_threshold)

    # Không có điểm biên mạnh nào -> toàn bộ điểm yếu cô lập bị loại bỏ
    if not np.any(strong_mask):
        return edges

    # Lan truyền 8-lân cận bằng phép giãn nhị phân lặp (morphological reconstruction).
    # Mỗi vòng lặp: hợp của 8 bản dịch (shift) của mặt nạ hiện tại, giao với weak_mask.
    # Kết quả trùng khít với BFS từng điểm nhưng chạy vector hóa trên toàn ảnh
    # (nhanh hơn ~10 lần so với deque Python ở ảnh 1000x800).
    shifts = ((-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1))

    keep = strong_mask.copy()
    while True:
        grown = keep.copy()
        for dr, dc in shifts:
            src_rows = slice(max(0, -dr), h - max(0, dr))
            src_cols = slice(max(0, -dc), w - max(0, dc))
            dst_rows = slice(max(0, dr), h - max(0, -dr))
            dst_cols = slice(max(0, dc), w - max(0, -dc))
            grown[dst_rows, dst_cols] |= keep[src_rows, src_cols]
        grown &= weak_mask
        if np.array_equal(grown, keep):
            break
        keep = grown

    edges[keep] = 255
    return edges


def canny_edge_detector(
    image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    kernel_size: int = 5,
    sigma: float = 1.4,
    preblurred: bool = False
) -> np.ndarray:
    """Thuật toán Phát hiện biên Canny hoàn chỉnh 4 bước (Code from Scratch bằng NumPy).

    Quy trình 4 giai đoạn chuẩn:
      1. Làm mịn khử nhiễu bằng Gaussian Blur 2D (kernel_size, sigma) nếu preblurred=False.
      2. Tính đạo hàm bậc nhất bằng toán tử Sobel 3x3 và tính độ lớn Gradient L2.
      3. Triệt tiêu không cực đại (NMS) vector hóa 4 hướng.
      4. Phân ngưỡng kép và Nối biên trễ bằng tìm kiếm theo chiều rộng (BFS Hysteresis).

    Args:
        image: Ảnh đầu vào 2D mức xám (hoặc ảnh màu BGR/RGB sẽ tự chuyển mức xám).
        low_threshold: Ngưỡng biên yếu (mặc định 50.0).
        high_threshold: Ngưỡng biên mạnh (mặc định 150.0).
        kernel_size: Kích thước kernel Gaussian làm mịn (mặc định 5).
        sigma: Độ lệch chuẩn Gaussian (mặc định 1.4).
        preblurred: Nếu True, bỏ qua bước làm mịn Gaussian (khi ảnh đầu vào đã làm mịn).

    Returns:
        Ảnh nhị phân biên uint8 shape (H, W) mang giá trị 0 hoặc 255.
    """
    img = np.asarray(image)
    if img.ndim != 2:
        img = rgb_to_gray_scratch(img)

    # Bước 1: Làm mịn Gaussian
    if not preblurred and kernel_size > 1:
        blurred = gaussian_blur_2d(img, kernel_size=kernel_size, sigma=sigma)
    else:
        blurred = img

    # Bước 2: Gradient Sobel 3x3
    gx, gy, magnitude, _ = sobel_gradients(blurred)

    # Bước 3: Triệt tiêu không cực đại (NMS)
    nms = non_maximum_suppression(magnitude, gx, gy, low_threshold=low_threshold)

    # Bước 4: Phân ngưỡng kép & Nối biên trễ
    edges = double_threshold_hysteresis(nms, low_threshold=low_threshold, high_threshold=high_threshold)

    return edges


def canny_edge_scratch(
    image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    kernel_size: int = 5,
    sigma: float = 1.4
) -> np.ndarray:
    """Bí danh (Alias) cho canny_edge_detector theo đúng giao diện gọi chuẩn trong TV4."""
    return canny_edge_detector(
        image,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        kernel_size=kernel_size,
        sigma=sigma,
        preblurred=False
    )
