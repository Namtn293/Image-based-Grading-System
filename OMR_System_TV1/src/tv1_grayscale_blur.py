"""Module: TV1 - Chuyển đổi mức xám & Làm mịn khử nhiễu Gaussian 2D
Đề tài: Hệ thống Chấm bài Trắc nghiệm Tự động (OMR System)
Môn học: Xử lý ảnh số (INT13146) - Học viện Công nghệ Bưu chính Viễn thông (PTIT)
Giảng viên hướng dẫn: TS. Phạm Hoàng Việt
Thành viên phụ trách: Thành viên 1 (Tiền xử lý & Phát hiện khung phiếu)

YÊU CẦU BẮT BUỘC:
- 100% Cài đặt từ đầu (Code from Scratch) bằng NumPy.
- KHÔNG sử dụng các hàm xử lý có sẵn của OpenCV trong luồng chính (cv2.cvtColor, cv2.GaussianBlur).
- OpenCV chỉ dùng độc lập trong module kiểm thử để đối chứng số học (MSE, PSNR) và thời gian thực thi.
"""

from __future__ import annotations
from typing import Tuple, Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rgb_to_gray_scratch(image: np.ndarray, is_bgr: bool = False) -> np.ndarray:
    """Chuyển đổi ảnh màu sang ảnh mức xám theo chuẩn Luminance ITU-R BT.601.

    Cơ sở toán học:
      Mức xám cảm nhận theo thị giác con người (độ nhạy võng mạc cao nhất với ánh sáng xanh lục G,
      sau đó đến đỏ R, và thấp nhất với xanh lam B):
        I_gray(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)

    Args:
        image: Mảng numpy shape (H, W), (H, W, 1), (H, W, 3), hoặc (H, W, 4).
               Mặc định kênh màu là RGB trừ khi is_bgr=True.
        is_bgr: Nếu True, ảnh được coi là định dạng BGR (chuẩn mặc định của cv2.imread).
                Nếu False, ảnh được coi là RGB chuẩn.

    Returns:
        Mảng numpy 2D uint8 shape (H, W) với mức xám trong khoảng [0, 255].
    """
    img = np.asarray(image)

    # Trường hợp ảnh đã là ảnh mức xám 2D
    if img.ndim == 2:
        if img.dtype == np.uint8:
            return img.copy()
        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    # Trường hợp ảnh có shape (H, W, 1)
    if img.ndim == 3 and img.shape[2] == 1:
        squeezed = img[:, :, 0]
        if squeezed.dtype == np.uint8:
            return squeezed.copy()
        return np.clip(np.round(squeezed), 0, 255).astype(np.uint8)

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Kích thước ảnh không hợp lệ cho chuyển đổi mức xám: {img.shape}")

    # Chuyển đổi dữ liệu sang float64 để tránh tràn số khi nhân cộng
    img_f = img.astype(np.float64)

    if is_bgr:
        b = img_f[:, :, 0]
        g = img_f[:, :, 1]
        r = img_f[:, :, 2]
    else:
        r = img_f[:, :, 0]
        g = img_f[:, :, 1]
        b = img_f[:, :, 2]

    # Tổ hợp tuyến tính theo trọng số ITU-R BT.601
    gray_f = 0.299 * r + 0.587 * g + 0.114 * b

    return np.clip(np.round(gray_f), 0, 255).astype(np.uint8)


def grayscale_luminance(image: np.ndarray, is_bgr: bool = False) -> np.ndarray:
    """Bí danh (Alias) cho rgb_to_gray_scratch theo đúng tên gọi trong Bảng 6.1 đề cương."""
    return rgb_to_gray_scratch(image, is_bgr=is_bgr)


def get_gaussian_kernel_1d(ksize: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Sinh ma trận nhân Gaussian 1 chiều chuẩn hóa.

    Cơ sở toán học:
      g(u) = exp( - u^2 / (2 * sigma^2) ) với u thuộc [-r, r], r = floor(ksize / 2)
      Chuẩn hóa: K = g / sum(g) để bảo toàn tổng năng lượng = 1.0.

    Args:
        ksize: Kích thước kernel (phải là số nguyên lẻ dương >= 1).
        sigma: Độ lệch chuẩn của phân phối Gaussian (sigma > 0).

    Returns:
        Mảng 1D float64 shape (ksize,) có tổng bằng 1.0.
    """
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"Kích thước kernel Gaussian phải là số lẻ >= 1, nhận được {ksize}")
    if sigma <= 0:
        raise ValueError(f"Sigma phải > 0, nhận được {sigma}")

    radius = ksize // 2
    u = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (u / sigma) ** 2)
    sum_val = np.sum(kernel_1d)
    if sum_val > 0:
        kernel_1d /= sum_val
    return kernel_1d


def get_gaussian_kernel_2d(ksize: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Sinh ma trận nhân Gaussian 2 chiều kích thước (ksize x ksize) chuẩn hóa.

    Cơ sở toán học:
      G(u, v) = (1 / (2 * pi * sigma^2)) * exp( - (u^2 + v^2) / (2 * sigma^2) )
      Do tính chất khả tích phân tách (Separable Kernel): G(u, v) = g(u) * g(v).

    Args:
        ksize: Kích thước ma trận vuông kernel (phải là số lẻ >= 1).
        sigma: Độ lệch chuẩn phân phối Gaussian.

    Returns:
        Mảng 2D float64 shape (ksize, ksize) có tổng bằng 1.0.
    """
    k1d = get_gaussian_kernel_1d(ksize, sigma)
    # Tích ngoài (Outer product) của 2 vector 1D
    return np.outer(k1d, k1d)


def pad_image_reflect(image: np.ndarray, pad_size: int) -> np.ndarray:
    """Đệm biên đối xứng (Reflect Padding) cho ảnh mức xám 2D.

    Tuân thủ đúng chuẩn BORDER_REFLECT_101 của OpenCV (không lặp lại điểm ảnh ở rìa biên):
      fedcba | abcdefgh | hgfedc

    Args:
        image: Mảng ảnh mức xám 2D shape (H, W).
        pad_size: Số hàng/cột cần đệm ở mỗi phía.

    Returns:
        Mảng ảnh đã đệm shape (H + 2*pad_size, W + 2*pad_size).
    """
    if pad_size <= 0:
        return image.copy()
    return np.pad(image, pad_size, mode="reflect")


def convolve2d_scratch(
    image: np.ndarray,
    kernel: np.ndarray,
    mode: str = "reflect"
) -> np.ndarray:
    """Thực hiện phép tích chập không gian 2D (2D Spatial Convolution) tổng quát từ đầu bằng NumPy.

    Cơ sở toán học:
      I_out(x, y) = sum_u sum_v I_padded(x + r - u, y + r - v) * K(u, v)
      với phép lật nhân (kernel flip 180 độ) theo đúng định nghĩa tích chập giải tích.
      Tối ưu hóa vector hóa 100% bằng sliding_window_view, không dùng 4 vòng lặp for lồng nhau.

    Args:
        image: Ảnh mức xám 2D shape (H, W).
        kernel: Ma trận nhân 2D shape (kh, kw), kh và kw là số lẻ.
        mode: Kiểu đệm biên ('reflect', 'edge', 'constant').

    Returns:
        Mảng kết quả float64 shape (H, W).
    """
    img = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)

    if img.ndim != 2:
        raise ValueError(f"convolve2d_scratch chỉ hỗ trợ ảnh 2D, nhận được shape {img.shape}")
    if k.ndim != 2:
        raise ValueError(f"Kernel phải là ma trận 2D, nhận được shape {k.shape}")

    kh, kw = k.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Lật kernel 180 độ theo chuẩn định nghĩa toán học của Tích chập (Convolution)
    # (Phân biệt với Phép tương quan chéo Correlation)
    k_flipped = np.flipud(np.fliplr(k))

    if mode == "reflect":
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    elif mode == "edge":
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    else:
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0.0)

    # Trích xuất toàn bộ cửa sổ cục bộ kích thước (kh, kw) đồng thời qua con trỏ bộ nhớ strided
    windows = sliding_window_view(padded, window_shape=(kh, kw))

    # Tích vô hướng (einsum) giữa các cửa sổ và kernel đã lật
    out = np.einsum("ijkl,kl->ij", windows, k_flipped)
    return out


def gaussian_blur_2d(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.4
) -> np.ndarray:
    """Làm mịn ảnh khử nhiễu bằng Bộ lọc Gaussian 2D (Code from Scratch bằng NumPy).

    Sử dụng kỹ thuật Tích chập phân tách (Separable 1D Filtering):
      Tích chập theo trục ngang bằng kernel 1D g(u) (O(W*H*k) phép tính),
      tiếp theo tích chập theo trục dọc bằng kernel 1D g(v) (O(W*H*k) phép tính).
      Tổng độ phức tạp O(2k) thay vì O(k^2) của tích chập 2D thông thường,
      đạt tốc độ nhanh gấp ~3-5 lần mà độ chính xác toán học hoàn toàn tương đương.

    Args:
        image: Ảnh đầu vào 2D mức xám (H, W).
        kernel_size: Kích thước cửa sổ lọc (số lẻ >= 1, mặc định k=5).
        sigma: Độ lệch chuẩn phân phối Gaussian (mặc định sigma=1.4 theo Canny chuẩn).

    Returns:
        Mảng uint8 shape (H, W) đã làm mịn, mức xám [0, 255].
    """
    img = np.asarray(image)
    if img.ndim != 2:
        if img.ndim == 3:
            img = rgb_to_gray_scratch(img)
        else:
            raise ValueError(f"gaussian_blur_2d yêu cầu ảnh 2D, nhận được shape {img.shape}")

    if kernel_size <= 1:
        return img.astype(np.uint8)

    img_f = img.astype(np.float64)
    r = kernel_size // 2
    k1d = get_gaussian_kernel_1d(kernel_size, sigma)

    # Đệm biên đối xứng (Reflect 101)
    padded = pad_image_reflect(img_f, r)

    # Bước 1: Tích chập theo hàng (Horizontal axis 1)
    windows_h = sliding_window_view(padded, window_shape=kernel_size, axis=1)
    blurred_h = np.sum(windows_h * k1d, axis=-1)

    # Bước 2: Tích chập theo cột (Vertical axis 0)
    windows_v = sliding_window_view(blurred_h, window_shape=kernel_size, axis=0)
    blurred_v = np.sum(windows_v * k1d, axis=-1)

    # Làm tròn và ép kiểu uint8
    return np.clip(np.round(blurred_v), 0, 255).astype(np.uint8)


def gaussian_blur_scratch(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.4
) -> np.ndarray:
    """Bí danh (Alias) cho gaussian_blur_2d theo đúng giao diện gọi của TV4."""
    return gaussian_blur_2d(image, kernel_size=kernel_size, sigma=sigma)
