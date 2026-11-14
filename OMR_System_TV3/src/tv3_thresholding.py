"""
DỰ ÁN XỬ LÝ ẢNH (INT13146) - HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM OMR
Module: tv3_thresholding.py (Phân công: Thành viên 3)
Chức năng:
    1. Phân ngưỡng tự động Otsu toàn cục (Otsu's Global Thresholding) - Code from Scratch bằng NumPy.
    2. Bảng Ảnh tích phân (Integral Image) & Phân ngưỡng thích nghi cục bộ (Adaptive Mean Thresholding) - Code from Scratch O(1).
    3. Bộ đo Benchmark & Đối chứng độ chính xác với thư viện OpenCV chuẩn.

Tác giả: Nhóm 4 - Thành viên 3 (Thresholding Builder)
Nguyên tắc: Thuật toán luồng chính cài đặt 100% bằng NumPy thuần, không sử dụng các hàm xử lý nhị phân sẵn có của OpenCV.
            OpenCV chỉ được sử dụng độc lập để so sánh đối chứng độ chính xác và tốc độ.
"""

import time
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Union


# ==============================================================================
# 1. TÍNH TOÁN HISTOGRAM VÀ PHÂN NGƯỠNG OTSU TOÀN CỤC FROM SCRATCH
# ==============================================================================

def compute_histogram_scratch(image: np.ndarray) -> np.ndarray:
    """
    Tính biểu đồ phân bố mức xám (Histogram) 256 mức xám [0, 255] thuần túy bằng NumPy.
    Không sử dụng cv2.calcHist.

    Tham số:
        image (np.ndarray): Ma trận ảnh mức xám 2D (uint8).

    Trả về:
        np.ndarray: Mảng 1D gồm 256 phần tử biểu diễn số lượng pixel tại mỗi mức xám.
    """
    if image.ndim != 2:
        raise ValueError(f"Đầu vào phải là ảnh mức xám 2D, nhận được shape: {image.shape}")
    
    # Ép kiểu an toàn sang uint8 nếu cần
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
        
    hist = np.bincount(image.ravel(), minlength=256)
    return hist


def otsu_threshold_scratch(
    image: np.ndarray, 
    method: str = 'BINARY', 
    max_val: int = 255
) -> Tuple[float, np.ndarray]:
    """
    Thuật toán Phân ngưỡng tự động Otsu (Otsu's Global Thresholding) từ đầu bằng NumPy.
    Cơ sở lý thuyết:
        - Tính phân bố xác suất p_i = n_i / N cho mỗi mức xám i in [0, 255].
        - Với mỗi ngưỡng thử t in [0, 255]:
            w_0(t) = sum_{i=0}^t p_i           (Trọng số lớp C0: nền/tiền cảnh)
            w_1(t) = 1 - w_0(t)                (Trọng số lớp C1: hậu cảnh)
            mu_0(t) = sum_{i=0}^t (i * p_i) / w_0(t)
            mu_1(t) = sum_{i=t+1}^255 (i * p_i) / w_1(t)
            sigma_B^2(t) = w_0(t) * w_1(t) * [mu_0(t) - mu_1(t)]^2
        - Ngưỡng tối ưu T* = argmax_{t} sigma_B^2(t).

    Tham số:
        image (np.ndarray): Ảnh mức xám đầu vào 2D (uint8).
        method (str): 
            - 'BINARY': pixel > T* -> max_val, pixel <= T* -> 0
            - 'BINARY_INV': pixel <= T* -> max_val, pixel > T* -> 0 (Nền đen - Chữ/Ô tô trắng phục vụ OMR)
        max_val (int): Giá trị gán cho điểm ảnh vượt ngưỡng (mặc định 255).

    Trả về:
        Tuple[float, np.ndarray]: (T_star, binary_image)
            - T_star: Ngưỡng tối ưu tìm được (float/int).
            - binary_image: Ma trận ảnh nhị phân kích thước như ảnh gốc (uint8).
    """
    if image.ndim != 2:
        raise ValueError(f"Ảnh đầu vào phải là ảnh 2D mức xám, nhận được shape: {image.shape}")
        
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    hist = compute_histogram_scratch(image).astype(np.float64)
    total_pixels = image.size
    if total_pixels == 0:
        return 0.0, np.zeros_like(image)

    scale = 1.0 / total_pixels
    # Trung bình toàn cục mu_T
    mu_total = np.sum(np.arange(256) * hist) * scale

    q1 = 0.0          # w_0(t)
    sum1 = 0.0        # w_0(t) * mu_0(t)
    max_sigma = -1.0  # Phương sai liên lớp cực đại
    optimal_t = 0     # Ngưỡng T*

    # Duyệt qua 256 mức xám
    for t in range(256):
        p_t = hist[t] * scale
        q1 += p_t
        q2 = 1.0 - q1  # w_1(t)

        # Tránh chia cho 0 tại biên
        if q1 <= 1e-7 or q2 <= 1e-7:
            continue

        sum1 += t * p_t
        mu1 = sum1 / q1
        mu2 = (mu_total - sum1) / q2

        # sigma_B^2(t) = w_0 * w_1 * (mu_0 - mu_1)^2
        sigma_b_squared = q1 * q2 * ((mu1 - mu2) ** 2)

        # Cực đại hóa phương sai giữa hai lớp (giữ ngưỡng đầu tiên nếu bằng nhau theo chuẩn OpenCV)
        if sigma_b_squared > max_sigma:
            max_sigma = sigma_b_squared
            optimal_t = t

    optimal_threshold = float(optimal_t)

    # Áp dụng ngưỡng nhị phân hóa
    method_upper = method.upper()
    if method_upper == 'BINARY':
        binary_image = np.where(image > optimal_threshold, max_val, 0).astype(np.uint8)
    elif method_upper in ('BINARY_INV', 'INV'):
        binary_image = np.where(image <= optimal_threshold, max_val, 0).astype(np.uint8)
    else:
        raise ValueError(f"Phương thức nhị phân '{method}' không hỗ trợ. Dùng 'BINARY' hoặc 'BINARY_INV'.")

    return optimal_threshold, binary_image


# ==============================================================================
# 2. BẢNG ẢNH TÍCH PHÂN VÀ PHÂN NGƯỠNG THÍCH NGHI CỤC BỘ FROM SCRATCH
# ==============================================================================

def compute_integral_image_scratch(image: np.ndarray) -> np.ndarray:
    """
    Tính Bảng Ảnh tích phân (Integral Image / Summed Area Table) từ đầu bằng NumPy cumsum.
    Cơ sở lý thuyết:
        S(y, x) = sum_{r <= y, c <= x} I(r, c)
        Ma trận đầu ra có kích thước (H + 1, W + 1) với dòng 0 và cột 0 bằng 0.
        Giúp tính tổng mức xám trong bất kỳ cửa sổ hình chữ nhật nào với độ phức tạp O(1).

    Tham số:
        image (np.ndarray): Ảnh mức xám 2D (uint8 hoặc int/float).

    Trả về:
        np.ndarray: Ma trận tích phân kích thước (H + 1, W + 1) kiểu np.uint64 (chống tràn số).
    """
    if image.ndim != 2:
        raise ValueError(f"Ảnh đầu vào phải là ảnh mức xám 2D, nhận được shape: {image.shape}")
        
    H, W = image.shape
    integral = np.zeros((H + 1, W + 1), dtype=np.uint64)
    # Tích lũy theo cột dọc rồi theo hàng ngang
    integral[1:, 1:] = np.cumsum(np.cumsum(image.astype(np.uint64), axis=0), axis=1)
    return integral


def adaptive_mean_threshold_scratch(
    image: np.ndarray,
    block_size: int = 25,
    C: float = 10.0,
    method: str = 'BINARY_INV',
    max_val: int = 255,
    border_type: str = 'replicate'
) -> np.ndarray:
    """
    Phân ngưỡng thích nghi cục bộ theo giá trị trung bình (Adaptive Mean Thresholding)
    tối ưu tốc độ O(1) mỗi điểm ảnh bằng Bảng Ảnh tích phân (Integral Image).

    Cơ sở lý thuyết:
        - Với mỗi điểm ảnh (y, x), ngưỡng cục bộ được xác định bởi:
            T(y, x) = mu_{local}(y, x) - C
          với mu_{local}(y, x) là trung bình mức xám của cửa sổ b x b quanh (y, x).
        - Khi ảnh có bóng đổ hoặc chùm sáng không đều (tập DS-03), mức xám nền biến thiên
          liên tục khiến Otsu toàn cục thất bại. Adaptive Thresholding giải quyết triệt để vấn đề này.
        - Xử lý biên đệm (Padding) chuẩn 'replicate' (nhân bản pixel biên) khớp với OpenCV boxFilter.
        - Tính tổng cửa sổ b x b đồng thời trên toàn bộ mảng bằng ma trận tích phân (Vectorization).

    Tham số:
        image (np.ndarray): Ảnh mức xám 2D (uint8).
        block_size (int): Kích thước cạnh cửa sổ vuông lân cận b (phải là số nguyên lẻ >= 3, ví dụ 15, 21, 25, 31).
        C (float): Hằng số trừ đi từ trung bình cục bộ (thường từ 5.0 đến 15.0).
        method (str): 'BINARY' hoặc 'BINARY_INV' (mặc định 'BINARY_INV' cho bài toán OMR).
        max_val (int): Giá trị sáng gán cho pixel đạt ngưỡng (mặc định 255).
        border_type (str): Kiểu đệm biên ('replicate' tương ứng cv2.BORDER_REPLICATE).

    Trả về:
        np.ndarray: Ảnh nhị phân kích thước (H, W) kiểu np.uint8.
    """
    if image.ndim != 2:
        raise ValueError(f"Ảnh đầu vào phải là ảnh 2D mức xám, nhận được shape: {image.shape}")
        
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(f"block_size phải là số nguyên lẻ >= 3, nhận được: {block_size}")
        
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    H, W = image.shape
    r = block_size // 2

    # 1. Đệm biên ảnh theo chuẩn BORDER_REPLICATE (edge padding)
    if border_type.lower() in ('replicate', 'edge'):
        pad_image = np.pad(image, r, mode='edge')
    elif border_type.lower() == 'reflect':
        pad_image = np.pad(image, r, mode='reflect')
    elif border_type.lower() == 'constant':
        pad_image = np.pad(image, r, mode='constant', constant_values=0)
    else:
        pad_image = np.pad(image, r, mode='edge')

    # 2. Xây dựng ảnh tích phân trên ảnh đã đệm biên
    pad_integral = compute_integral_image_scratch(pad_image)

    # 3. Tính tổng cửa sổ b x b với độ phức tạp O(1) thông qua 4 góc của ảnh tích phân
    # Vùng cửa sổ tương ứng ảnh gốc (H, W) sau khi đệm r pixel:
    # Top-Left: (0, 0) đến (H, W); Bottom-Right: (b, b) đến (H+b, W+b)
    b = block_size
    window_sum = (
        pad_integral[b:, b:]
        - pad_integral[:-b, b:]
        - pad_integral[b:, :-b]
        + pad_integral[:-b, :-b]
    )

    # 4. Tính trung bình mức xám cục bộ
    # OpenCV boxFilter với CV_8U làm tròn số nguyên gần nhất (round half to even / cvRound)
    window_area = float(b * b)
    local_mean = np.round(window_sum / window_area).astype(np.int32)

    # 5. Phân ngưỡng nhị phân
    method_upper = method.upper()
    img_int = image.astype(np.int32)
    threshold_matrix = local_mean - int(np.round(C))

    if method_upper == 'BINARY':
        # src(y, x) > T(y, x) -> max_val, ngược lại 0
        binary_image = np.where(img_int > threshold_matrix, max_val, 0).astype(np.uint8)
    elif method_upper in ('BINARY_INV', 'INV'):
        # src(y, x) <= T(y, x) -> max_val, ngược lại 0 (Chuẩn OpenCV và OMR)
        binary_image = np.where(img_int <= threshold_matrix, max_val, 0).astype(np.uint8)
    else:
        raise ValueError(f"Phương thức nhị phân '{method}' không hỗ trợ. Dùng 'BINARY' hoặc 'BINARY_INV'.")

    return binary_image


# ==============================================================================
# 3. CHỈ SỐ ĐÁNH GIÁ ĐỊNH LƯỢNG (MSE, PSNR, IOU, DICE)
# ==============================================================================

def calculate_segmentation_metrics(
    binary_scratch: np.ndarray, 
    binary_opencv: np.ndarray
) -> Dict[str, float]:
    """
    Tính toán đầy đủ các chỉ số định lượng đánh giá mức độ trùng khớp giữa
    ảnh nhị phân tự cài đặt (NumPy scratch) và ảnh đối chứng thư viện OpenCV.

    Chỉ số bao gồm:
        - MSE: Sai số bình phương trung bình.
        - PSNR (dB): Tỷ số tín hiệu cực đại trên nhiễu.
        - IoU (%): Hệ số giao trên hợp (Intersection over Union / Jaccard Index).
        - Dice (%): Hệ số tương đồng Sørensen–Dice.
        - Accuracy (%): Tỷ lệ số điểm ảnh trùng khớp hoàn toàn.

    Tham số:
        binary_scratch (np.ndarray): Ảnh nhị phân từ thuật toán Scratch (uint8).
        binary_opencv (np.ndarray): Ảnh nhị phân từ hàm OpenCV đối chứng (uint8).

    Trả về:
        Dict[str, float]: Bảng kết quả các chỉ số định lượng.
    """
    if binary_scratch.shape != binary_opencv.shape:
        raise ValueError(f"Kích thước hai ma trận không khớp: {binary_scratch.shape} vs {binary_opencv.shape}")

    # Chuyển đổi nhị phân boolean (coi pixel > 0 là vùng tiền cảnh 1)
    mask_scratch = (binary_scratch > 0)
    mask_opencv = (binary_opencv > 0)

    # 1. MSE & PSNR
    mse = float(np.mean((binary_scratch.astype(np.float64) - binary_opencv.astype(np.float64)) ** 2))
    if mse == 0.0:
        psnr = float('inf')
    else:
        psnr = float(10.0 * np.log10((255.0 ** 2) / mse))

    # 2. IoU (Jaccard Index)
    intersection = np.logical_and(mask_scratch, mask_opencv).sum()
    union = np.logical_or(mask_scratch, mask_opencv).sum()

    if union == 0:
        iou_percent = 100.0
    else:
        iou_percent = float((intersection / union) * 100.0)

    # 3. Dice Score
    total_fg = mask_scratch.sum() + mask_opencv.sum()
    if total_fg == 0:
        dice_percent = 100.0
    else:
        dice_percent = float((2.0 * intersection / total_fg) * 100.0)

    # 4. Pixel Accuracy
    matching_pixels = (binary_scratch == binary_opencv).sum()
    accuracy_percent = float((matching_pixels / binary_scratch.size) * 100.0)

    return {
        'mse': round(mse, 6),
        'psnr_db': round(psnr, 2) if psnr != float('inf') else 999.0,
        'iou_percent': round(iou_percent, 4),
        'dice_percent': round(dice_percent, 4),
        'pixel_accuracy_percent': round(accuracy_percent, 4)
    }


# ==============================================================================
# 4. MODULE ĐỐI CHỨNG VÀ BENCHMARK VỚI OPENCV
# ==============================================================================

def benchmark_otsu_vs_opencv(
    image: np.ndarray, 
    method: str = 'BINARY', 
    max_val: int = 255
) -> Dict[str, Any]:
    """
    So sánh đối chứng toàn diện thuật toán Otsu Scratch với cv2.threshold(..., cv2.THRESH_OTSU).

    Tham số:
        image (np.ndarray): Ảnh mức xám đầu vào 2D.
        method (str): 'BINARY' hoặc 'BINARY_INV'.
        max_val (int): Giá trị pixel nhị phân cực đại.

    Trả về:
        Dict[str, Any]: Báo cáo so sánh gồm:
            - t_scratch, t_opencv, delta_t (|T_scratch - T_opencv|)
            - thời gian chạy (latency_scratch_ms, latency_opencv_ms)
            - chỉ số định lượng: mse, psnr_db, iou_percent, dice_percent
            - ma trận ảnh nhị phân: binary_scratch, binary_opencv
    """
    if image.ndim != 2:
        raise ValueError(f"Ảnh đầu vào phải là ảnh mức xám 2D, nhận được shape: {image.shape}")
        
    img_gray = np.clip(image, 0, 255).astype(np.uint8)

    # 1. Chạy Otsu from Scratch
    t0 = time.perf_counter()
    t_scratch, bin_scratch = otsu_threshold_scratch(img_gray, method=method, max_val=max_val)
    latency_scratch_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Chạy OpenCV đối chứng
    cv_flag = cv2.THRESH_BINARY if method.upper() == 'BINARY' else cv2.THRESH_BINARY_INV
    t1 = time.perf_counter()
    ret_opencv, bin_opencv = cv2.threshold(img_gray, 0, max_val, cv_flag + cv2.THRESH_OTSU)
    latency_opencv_ms = (time.perf_counter() - t1) * 1000.0

    delta_t = abs(float(t_scratch) - float(ret_opencv))
    metrics = calculate_segmentation_metrics(bin_scratch, bin_opencv)

    return {
        'algorithm': 'Otsu Global Thresholding',
        'threshold_scratch': float(t_scratch),
        'threshold_opencv': float(ret_opencv),
        'delta_threshold': round(delta_t, 2),
        'latency_scratch_ms': round(latency_scratch_ms, 3),
        'latency_opencv_ms': round(latency_opencv_ms, 3),
        'metrics': metrics,
        'binary_scratch': bin_scratch,
        'binary_opencv': bin_opencv
    }


def benchmark_adaptive_vs_opencv(
    image: np.ndarray,
    block_size: int = 25,
    C: float = 10.0,
    method: str = 'BINARY_INV',
    max_val: int = 255
) -> Dict[str, Any]:
    """
    So sánh đối chứng toàn diện thuật toán Adaptive Mean Thresholding (Integral Image) Scratch
    với cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_MEAN_C, ...).

    Tham số:
        image (np.ndarray): Ảnh mức xám đầu vào 2D.
        block_size (int): Kích thước cửa sổ vuông lân cận (lẻ >= 3).
        C (float): Hằng số trừ đi từ trung bình cục bộ.
        method (str): 'BINARY' hoặc 'BINARY_INV'.
        max_val (int): Giá trị cực đại.

    Trả về:
        Dict[str, Any]: Báo cáo so sánh gồm thời gian thực thi, chỉ số định lượng và ảnh nhị phân.
    """
    if image.ndim != 2:
        raise ValueError(f"Ảnh đầu vào phải là ảnh mức xám 2D, nhận được shape: {image.shape}")
        
    img_gray = np.clip(image, 0, 255).astype(np.uint8)

    # 1. Chạy Adaptive Thresholding Scratch (Integral Image O(1))
    t0 = time.perf_counter()
    bin_scratch = adaptive_mean_threshold_scratch(
        img_gray, 
        block_size=block_size, 
        C=C, 
        method=method, 
        max_val=max_val, 
        border_type='replicate'
    )
    latency_scratch_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Chạy OpenCV đối chứng
    cv_flag = cv2.THRESH_BINARY if method.upper() == 'BINARY' else cv2.THRESH_BINARY_INV
    t1 = time.perf_counter()
    bin_opencv = cv2.adaptiveThreshold(
        img_gray, 
        max_val, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv_flag, 
        block_size, 
        C
    )
    latency_opencv_ms = (time.perf_counter() - t1) * 1000.0

    metrics = calculate_segmentation_metrics(bin_scratch, bin_opencv)

    return {
        'algorithm': 'Adaptive Mean Thresholding (Integral Image)',
        'block_size': block_size,
        'C': C,
        'latency_scratch_ms': round(latency_scratch_ms, 3),
        'latency_opencv_ms': round(latency_opencv_ms, 3),
        'metrics': metrics,
        'binary_scratch': bin_scratch,
        'binary_opencv': bin_opencv
    }


# ==============================================================================
# 5. KHỞI CHẠY THỬ NGHIỆM ĐỘC LẬP KHI CHẠY MODULE TRỰC TIẾP
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("KIỂM THỬ ĐỘC LẬP MODULE TV3_THRESHOLDING (NHÓM 4 - THÀNH VIÊN 3)")
    print("=" * 70)

    # 1. Tạo ảnh mẫu mô phỏng phiếu thi OMR với bóng đổ và ô tô
    np.random.seed(42)
    H, W = 400, 400
    test_img = np.full((H, W), 230, dtype=np.uint8)  # Nền giấy trắng

    # Thêm bóng đổ gradient góc chéo (mô phỏng tập DS-03)
    Y, X = np.ogrid[:H, :W]
    shadow_gradient = ((X + Y) / (H + W) * 120).astype(np.uint8)
    test_img = cv2.subtract(test_img, shadow_gradient)

    # Vẽ vài ô tròn đáp án tô chì đen (mức xám 30 - 60)
    test_img[100:130, 100:130] = 40  # Ô tô chì ở vùng sáng
    test_img[300:330, 300:330] = 40  # Ô tô chì ở vùng bóng đổ
    test_img[200:230, 200:230] = 50  # Ô tô chì ở giữa

    print("\n--- [1] BENCHMARK OTSU TOÀN CỤC (BINARY & BINARY_INV) ---")
    res_otsu = benchmark_otsu_vs_opencv(test_img, method='BINARY_INV')
    print(f"Ngưỡng T* Scratch: {res_otsu['threshold_scratch']} | OpenCV: {res_otsu['threshold_opencv']} | Delta T: {res_otsu['delta_threshold']}")
    print(f"Thời gian Scratch: {res_otsu['latency_scratch_ms']} ms | OpenCV: {res_otsu['latency_opencv_ms']} ms")
    print(f"Chỉ số IoU: {res_otsu['metrics']['iou_percent']}% | Dice: {res_otsu['metrics']['dice_percent']}% | MSE: {res_otsu['metrics']['mse']}")

    print("\n--- [2] BENCHMARK ADAPTIVE MEAN THRESHOLDING (INTEGRAL IMAGE) ---")
    res_adapt = benchmark_adaptive_vs_opencv(test_img, block_size=25, C=10.0, method='BINARY_INV')
    print(f"Thời gian Scratch (Integral O(1)): {res_adapt['latency_scratch_ms']} ms | OpenCV: {res_adapt['latency_opencv_ms']} ms")
    print(f"Chỉ số IoU: {res_adapt['metrics']['iou_percent']}% | Dice: {res_adapt['metrics']['dice_percent']}% | MSE: {res_adapt['metrics']['mse']}")
    print("=" * 70)
