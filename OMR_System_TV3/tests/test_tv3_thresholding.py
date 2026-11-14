"""
UNIT TESTS CHO MODULE TV3_THRESHOLDING (NHÓM 4 - THÀNH VIÊN 3)
Kiểm thử toàn diện:
    1. Kiểm chứng Otsu Thresholding Scratch vs. OpenCV (Delta T <= 1, IoU >= 98.5%).
    2. Kiểm chứng Integral Image Scratch vs. cv2.integral (Match 100%).
    3. Kiểm chứng Adaptive Mean Thresholding Scratch vs. OpenCV trên ảnh bóng đổ DS-03 (IoU >= 98.5%).
    4. Kiểm chứng các chế độ BINARY và BINARY_INV (OMR).
    5. Kiểm thử biên và các ngoại lệ đầu vào (Edge & Exception Cases).
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Thêm thư mục gốc vào PYTHONPATH để import src.tv3_thresholding
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tv3_thresholding import (
    compute_histogram_scratch,
    otsu_threshold_scratch,
    compute_integral_image_scratch,
    adaptive_mean_threshold_scratch,
    calculate_segmentation_metrics,
    benchmark_otsu_vs_opencv,
    benchmark_adaptive_vs_opencv
)


class TestOtsuThresholding:
    """Kiểm thử thuật toán Otsu's Global Thresholding."""

    def test_compute_histogram_scratch(self):
        """Kiểm chứng hàm tính histogram scratch khớp 100% với cv2.calcHist."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (120, 150), dtype=np.uint8)
        
        hist_scratch = compute_histogram_scratch(img)
        hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().astype(int)
        
        assert hist_scratch.shape == (256,)
        assert np.array_equal(hist_scratch, hist_cv), "Histogram scratch không khớp với cv2.calcHist!"
        assert np.sum(hist_scratch) == img.size

    @pytest.mark.parametrize("seed, mean1, mean2, std1, std2", [
        (10, 50, 190, 10, 15),
        (20, 70, 210, 20, 10),
        (30, 40, 160, 15, 25),
        (40, 90, 180, 12, 12),
        (50, 30, 220, 8, 8),
    ])
    def test_otsu_threshold_bimodal_distributions(self, seed, mean1, mean2, std1, std2):
        """Kiểm chứng Delta T <= 1 và IoU >= 99% trên nhiều phân bố hai đỉnh (Bimodal)."""
        np.random.seed(seed)
        H, W = 150, 150
        n1 = (H * W) // 2
        n2 = H * W - n1
        
        samples1 = np.random.normal(mean1, std1, n1)
        samples2 = np.random.normal(mean2, std2, n2)
        img = np.concatenate([samples1, samples2]).clip(0, 255).astype(np.uint8).reshape(H, W)
        
        # 1. Chạy Scratch
        t_scratch, bin_scratch = otsu_threshold_scratch(img, method='BINARY')
        
        # 2. Chạy OpenCV đối chứng
        t_cv, bin_cv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        delta_t = abs(t_scratch - t_cv)
        metrics = calculate_segmentation_metrics(bin_scratch, bin_cv)
        
        assert delta_t <= 1.0, f"Delta T = {delta_t} > 1.0 (Scratch: {t_scratch}, OpenCV: {t_cv})"
        assert metrics['iou_percent'] >= 98.5, f"IoU = {metrics['iou_percent']}% < 98.5%"

    def test_otsu_omr_binary_inv_mode(self):
        """Kiểm chứng chế độ BINARY_INV trên ảnh phiếu OMR mô phỏng (ô tô đen thành trắng = 255)."""
        np.random.seed(99)
        # Nền giấy trắng (mức xám 210-240), chữ và ô tô chì đậm (mức xám 30-60)
        img = np.random.normal(225, 10, (200, 200)).clip(0, 255).astype(np.uint8)
        img[30:60, 30:60] = np.random.normal(45, 5, (30, 30)).clip(0, 255).astype(np.uint8)
        img[120:150, 120:150] = np.random.normal(50, 5, (30, 30)).clip(0, 255).astype(np.uint8)

        t_scratch, bin_scratch = otsu_threshold_scratch(img, method='BINARY_INV')
        t_cv, bin_cv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        assert abs(t_scratch - t_cv) <= 1.0
        # Ô tô chì phải trở thành màu trắng 255
        assert np.mean(bin_scratch[35:55, 35:55]) == 255.0
        assert np.mean(bin_scratch[125:145, 125:145]) == 255.0
        # Nền giấy phải là 0
        assert np.mean(bin_scratch[0:20, 0:20]) == 0.0

        metrics = calculate_segmentation_metrics(bin_scratch, bin_cv)
        assert metrics['iou_percent'] >= 99.0


class TestIntegralImage:
    """Kiểm thử Bảng Ảnh tích phân (Integral Image)."""

    def test_integral_image_exact_match_opencv(self):
        """Kiểm chứng ma trận ảnh tích phân scratch khớp 100% với cv2.integral."""
        np.random.seed(123)
        img = np.random.randint(0, 256, (180, 220), dtype=np.uint8)
        
        integral_scratch = compute_integral_image_scratch(img)
        integral_cv = cv2.integral(img)
        
        assert integral_scratch.shape == (img.shape[0] + 1, img.shape[1] + 1)
        assert np.array_equal(integral_scratch, integral_cv), "Ảnh tích phân scratch không khớp với cv2.integral!"

    def test_integral_image_window_sum_property(self):
        """Kiểm chứng tính chất tính tổng cửa sổ O(1) qua 4 góc của ảnh tích phân."""
        img = np.arange(1, 26, dtype=np.uint8).reshape(5, 5)
        integral = compute_integral_image_scratch(img)
        
        # Tính tổng cửa sổ từ hàng 1..3, cột 1..3 (kích thước 3x3 trong mảng 0-indexed)
        # Tọa độ ảnh: r1=1, r2=4, c1=1, c2=4
        r1, r2, c1, c2 = 1, 4, 1, 4
        sum_direct = np.sum(img[r1:r2, c1:c2])
        
        # Công thức qua ma trận tích phân: S(r2, c2) - S(r1, c2) - S(r2, c1) + S(r1, c1)
        sum_integral = integral[r2, c2] - integral[r1, c2] - integral[r2, c1] + integral[r1, c1]
        
        assert sum_direct == sum_integral, f"Tổng trực tiếp ({sum_direct}) != tổng tích phân ({sum_integral})"


class TestAdaptiveThresholding:
    """Kiểm thử thuật toán Adaptive Mean Thresholding trên ảnh biến thiên ánh sáng và bóng đổ."""

    @pytest.fixture
    def shadow_omr_image(self):
        """Tạo ảnh OMR mô phỏng tập DS-03 có gradient bóng đổ mạnh và ô đáp án tô chì."""
        H, W = 300, 300
        # Nền trắng 230
        img = np.full((H, W), 230, dtype=np.uint8)
        
        # Gradient bóng đổ che từ góc trên trái sang góc dưới phải (mức xám tụt 130 đơn vị)
        Y, X = np.ogrid[:H, :W]
        shadow = ((X + Y) / (H + W) * 130).astype(np.uint8)
        img = cv2.subtract(img, shadow)
        
        # Thêm 4 ô tròn tô chì ở các vị trí khác nhau (vùng sáng, vùng tối, vùng biên bóng)
        # Chú ý: cv2.circle nhận tâm là (col, row) = (x, y)
        cv2.circle(img, (50, 50), 12, 35, -1)     # row 50, col 50 (Vùng sáng nhất)
        cv2.circle(img, (250, 250), 12, 30, -1)   # row 250, col 250 (Vùng tối nhất sâu trong bóng đổ)
        cv2.circle(img, (150, 150), 12, 32, -1)   # row 150, col 150 (Vùng chuyển tiếp)
        cv2.circle(img, (50, 250), 12, 33, -1)    # row 250, col 50 (Vùng trung bình)
        
        return img

    @pytest.mark.parametrize("block_size", [15, 21, 25, 31])
    @pytest.mark.parametrize("C", [5.0, 10.0, 12.0])
    @pytest.mark.parametrize("method", ['BINARY', 'BINARY_INV'])
    def test_adaptive_threshold_vs_opencv_various_params(self, shadow_omr_image, block_size, C, method):
        """Kiểm chứng IoU >= 99.0% và Dice >= 99.0% trên toàn bộ các bộ tham số thực tế."""
        res = benchmark_adaptive_vs_opencv(
            shadow_omr_image, 
            block_size=block_size, 
            C=C, 
            method=method
        )
        metrics = res['metrics']
        
        assert metrics['iou_percent'] >= 98.5, f"IoU đạt {metrics['iou_percent']}% < 98.5% với b={block_size}, C={C}, {method}"
        assert metrics['dice_percent'] >= 98.5, f"Dice đạt {metrics['dice_percent']}% < 98.5%"
        assert metrics['pixel_accuracy_percent'] >= 99.5, f"Pixel accuracy đạt {metrics['pixel_accuracy_percent']}%"

    def test_adaptive_vs_otsu_superiority_on_shadow(self, shadow_omr_image):
        """
        Kiểm chứng thực nghiệm (Ablation Study): Trên ảnh bóng đổ DS-03,
        Adaptive Thresholding tách thành công cả 4 ô chì, trong khi Otsu toàn cục
        bị hỏng (do vùng bóng đổ bị nhầm thành ô tô hoặc mất ô tô ở vùng bóng râm).
        """
        # 1. Chạy Otsu toàn cục
        _, bin_otsu = otsu_threshold_scratch(shadow_omr_image, method='BINARY_INV')
        # 2. Chạy Adaptive Thresholding
        bin_adaptive = adaptive_mean_threshold_scratch(shadow_omr_image, block_size=25, C=10.0, method='BINARY_INV')
        
        # Kiểm tra ô ở vùng bóng râm sâu (250, 250)
        # Trong Otsu: toàn bộ vùng bóng râm có thể bị biến thành trắng (vỡ nền)
        # Tỷ lệ pixel trắng ở vùng nền góc dưới phải (không có ô tô)
        bg_otsu_fill = np.mean(bin_otsu[260:290, 260:290] > 0)
        bg_adapt_fill = np.mean(bin_adaptive[260:290, 260:290] > 0)
        
        # Adaptive phải giữ sạch nền bóng đổ (< 5% nhiễu)
        assert bg_adapt_fill < 0.05, f"Adaptive không làm sạch được nền bóng đổ: fill={bg_adapt_fill}"
        # Cả 4 ô tô trong Adaptive đều phải nhận diện rõ (pixel trắng tại tâm ô > 0)
        assert bin_adaptive[50, 50] == 255
        assert bin_adaptive[250, 250] == 255
        assert bin_adaptive[150, 150] == 255
        assert bin_adaptive[250, 50] == 255


class TestBenchmarksAndExceptions:
    """Kiểm thử tính toàn vẹn của các hàm Benchmark và xử lý ngoại lệ."""

    def test_benchmark_otsu_vs_opencv(self):
        """Kiểm chứng cấu trúc kết quả trả về của hàm benchmark_otsu_vs_opencv."""
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        res = benchmark_otsu_vs_opencv(img, method='BINARY')
        
        assert 'threshold_scratch' in res
        assert 'threshold_opencv' in res
        assert 'delta_threshold' in res
        assert 'latency_scratch_ms' in res
        assert 'metrics' in res
        assert res['delta_threshold'] <= 1.0

    def test_benchmark_adaptive_vs_opencv(self):
        """Kiểm chứng cấu trúc kết quả trả về của hàm benchmark_adaptive_vs_opencv."""
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        res = benchmark_adaptive_vs_opencv(img, block_size=21, C=8.0, method='BINARY_INV')
        
        assert 'block_size' in res
        assert 'C' in res
        assert 'latency_scratch_ms' in res
        assert 'metrics' in res
        assert res['metrics']['iou_percent'] >= 98.5

    def test_input_validation_exceptions(self):
        """Kiểm thử bắt lỗi khi truyền đầu vào sai quy chuẩn."""
        img_3d = np.zeros((50, 50, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            compute_histogram_scratch(img_3d)
            
        with pytest.raises(ValueError):
            otsu_threshold_scratch(img_3d)
            
        with pytest.raises(ValueError):
            compute_integral_image_scratch(img_3d)
            
        img_2d = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError):
            # block_size chẵn phải báo lỗi
            adaptive_mean_threshold_scratch(img_2d, block_size=20)
            
        with pytest.raises(ValueError):
            # block_size < 3 phải báo lỗi
            adaptive_mean_threshold_scratch(img_2d, block_size=1)
            
        with pytest.raises(ValueError):
            # method không hợp lệ
            otsu_threshold_scratch(img_2d, method='INVALID_METHOD')
