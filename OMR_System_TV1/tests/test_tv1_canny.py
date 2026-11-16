"""Unit tests: Module TV1 - Phát hiện biên Canny (Canny Edge Detector)
Kiểm thử 4 bước thuật toán Canny thuần NumPy và đối chứng với OpenCV.
"""

import unittest
import numpy as np
import cv2

import sys
from pathlib import Path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tv1_canny import (
    sobel_gradients,
    non_maximum_suppression,
    double_threshold_hysteresis,
    canny_edge_detector,
    canny_edge_scratch
)
from tv1_grayscale_blur import gaussian_blur_2d


class TestTV1Canny(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho thuật toán Phát hiện biên Canny 4 bước."""

    def setUp(self):
        # Tạo ảnh tổng hợp có các cạnh rõ nét theo 4 hướng
        self.test_img = np.zeros((120, 120), dtype=np.uint8)
        self.test_img[30:90, 30:90] = 200
        # Thêm cạnh chéo
        for i in range(20):
            self.test_img[10 + i, 10 + i] = 220

    def test_sobel_gradients_vs_opencv(self):
        """Bước 1: Kiểm chứng Sobel Gx, Gy khớp 100% với cv2.Sobel."""
        blurred = cv2.GaussianBlur(self.test_img, (5, 5), 1.4, borderType=cv2.BORDER_REFLECT_101)

        gx_sc, gy_sc, mag_sc, _ = sobel_gradients(blurred)
        gx_cv = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        gy_cv = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)

        np.testing.assert_allclose(gx_sc, gx_cv, atol=1e-10, err_msg="Sobel Gx Scratch phải khớp 100% cv2.Sobel")
        np.testing.assert_allclose(gy_sc, gy_cv, atol=1e-10, err_msg="Sobel Gy Scratch phải khớp 100% cv2.Sobel")

    def test_nms_suppression_thin_edges(self):
        """Bước 2: Kiểm chứng NMS làm mảnh biên về độ dày 1 pixel."""
        blurred = gaussian_blur_2d(self.test_img, 5, 1.4)
        gx, gy, mag, _ = sobel_gradients(blurred)
        nms = non_maximum_suppression(mag, gx, gy, low_threshold=40.0)

        # Biên sau NMS phải mỏng hơn và số pixel khác 0 phải giảm đi đáng kể
        non_zero_mag = np.count_nonzero(mag >= 40.0)
        non_zero_nms = np.count_nonzero(nms >= 40.0)
        self.assertLess(non_zero_nms, non_zero_mag, "NMS phải triệt tiêu bớt các điểm không cực đại.")

    def test_hysteresis_connectivity(self):
        """Bước 3 & 4: Kiểm chứng điểm biên yếu kết nối được giữ lại, điểm cô lập bị xóa."""
        # Tạo ma trận NMS nhân tạo: 1 điểm mạnh nối với 1 điểm yếu, và 1 điểm yếu cô lập
        nms_mock = np.zeros((10, 10), dtype=np.float64)
        nms_mock[5, 5] = 160.0  # Strong (> 150)
        nms_mock[5, 6] = 80.0   # Weak (nối với [5, 5])
        nms_mock[1, 1] = 80.0   # Weak cô lập

        edges = double_threshold_hysteresis(nms_mock, low_threshold=50.0, high_threshold=150.0)

        self.assertEqual(edges[5, 5], 255, "Điểm mạnh phải giữ nguyên 255.")
        self.assertEqual(edges[5, 6], 255, "Điểm yếu kết nối với điểm mạnh phải được nâng lên 255.")
        self.assertEqual(edges[1, 1], 0, "Điểm yếu cô lập bắt buộc phải bị xóa về 0.")

    def test_full_canny_vs_opencv_iou(self):
        """Kiểm chứng toàn bộ pipeline Canny đạt độ trùng khớp IoU >= 98.0% so với cv2.Canny."""
        low_t, high_t = 50.0, 150.0
        b_cv = cv2.GaussianBlur(self.test_img, (5, 5), 1.4, borderType=cv2.BORDER_REFLECT_101)

        canny_sc = canny_edge_detector(self.test_img, low_threshold=low_t, high_threshold=high_t, preblurred=False)
        canny_cv = cv2.Canny(b_cv, int(low_t), int(high_t), L2gradient=True)

        inter = int(np.logical_and(canny_sc > 0, canny_cv > 0).sum())
        union = int(np.logical_or(canny_sc > 0, canny_cv > 0).sum())
        iou = inter / union if union > 0 else 1.0
        dice = 2.0 * inter / (int((canny_sc > 0).sum()) + int((canny_cv > 0).sum()))

        self.assertGreaterEqual(iou, 0.980, f"Canny IoU phải >= 98.0%, thực tế đạt {iou*100:.2f}%")
        self.assertGreaterEqual(dice, 0.985, f"Canny Dice phải >= 0.985, thực tế đạt {dice:.4f}")

    def test_canny_uniform_image(self):
        """Kiểm tra trường hợp biên (Edge case): Ảnh đồng nhất (không có biên nào)."""
        flat = np.full((50, 50), 128, dtype=np.uint8)
        edges = canny_edge_detector(flat)
        self.assertEqual(int(np.sum(edges)), 0, "Ảnh đồng nhất không được sinh ra bất kỳ biên nào.")


if __name__ == "__main__":
    unittest.main()
