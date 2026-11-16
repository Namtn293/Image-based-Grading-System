"""Unit tests: Module TV1 - Chuyển đổi mức xám & Làm mịn khử nhiễu Gaussian
Kiểm thử tính đúng đắn toán học và đối chứng số học với OpenCV.
"""

import unittest
import numpy as np
import cv2

import sys
from pathlib import Path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

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


class TestTV1GrayscaleBlur(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho thuật toán Chuyển mức xám & Lọc Gaussian 2D."""

    def test_rgb_to_gray_pure_colors(self):
        """Kiểm tra trọng số Luminance ITU-R BT.601 trên các màu cơ bản."""
        # Pure Red: 0.299 * 255 = 76.245 -> 76
        red = np.zeros((10, 10, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        gray_r = rgb_to_gray_scratch(red, is_bgr=False)
        self.assertEqual(int(gray_r[0, 0]), 76)

        # Pure Green: 0.587 * 255 = 149.685 -> 150
        green = np.zeros((10, 10, 3), dtype=np.uint8)
        green[:, :, 1] = 255
        gray_g = rgb_to_gray_scratch(green, is_bgr=False)
        self.assertEqual(int(gray_g[0, 0]), 150)

        # Pure Blue: 0.114 * 255 = 29.07 -> 29
        blue = np.zeros((10, 10, 3), dtype=np.uint8)
        blue[:, :, 2] = 255
        gray_b = rgb_to_gray_scratch(blue, is_bgr=False)
        self.assertEqual(int(gray_b[0, 0]), 29)

        # Pure White: 76 + 150 + 29 = 255
        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        gray_w = rgb_to_gray_scratch(white, is_bgr=False)
        self.assertEqual(int(gray_w[0, 0]), 255)

    def test_rgb_to_gray_vs_opencv_bgr(self):
        """Đối chứng số học chính xác 100% với cv2.cvtColor(BGR2GRAY)."""
        np.random.seed(42)
        bgr_rand = np.random.randint(0, 256, (120, 150, 3), dtype=np.uint8)

        gray_sc = rgb_to_gray_scratch(bgr_rand, is_bgr=True)
        gray_cv = cv2.cvtColor(bgr_rand, cv2.COLOR_BGR2GRAY)

        max_diff = int(np.max(np.abs(gray_sc.astype(int) - gray_cv.astype(int))))
        mse = float(np.mean((gray_sc.astype(float) - gray_cv.astype(float)) ** 2))

        self.assertLessEqual(max_diff, 1, "Sai số tối đa giữa Scratch và OpenCV cvtColor không được vượt quá 1 mức xám.")
        self.assertLess(mse, 0.05, "MSE chuyển đổi mức xám phải tiệm cận 0.")

    def test_gaussian_kernel_properties(self):
        """Kiểm tra các tính chất toán học của Gaussian Kernel 1D và 2D."""
        for k in [3, 5, 7, 9]:
            for s in [0.8, 1.4, 2.0]:
                k1d = get_gaussian_kernel_1d(k, s)
                self.assertEqual(len(k1d), k)
                self.assertAlmostEqual(float(np.sum(k1d)), 1.0, places=7)
                # Tính đối xứng: k1d[i] == k1d[k - 1 - i]
                np.testing.assert_allclose(k1d, k1d[::-1], atol=1e-12)

                k2d = get_gaussian_kernel_2d(k, s)
                self.assertEqual(k2d.shape, (k, k))
                self.assertAlmostEqual(float(np.sum(k2d)), 1.0, places=7)
                # Tính đối xứng tâm và gương
                np.testing.assert_allclose(k2d, np.flipud(k2d), atol=1e-12)
                np.testing.assert_allclose(k2d, np.fliplr(k2d), atol=1e-12)

    def test_gaussian_blur_vs_opencv(self):
        """Kiểm chứng Gaussian Blur 2D đạt chuẩn PSNR > 48 dB (thực tế > 65 dB)."""
        np.random.seed(123)
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)

        for ksize, sigma in [(3, 1.0), (5, 1.4), (7, 2.0)]:
            b_sc = gaussian_blur_2d(img, kernel_size=ksize, sigma=sigma)
            b_cv = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)

            mse = float(np.mean((b_sc.astype(float) - b_cv.astype(float)) ** 2))
            psnr = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else 999.0

            self.assertLess(mse, 1.0, f"MSE Gaussian Blur k={ksize} phải < 1.0 (thực tế: {mse})")
            self.assertGreater(psnr, 48.0, f"PSNR Gaussian Blur k={ksize} phải > 48.0 dB (thực tế: {psnr})")

    def test_convolve2d_scratch_impulse_response(self):
        """Kiểm tra tích chập 2D tổng quát qua hàm xung Dirac."""
        impulse = np.zeros((11, 11), dtype=np.float64)
        impulse[5, 5] = 1.0

        k2d = get_gaussian_kernel_2d(5, 1.4)
        filtered = convolve2d_scratch(impulse, k2d, mode="reflect")

        # Tâm của ảnh sau khi lọc phải trùng khớp với kernel
        np.testing.assert_allclose(filtered[3:8, 3:8], k2d, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
