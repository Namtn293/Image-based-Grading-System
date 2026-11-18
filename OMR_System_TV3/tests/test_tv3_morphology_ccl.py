"""
Unit tests cho Module tv3_morphology_ccl.py
Thành viên 3: Toán tử Hình thái học nhị phân & Gán nhãn thành phần liên thông Two-Pass CCL
Đề tài: Hệ thống Chấm bài Trắc nghiệm Tự động từ Ảnh chụp (OMR) - Nhóm 4
"""

import sys
import os
import unittest
import numpy as np
import cv2

# Thêm thư mục src vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tv3_morphology_ccl import (
    create_structuring_element,
    erode_binary_scratch,
    dilate_binary_scratch,
    opening_binary_scratch,
    closing_binary_scratch,
    morphology_binary_scratch,
    DisjointSet,
    two_pass_ccl_scratch,
    filter_bubble_components,
    calc_binary_metrics,
    benchmark_morphology_vs_opencv,
    benchmark_ccl_vs_opencv,
)


class TestStructuringElement(unittest.TestCase):
    """Kiểm thử việc tạo phần tử cấu trúc (Structuring Element) từ đầu bằng NumPy."""

    def test_rect_kernel_matches_opencv(self):
        for ksize in [(3, 3), (5, 5), (7, 7), (3, 5), (5, 3)]:
            sc_k = create_structuring_element("RECT", ksize)
            kw, kh = ksize[1], ksize[0]
            cv_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            np.testing.assert_array_equal(sc_k, cv_k, err_msg=f"RECT kernel không khớp OpenCV tại ksize={ksize}")

    def test_cross_kernel_matches_opencv(self):
        for ksize in [(3, 3), (5, 5), (7, 7), (5, 3)]:
            sc_k = create_structuring_element("CROSS", ksize)
            kw, kh = ksize[1], ksize[0]
            cv_k = cv2.getStructuringElement(cv2.MORPH_CROSS, (kw, kh))
            np.testing.assert_array_equal(sc_k, cv_k, err_msg=f"CROSS kernel không khớp OpenCV tại ksize={ksize}")

    def test_ellipse_kernel_matches_opencv(self):
        for ksize in [(3, 3), (5, 5), (7, 7), (9, 9), (5, 3), (7, 5)]:
            sc_k = create_structuring_element("ELLIPSE", ksize)
            kw, kh = ksize[1], ksize[0]
            cv_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
            np.testing.assert_array_equal(sc_k, cv_k, err_msg=f"ELLIPSE kernel không khớp OpenCV tại ksize={ksize}")

    def test_invalid_parameters(self):
        # Kernel chẵn hoặc âm phải ném ngoại lệ ValueError
        with self.assertRaises(ValueError):
            create_structuring_element("RECT", (4, 4))
        with self.assertRaises(ValueError):
            create_structuring_element("RECT", (-3, 3))
        with self.assertRaises(ValueError):
            create_structuring_element("UNKNOWN_SHAPE", (3, 3))


class TestBinaryMorphology(unittest.TestCase):
    """Kiểm thử 4 phép toán Hình thái học nhị phân (Erosion, Dilation, Opening, Closing)."""

    def setUp(self):
        np.random.seed(42)
        # Tạo ảnh nhị phân mẫu đa dạng: khối chữ nhật, hình tròn, nét đứt, lỗ thủng và nhiễu
        self.H, self.W = 120, 150
        self.img = np.zeros((self.H, self.W), dtype=np.uint8)

        # Khối chữ nhật đặc
        self.img[20:50, 20:60] = 255
        # Lỗ thủng bên trong khối chữ nhật
        self.img[30:35, 35:45] = 0

        # Khối thứ 2
        self.img[70:100, 80:130] = 255

        # Nhiễu hạt muối (1 pixel cô lập)
        self.img[10, 10] = 255
        self.img[10, 80] = 255
        self.img[105, 25] = 255

        # Nét mảnh hẹp
        self.img[60, 20:50] = 255

    def test_erosion_vs_opencv(self):
        for shape in ["RECT", "CROSS", "ELLIPSE"]:
            for ksize in [(3, 3), (5, 5)]:
                k_sc = create_structuring_element(shape, ksize)
                kw, kh = ksize[1], ksize[0]
                cv_kshape = {
                    "RECT": cv2.MORPH_RECT,
                    "CROSS": cv2.MORPH_CROSS,
                    "ELLIPSE": cv2.MORPH_ELLIPSE
                }[shape]
                k_cv = cv2.getStructuringElement(cv_kshape, (kw, kh))

                sc_erode = erode_binary_scratch(self.img, k_sc)
                cv_erode = cv2.erode(self.img, k_cv)

                metrics = calc_binary_metrics(sc_erode, cv_erode)
                self.assertEqual(metrics["diff_pixels"], 0, f"Erosion khác OpenCV với shape={shape}, ksize={ksize}")
                self.assertEqual(metrics["iou_percent"], 100.0)
                self.assertEqual(metrics["mse"], 0.0)

    def test_dilation_vs_opencv(self):
        for shape in ["RECT", "CROSS", "ELLIPSE"]:
            for ksize in [(3, 3), (5, 5)]:
                k_sc = create_structuring_element(shape, ksize)
                kw, kh = ksize[1], ksize[0]
                cv_kshape = {
                    "RECT": cv2.MORPH_RECT,
                    "CROSS": cv2.MORPH_CROSS,
                    "ELLIPSE": cv2.MORPH_ELLIPSE
                }[shape]
                k_cv = cv2.getStructuringElement(cv_kshape, (kw, kh))

                sc_dilate = dilate_binary_scratch(self.img, k_sc)
                cv_dilate = cv2.dilate(self.img, k_cv)

                metrics = calc_binary_metrics(sc_dilate, cv_dilate)
                self.assertEqual(metrics["diff_pixels"], 0, f"Dilation khác OpenCV với shape={shape}, ksize={ksize}")
                self.assertEqual(metrics["iou_percent"], 100.0)
                self.assertEqual(metrics["mse"], 0.0)

    def test_opening_removes_noise_and_matches_opencv(self):
        k_sc = create_structuring_element("RECT", (3, 3))
        k_cv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        sc_open = opening_binary_scratch(self.img, k_sc)
        cv_open = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, k_cv)

        # Phép mở phải loại bỏ các điểm nhiễu cô lập (10, 10), (10, 80)
        self.assertEqual(sc_open[10, 10], 0)
        self.assertEqual(sc_open[10, 80], 0)

        metrics = calc_binary_metrics(sc_open, cv_open)
        self.assertEqual(metrics["diff_pixels"], 0)
        self.assertEqual(metrics["iou_percent"], 100.0)

    def test_closing_fills_holes_and_matches_opencv(self):
        k_sc = create_structuring_element("RECT", (7, 7))
        k_cv = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        sc_close = closing_binary_scratch(self.img, k_sc)
        cv_close = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, k_cv)

        # Phép đóng với kernel 7x7 phải lấp kín lỗ thủng (30:35, 35:45)
        hole_region = sc_close[31:34, 37:43]
        self.assertTrue(np.all(hole_region == 255), "Phép đóng chưa lấp đầy lỗ thủng trong ô!")

        metrics = calc_binary_metrics(sc_close, cv_close)
        self.assertEqual(metrics["diff_pixels"], 0)
        self.assertEqual(metrics["iou_percent"], 100.0)

    def test_morphology_scale_0_1(self):
        # Kiểm tra hoạt động với ảnh nhị phân dải {0, 1}
        img_01 = (self.img > 0).astype(np.uint8)
        k = create_structuring_element("RECT", (3, 3))

        res_erode = erode_binary_scratch(img_01, k)
        res_dilate = dilate_binary_scratch(img_01, k)

        self.assertIn(res_erode.max(), (0, 1))
        self.assertIn(res_dilate.max(), (0, 1))
        # Khớp tuyệt đối với tỷ lệ scale
        np.testing.assert_array_equal(res_erode * 255, erode_binary_scratch(self.img, k))


class TestDisjointSet(unittest.TestCase):
    """Kiểm thử cấu trúc Disjoint-Set (Union-Find)."""

    def test_disjoint_set_basic(self):
        ds = DisjointSet(5)
        for i in range(5):
            self.assertEqual(ds.find(i), i)

        ds.union(1, 2)
        self.assertEqual(ds.find(1), ds.find(2))
        self.assertEqual(ds.find(2), 1)  # Ưu tiên nhãn nhỏ hơn

        ds.union(3, 4)
        self.assertEqual(ds.find(3), ds.find(4))

        ds.union(2, 4)
        # Cả 1, 2, 3, 4 phải cùng chung gốc là 1
        self.assertEqual(ds.find(1), 1)
        self.assertEqual(ds.find(2), 1)
        self.assertEqual(ds.find(3), 1)
        self.assertEqual(ds.find(4), 1)
        self.assertEqual(ds.find(0), 0)  # 0 vẫn đứng riêng


class TestTwoPassCCL(unittest.TestCase):
    """Kiểm thử thuật toán Two-Pass Connected Component Labeling."""

    def test_empty_image(self):
        img_zeros = np.zeros((50, 50), dtype=np.uint8)
        K, labels, stats = two_pass_ccl_scratch(img_zeros, connectivity=8)
        self.assertEqual(K, 0)
        self.assertEqual(labels.max(), 0)
        self.assertEqual(len(stats), 0)

    def test_single_component(self):
        img = np.zeros((30, 30), dtype=np.uint8)
        img[5:15, 5:25] = 255
        K, labels, stats = two_pass_ccl_scratch(img, connectivity=8)
        self.assertEqual(K, 1)
        self.assertEqual(stats[0]["area"], 10 * 20)
        self.assertEqual(stats[0]["bbox"], (5, 5, 20, 10))
        self.assertEqual(stats[0]["centroid"], (14.5, 9.5))
        self.assertAlmostEqual(stats[0]["extent"], 1.0)
        self.assertAlmostEqual(stats[0]["aspect_ratio"], 2.0)

    def test_diagonal_connectivity_difference(self):
        # 2 điểm chỉ chạm nhau ở đường chéo:
        # Trong 4-lân cận: tách làm 2 thành phần.
        # Trong 8-lân cận: gộp thành 1 thành phần duy nhất.
        img = np.zeros((5, 5), dtype=np.uint8)
        img[1, 1] = 255
        img[2, 2] = 255

        K4, labels4, _ = two_pass_ccl_scratch(img, connectivity=4)
        K8, labels8, _ = two_pass_ccl_scratch(img, connectivity=8)

        self.assertEqual(K4, 2, "4-lân cận phải tách 2 điểm chạm chéo thành 2 thành phần.")
        self.assertEqual(K8, 1, "8-lân cận phải nối 2 điểm chạm chéo thành 1 thành phần.")

    def test_ccl_matches_opencv_random_image(self):
        np.random.seed(123)
        H, W = 80, 100
        # Tạo ảnh có nhiều hình khối và đốm rời rạc
        img = (np.random.rand(H, W) > 0.8).astype(np.uint8) * 255
        img[10:25, 10:25] = 255
        img[40:60, 50:80] = 255

        for conn in [4, 8]:
            num_cv, labels_cv, stats_cv, cents_cv = cv2.connectedComponentsWithStats(img, connectivity=conn)
            K_sc, labels_sc, stats_sc = two_pass_ccl_scratch(img, connectivity=conn)

            # 1. Số lượng thành phần liên thông trùng khớp 100%
            self.assertEqual(K_sc, num_cv - 1, f"Sai số lượng thành phần với connectivity={conn}")

            # 2. Đối chứng từng thành phần: IoU = 1.0, Bbox và Centroid khớp chính xác
            stats_map = {item["label"]: item for item in stats_sc}
            for cv_id in range(1, num_cv):
                pts = np.argwhere(labels_cv == cv_id)
                sc_id = labels_sc[pts[0, 0], pts[0, 1]]

                mask_cv = (labels_cv == cv_id)
                mask_sc = (labels_sc == sc_id)
                iou = np.sum(mask_cv & mask_sc) / np.sum(mask_cv | mask_sc)
                self.assertEqual(iou, 1.0, f"Thành phần {cv_id} không khớp mặt nạ (IoU={iou})")

                sc_stat = stats_map[sc_id]
                self.assertEqual(sc_stat["area"], int(stats_cv[cv_id, cv2.CC_STAT_AREA]))
                cv_bbox = tuple(stats_cv[cv_id, 0:4])
                self.assertEqual(sc_stat["bbox"], cv_bbox)

                cv_cent = (float(cents_cv[cv_id, 0]), float(cents_cv[cv_id, 1]))
                np.testing.assert_allclose(sc_stat["centroid"], cv_cent, atol=1e-4)


class TestFilterBubbleComponents(unittest.TestCase):
    """Kiểm thử hàm lọc ứng viên ô tròn trắc nghiệm (filter_bubble_components)."""

    def test_bubble_filtering(self):
        img = np.zeros((200, 200), dtype=np.uint8)

        # 3 ô tròn trắc nghiệm chuẩn (bán kính R=8, diện tích ~201 px, AR ~ 1.0, extent ~ 0.75)
        for (cy, cx) in [(40, 40), (40, 100), (40, 160)]:
            y, x = np.ogrid[-8:9, -8:9]
            mask = x**2 + y**2 <= 64
            img[cy-8:cy+9, cx-8:cx+9][mask] = 255

        # 1 đường kẻ bảng ngang dài (bị loại do AR quá cao)
        img[120:122, 10:190] = 255

        # 3 chấm nhiễu muối (bị loại do diện tích quá nhỏ)
        img[10, 10] = 255
        img[15, 25] = 255
        img[180, 50] = 255

        K, labels, stats = two_pass_ccl_scratch(img, connectivity=8)
        self.assertEqual(K, 7)

        filt_labels, filt_stats = filter_bubble_components(
            labels, stats, min_area=50, max_area=500, aspect_ratio_range=(0.7, 1.4), min_extent=0.5
        )

        # Chỉ 3 ô tròn được giữ lại
        self.assertEqual(len(filt_stats), 3)
        self.assertEqual(filt_labels.max(), 3)
        for bubble in filt_stats:
            self.assertAlmostEqual(bubble["aspect_ratio"], 1.0, delta=0.2)
            self.assertGreaterEqual(bubble["extent"], 0.65)


class TestBenchmarkFunctions(unittest.TestCase):
    """Kiểm thử các hàm đo đạc hiệu năng và đối chứng OpenCV."""

    def test_benchmark_morphology_vs_opencv(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:70, 20:70] = 255
        res = benchmark_morphology_vs_opencv(img, kernel_size=(3, 3), op="all", num_runs=2)

        self.assertIn("erode", res)
        self.assertIn("dilate", res)
        self.assertIn("open", res)
        self.assertIn("close", res)

        for op_name, metrics in res.items():
            self.assertGreaterEqual(metrics["iou_percent"], 99.0)
            self.assertLess(metrics["mse"], 0.01)
            self.assertIn("time_scratch_ms", metrics)
            self.assertIn("time_opencv_ms", metrics)

    def test_benchmark_ccl_vs_opencv(self):
        img = np.zeros((80, 80), dtype=np.uint8)
        img[10:30, 10:30] = 255
        img[40:70, 40:70] = 255

        ccl_res = benchmark_ccl_vs_opencv(img, connectivity=8, num_runs=2)
        self.assertTrue(ccl_res["count_match"])
        self.assertEqual(ccl_res["scratch_component_count"], 2)
        self.assertEqual(ccl_res["opencv_component_count"], 2)
        self.assertEqual(ccl_res["mean_iou_percent"], 100.0)
        self.assertEqual(ccl_res["max_bbox_delta_px"], 0)
        self.assertLess(ccl_res["max_centroid_delta_px"], 1e-4)

    def test_ccl_noisy_image_performance_and_psnr_consistency(self):
        """Kiểm thử hiệu năng CCL O(K+L) trên ảnh nhiễu 15% và chuẩn PSNR=999.0 khi trùng khớp."""
        import time

        np.random.seed(7)
        noisy = (np.random.rand(250, 250) > 0.85).astype(np.uint8) * 255
        t0 = time.perf_counter()
        k_sc, labels_sc, _ = two_pass_ccl_scratch(noisy, connectivity=8)
        elapsed = time.perf_counter() - t0

        num_cv, _, _, _ = cv2.connectedComponentsWithStats(noisy, connectivity=8)
        self.assertEqual(k_sc, num_cv - 1)
        self.assertLess(elapsed, 0.5, f"CCL quá chậm trên ảnh nhiễu 15%: {elapsed:.3f}s")

        # PSNR khi hai ảnh trùng khớp phải thống nhất bằng 999.0
        metrics_same = calc_binary_metrics(noisy, noisy)
        self.assertEqual(metrics_same["psnr_db"], 999.0)


if __name__ == "__main__":
    unittest.main()
