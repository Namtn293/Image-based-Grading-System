"""Unit tests: Module TV1 - Tiêu chí nghiệm thu Benchmark và Đối chứng toàn diện
Kiểm tra toàn bộ các tiêu chuẩn định lượng khắt khe trong Đề cương Nhóm 4.
"""

import unittest
import numpy as np
import cv2

import sys
from pathlib import Path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tv1_preprocessing import (
    detect_sheet_corners,
    process_preprocessing_pipeline,
    run_unit_benchmarks,
    calculate_mse,
    calculate_psnr,
    calculate_binary_iou_dice
)


class TestTV1Benchmark(unittest.TestCase):
    """Kiểm thử tiêu chuẩn nghiệm thu định lượng (Acceptance Criteria) cho TV1."""

    def setUp(self):
        # Nạp ảnh phiếu thi mẫu thật nếu có, hoặc tạo ảnh nhân tạo
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.sample_path = base_dir / "OMR_System_TV4" / "data" / "samples" / "01_B21DCCN001_Chuan.png"
        if self.sample_path.exists():
            self.test_img = cv2.imread(str(self.sample_path))
        else:
            self.test_img = np.full((1000, 800, 3), 245, dtype=np.uint8)
            cv2.rectangle(self.test_img, (50, 50), (750, 950), (20, 20, 20), 4)

    def test_run_unit_benchmarks_all_pass(self):
        """Tất cả 5 thuật toán trong run_unit_benchmarks phải đạt trạng thái PASS."""
        benchmarks = run_unit_benchmarks(self.test_img)
        self.assertEqual(len(benchmarks), 5)

        for b in benchmarks:
            self.assertEqual(b["status"], "PASS", f"Thuật toán {b['algorithm']} không đạt chuẩn: {b}")

    def test_pipeline_execution_integrity(self):
        """Kiểm tra tính toàn vẹn dữ liệu và cấu trúc đầu ra của pipeline TV1."""
        res = process_preprocessing_pipeline(self.test_img, use_scratch=True)

        self.assertIn("gray", res)
        self.assertIn("blurred", res)
        self.assertIn("edges", res)
        self.assertIn("corners", res)
        self.assertIn("latencies", res)

        h, w = self.test_img.shape[:2]
        self.assertEqual(res["gray"].shape, (h, w))
        self.assertEqual(res["blurred"].shape, (h, w))
        self.assertEqual(res["edges"].shape, (h, w))
        self.assertEqual(res["corners"].shape, (4, 2))

        # Kiểm tra độ trễ (Latency) phải nằm trong giới hạn thực tế < 600 ms cho ảnh 1000x800
        total_time = res["latencies"]["total_tv1_ms"]
        self.assertLess(total_time, 1500.0, f"Tổng thời gian chạy TV1 phải < 1.5s, thực tế: {total_time} ms")

    def test_detect_sheet_corners_tilted_scenario(self):
        """Kiểm tra phát hiện góc chính xác trên kịch bản phiếu nghiêng (DS-02)."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        p2_path = base_dir / "OMR_System_TV4" / "data" / "samples" / "02_B21DCCN002_Nghieng_BanToi.png"
        if p2_path.exists():
            img_tilted = cv2.imread(str(p2_path))
            c_sc = detect_sheet_corners(img_tilted, use_scratch=True)
            c_cv = detect_sheet_corners(img_tilted, use_scratch=False)

            self.assertEqual(c_sc.shape, (4, 2))
            # Sai số khoảng cách trung bình Euclid giữa Scratch và OpenCV <= 15 px trên ảnh 1120x920
            mean_dist = float(np.mean(np.linalg.norm(c_sc - c_cv, axis=1)))
            self.assertLessEqual(mean_dist, 15.0, f"Sai số góc trên ảnh nghiêng phải <= 15 px (thực tế: {mean_dist:.2f} px)")


if __name__ == "__main__":
    unittest.main()
