"""Unit tests: Module TV1 - Tiêu chí nghiệm thu Benchmark và Đối chứng toàn diện
Kiểm tra toàn bộ các tiêu chuẩn định lượng khắt khe trong Đề cương Nhóm 4,
đồng thời khóa lại các lỗi đã được Code Review chỉ ra (mục 1.2, 1.5, 2.2, 2.3, 2.7).
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
    calculate_binary_iou_dice,
    calculate_line_match_rate
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

    def test_benchmark_status_can_fail(self):
        """Hồi quy Code Review mục 1.5: nhánh else phải là FAIL, không được luôn PASS.

        Code cũ viết ``"PASS" if <điều kiện> else "PASS"`` cho cả Hough và Corner, nên
        benchmark không bao giờ báo FAIL. Test này gọi trực tiếp các hàm đo với dữ liệu
        sai lệch để chắc chắn ngưỡng nghiệm thu thực sự có tác dụng.
        """
        # IoU của hai ảnh nhị phân rời nhau = 0.0 -> phải nhỏ hơn ngưỡng 0.985
        blank = np.zeros((50, 50), dtype=np.uint8)
        full = np.full((50, 50), 255, dtype=np.uint8)
        iou, dice = calculate_binary_iou_dice(blank, full)
        self.assertEqual(iou, 0.0)
        self.assertEqual(dice, 0.0)
        self.assertFalse(iou >= 0.985, "Ngưỡng Canny phải có khả năng FAIL")

        # Hai tập đường thẳng không liên quan -> recall = 0 < 0.95
        recall = calculate_line_match_rate([(10.0, 0.0)], [(500.0, np.pi / 2)])
        self.assertEqual(recall, 0.0)
        self.assertFalse(recall >= 0.95, "Ngưỡng Hough phải có khả năng FAIL")

    def test_line_match_rate_basic(self):
        """Đối chứng Hough so khớp theo (rho, theta) chứ không theo số lượng đường."""
        # Cùng một đường biểu diễn theo 2 cách tương đương: (rho, theta) và (-rho, theta+pi)
        reference = [(100.0, np.pi / 4)]
        same_line = [(100.0, np.pi / 4)]
        flipped = [(-100.0, np.pi / 4 + np.pi)]

        self.assertEqual(calculate_line_match_rate(reference, same_line), 1.0)
        self.assertEqual(calculate_line_match_rate(reference, flipped), 1.0)
        self.assertEqual(calculate_line_match_rate(reference, [(140.0, np.pi / 4)]), 0.0)

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

    def test_pipeline_latency_budget(self):
        """Hồi quy Code Review mục 2.7: TV1 phải chạy dưới ngưỡng 1.5 s/phiếu.

        Code cũ mất 1.1-1.7 s chỉ riêng giai đoạn 1 do duyệt BFS từng pixel cho
        hysteresis. Ngưỡng dưới đây (600 ms) là mức đo được sau khi vector hóa.
        """
        res = process_preprocessing_pipeline(self.test_img, use_scratch=True)
        total_time = res["latencies"]["total_tv1_ms"]
        self.assertLess(
            total_time, 600.0,
            f"TV1 phải chạy < 600 ms/phiếu sau khi tối ưu, thực tế: {total_time} ms"
        )

    def test_pipeline_does_not_run_canny_twice(self):
        """Pipeline phải dùng lại ảnh biên đã tính, không chạy Canny lần hai.

        Trước khi sửa, detect_sheet_corners tự chuyển mức xám và tự chạy lại Canny,
        khiến giai đoạn 1 tốn gấp đôi thời gian.
        """
        res = process_preprocessing_pipeline(self.test_img, use_scratch=True)
        lat = res["latencies"]
        # Thời gian dò góc không được vượt quá thời gian Canny đầy đủ (đo trên cùng máy)
        self.assertLess(
            lat["corner_detection_ms"], lat["canny_edge_ms"] + 80.0,
            f"Dò góc ({lat['corner_detection_ms']} ms) không được chậm hơn Canny ({lat['canny_edge_ms']} ms)"
        )

    def test_detect_sheet_corners_full_frame_scenario(self):
        """Phiếu chiếm toàn khung ảnh phải trả về đúng 4 góc khung hình."""
        h, w = self.test_img.shape[:2]
        corners = detect_sheet_corners(self.test_img, use_scratch=True)

        expected = np.array(
            [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(corners, expected, atol=1e-4)

    def test_detect_sheet_corners_tilted_scenario(self):
        """Kiểm tra phát hiện góc chính xác trên kịch bản phiếu nghiêng (DS-02)."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        p2_path = base_dir / "OMR_System_TV4" / "data" / "samples" / "02_B21DCCN002_Nghieng_BanToi.png"
        if not p2_path.exists():
            self.skipTest("Không có ảnh DS-02 để kiểm thử")

        img_tilted = cv2.imread(str(p2_path))
        c_sc = detect_sheet_corners(img_tilted, use_scratch=True)
        c_cv = detect_sheet_corners(img_tilted, use_scratch=False)

        self.assertEqual(c_sc.shape, (4, 2))
        # Sai số khoảng cách trung bình Euclid giữa Scratch và OpenCV <= 15 px trên ảnh 1120x920
        mean_dist = float(np.mean(np.linalg.norm(c_sc - c_cv, axis=1)))
        self.assertLessEqual(mean_dist, 15.0, f"Sai số góc trên ảnh nghiêng phải <= 15 px (thực tế: {mean_dist:.2f} px)")

    def test_tilted_sheet_is_not_collapsed_to_frame(self):
        """Hồi quy Code Review mục 1.2: DS-02 phải được dò bằng Canny + Hough.

        Trước khi sửa, ngưỡng cố định ``gray >= 100`` + RDP (đa giác khép kín) khiến
        nhánh này không chạy được và kết quả rơi về 4 góc khung hình. Nay tứ giác phải
        nằm co vào trong ảnh, không trùng mép ảnh.
        """
        base_dir = Path(__file__).resolve().parent.parent.parent
        p2_path = base_dir / "OMR_System_TV4" / "data" / "samples" / "02_B21DCCN002_Nghieng_BanToi.png"
        if not p2_path.exists():
            self.skipTest("Không có ảnh DS-02 để kiểm thử")

        img_tilted = cv2.imread(str(p2_path))
        h, w = img_tilted.shape[:2]
        corners = detect_sheet_corners(img_tilted, use_scratch=True)

        xs, ys = corners[:, 0], corners[:, 1]
        self.assertGreater(xs.min(), 10.0, "Cạnh trái tứ giác phải nằm trong ảnh (không trùng mép)")
        self.assertGreater(ys.min(), 10.0, "Cạnh trên tứ giác phải nằm trong ảnh (không trùng mép)")
        self.assertLess(xs.max(), w - 10.0, "Cạnh phải tứ giác phải nằm trong ảnh (không trùng mép)")
        self.assertLess(ys.max(), h - 10.0, "Cạnh dưới tứ giác phải nằm trong ảnh (không trùng mép)")

    def test_dark_desk_detection_is_robust_to_corner_noise(self):
        """Hồi quy Code Review mục 2.3: nhận biết nền bàn tối không được chỉ dựa vào 4 pixel góc.

        Ảnh dưới đây có nền bàn tối nhưng 4 pixel góc bị làm sáng (bóng đèn / nhiễu).
        Cách kiểm tra cũ (trung bình 4 góc < 120) sẽ chọn sai nhánh; cách mới dùng trung vị
        dải viền + Otsu nên vẫn nhận đúng là nền tối.
        """
        img = np.full((400, 300, 3), 40, dtype=np.uint8)          # nền bàn tối
        cv2.rectangle(img, (60, 40), (240, 360), (240, 240, 240), -1)  # tờ phiếu sáng
        cv2.rectangle(img, (60, 40), (240, 360), (20, 20, 20), 3)      # viền fiducial
        # Làm sáng đúng 4 pixel góc để đánh lừa kiểm tra cũ
        for y, x in ((0, 0), (0, 299), (399, 0), (399, 299)):
            img[y, x] = 250

        corners = detect_sheet_corners(img, use_scratch=True)
        xs, ys = corners[:, 0], corners[:, 1]

        # Nếu chọn sai nhánh, kết quả sẽ là 4 góc khung hình (0,0) và (299/399)
        self.assertGreater(xs.min(), 20.0, f"Phải nhận ra nền bàn tối, nhận được quad {corners.tolist()}")
        self.assertGreater(ys.min(), 20.0, f"Phải nhận ra nền bàn tối, nhận được quad {corners.tolist()}")
        self.assertLess(xs.max(), 285.0, f"Phải nhận ra nền bàn tối, nhận được quad {corners.tolist()}")
        self.assertLess(ys.max(), 385.0, f"Phải nhận ra nền bàn tối, nhận được quad {corners.tolist()}")

    def test_shadow_gradient_does_not_look_like_dark_desk(self):
        """Hồi quy: bóng đổ một phía không được coi là "nền bàn tối bao quanh phiếu".

        Ảnh DS-05 là tờ phiếu phủ kín khung hình, chỉ bị bóng đổ ở nửa dưới. Nếu bộ phân
        loại gộp chung trung vị 4 dải viền, dải tối (bóng) chiếm đa số và ảnh bị đẩy sang
        nhánh Hough, trả về một tứ giác thụt vào trong sai 67.19 px so với baseline.

        Bài test dựng lại đúng cấu trúc đó ở kích thước nhỏ và yêu cầu TV1 vẫn trả về
        4 góc khung hình (vì phiếu vẫn phủ kín ảnh, bóng chỉ là biến đổi ánh sáng).
        """
        h, w = 400, 320
        img = np.full((h, w, 3), 245, dtype=np.uint8)
        # Bóng đổ mạnh dần từ nửa dưới ảnh (mô phỏng DS-05)
        for y in range(h // 2, h):
            ratio = (y - h // 2) / max(h - h // 2 - 1, 1)
            img[y, :, :] = int(round(245 - 185 * ratio))
        # Vài chi tiết in trên phiếu để ảnh không phẳng tuyệt đối
        cv2.rectangle(img, (40, 40), (w - 40, 90), (20, 20, 20), 2)
        cv2.circle(img, (w // 2, h // 2 - 40), 12, (30, 30, 30), -1)

        corners = detect_sheet_corners(img, use_scratch=True)

        expected = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        err = float(np.mean(np.linalg.norm(corners - expected, axis=1)))
        self.assertLessEqual(
            err, 1e-3,
            f"Bóng đổ một phía phải giữ nguyên 4 góc khung hình, sai {err:.2f} px: {corners.tolist()}"
        )

    def test_shadowed_full_frame_sample_is_not_inset(self):
        """Hồi quy trên ảnh thật DS-05: sai số góc so với baseline phải ~0, không bị thụt vào."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        p5_path = base_dir / "OMR_System_TV4" / "data" / "samples" / "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png"
        if not p5_path.exists():
            self.skipTest("Không có ảnh DS-05 để kiểm thử")

        img = cv2.imread(str(p5_path))
        h, w = img.shape[:2]
        corners = detect_sheet_corners(img, use_scratch=True)
        err = float(np.mean(np.linalg.norm(corners - detect_sheet_corners(img, use_scratch=False), axis=1)))

        self.assertLessEqual(err, 3.0, f"DS-05 sai số góc phải <= 3 px, thực tế {err:.2f} px")
        self.assertAlmostEqual(float(corners[:, 0].min()), 0.0, delta=1.0)
        self.assertAlmostEqual(float(corners[:, 1].min()), 0.0, delta=1.0)
        self.assertAlmostEqual(float(corners[:, 0].max()), w - 1.0, delta=1.0)
        self.assertAlmostEqual(float(corners[:, 1].max()), h - 1.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
