"""Unit tests: Module TV1 - Biến đổi Hough & Định vị góc phiếu
Kiểm thử thuật toán Hough Line Transform, Giao điểm và Sắp xếp góc chuẩn.
"""

import unittest
import numpy as np
import cv2

import sys
from pathlib import Path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tv1_hough import (
    hough_lines_standard,
    intersect_lines_polar,
    order_quad_corners,
    convex_hull_2d,
    ramer_douglas_peucker,
    simplify_closed_polygon,
    apply_parameter_space_nms,
    otsu_threshold_scratch,
    collect_boundary_extremes,
    score_lines_by_edge_support,
    detect_quad_from_hough_lines,
    detect_quad_from_boundary_scratch
)


class TestTV1Hough(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho Biến đổi Hough và Xác định hình học 4 đỉnh."""

    def test_hough_lines_rectangle(self):
        """Kiểm chứng Biến đổi Hough phát hiện đúng 4 cạnh hình chữ nhật."""
        # Tạo ảnh nhị phân chứa khung chữ nhật 100x80
        edge_img = np.zeros((150, 150), dtype=np.uint8)
        # 2 cạnh ngang tại y=30 và y=110
        edge_img[30, 25:125] = 255
        edge_img[110, 25:125] = 255
        # 2 cạnh dọc tại x=25 và x=124
        edge_img[30:111, 25] = 255
        edge_img[30:111, 124] = 255

        lines, acc = hough_lines_standard(edge_img, rho_res=1.0, theta_res=np.pi / 180.0, threshold=70)

        self.assertGreaterEqual(len(lines), 4, f"Hough Transform phải tìm được ít nhất 4 đường, tìm được: {len(lines)}")

        # Kiểm tra tồn tại cả đường thẳng đứng (theta gần 0) và đường nằm ngang (theta gần pi/2)
        has_vertical = any(abs(l[1]) < np.pi / 18 or abs(l[1] - np.pi) < np.pi / 18 for l in lines)
        has_horizontal = any(abs(l[1] - np.pi / 2) < np.pi / 18 for l in lines)

        self.assertTrue(has_vertical, "Phải phát hiện được đường biên dọc.")
        self.assertTrue(has_horizontal, "Phải phát hiện được đường biên ngang.")

    def test_intersect_lines_polar(self):
        """Kiểm tra giao điểm giải tích của 2 đường thẳng cực."""
        # Đường thẳng đứng: x = 50 (rho=50, theta=0)
        # Đường thẳng ngang: y = 80 (rho=80, theta=pi/2)
        line_v = (50.0, 0.0)
        line_h = (80.0, np.pi / 2.0)

        pt = intersect_lines_polar(line_v, line_h)
        self.assertIsNotNone(pt)
        self.assertAlmostEqual(pt[0], 50.0, places=3)
        self.assertAlmostEqual(pt[1], 80.0, places=3)

        # Hai đường song song (det = 0)
        line_v2 = (120.0, 0.0)
        self.assertIsNone(intersect_lines_polar(line_v, line_v2))

    def test_order_quad_corners(self):
        """Kiểm chứng chuẩn hóa thứ tự 4 góc [TL, TR, BR, BL]."""
        # Thứ tự chuẩn mong muốn
        true_tl = [20.0, 30.0]
        true_tr = [400.0, 35.0]
        true_br = [390.0, 500.0]
        true_bl = [25.0, 490.0]

        # Xáo trộn thứ tự
        shuffled = np.array([true_br, true_tl, true_bl, true_tr], dtype=np.float32)

        ordered = order_quad_corners(shuffled)

        np.testing.assert_allclose(ordered[0], true_tl, atol=1e-5, err_msg="Góc TL sai thứ tự")
        np.testing.assert_allclose(ordered[1], true_tr, atol=1e-5, err_msg="Góc TR sai thứ tự")
        np.testing.assert_allclose(ordered[2], true_br, atol=1e-5, err_msg="Góc BR sai thứ tự")
        np.testing.assert_allclose(ordered[3], true_bl, atol=1e-5, err_msg="Góc BL sai thứ tự")

    def test_convex_hull_2d(self):
        """Kiểm chứng thuật toán Bao lồi Andrew's Monotone Chain."""
        # Tập điểm gồm 4 góc hình vuông và nhiều điểm ngẫu nhiên bên trong
        square_corners = [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]
        interior_points = [[50.0, 50.0], [30.0, 40.0], [70.0, 60.0], [45.0, 55.0]]
        all_pts = np.array(square_corners + interior_points, dtype=np.float32)

        hull = convex_hull_2d(all_pts)
        self.assertEqual(len(hull), 4, f"Bao lồi phải chỉ chứa đúng 4 đỉnh ngoài, nhận được: {len(hull)}")

    def test_ramer_douglas_peucker(self):
        """Kiểm chứng rút gọn đa giác bằng thuật toán RDP."""
        # Tạo đường thẳng có các điểm lượn sóng nhỏ < epsilon
        x = np.linspace(0, 100, 50)
        y = 2.0 * x + np.sin(x) * 0.5  # Nhiễu biên độ 0.5
        pts = np.column_stack([x, y]).astype(np.float32)

        simplified = ramer_douglas_peucker(pts, epsilon=2.0)
        # Sau khi rút gọn với epsilon=2.0, chỉ còn 2 điểm đầu và cuối
        self.assertEqual(len(simplified), 2, "RDP phải rút gọn đường thẳng gần như thẳng về 2 đầu mút.")

    def test_rdp_rejects_closed_polygon(self):
        """Hồi quy Code Review mục 2.2: RDP không được nhận đa giác khép kín.

        Trước khi sửa, code đưa đa giác khép kín (điểm đầu trùng điểm cuối) vào RDP nên
        "đoạn thẳng" tham chiếu suy biến thành 1 điểm, khoảng cách luôn bằng 0 và hàm
        luôn trả về 2 điểm -> nhánh RDP trong detect_quad_from_boundary_scratch
        không bao giờ chạy được (thực tế luôn rơi xuống nhánh dự phòng 4 điểm cực trị).
        """
        closed_square = np.array([
            [0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]
        ])

        with self.assertRaises(ValueError):
            ramer_douglas_peucker(closed_square, epsilon=1.0)

    def test_simplify_closed_polygon_keeps_four_corners(self):
        """RDP trên đa giác khép kín phải giữ đúng 4 đỉnh của hình vuông."""
        # Hình vuông có thêm nhiều điểm nằm trên cạnh (giả lập điểm biên dày đặc)
        t = np.linspace(0.0, 1.0, 12)
        bottom = np.column_stack([t * 100.0, np.zeros_like(t)])
        right = np.column_stack([np.full_like(t, 100.0), t * 100.0])
        top = np.column_stack([(1.0 - t) * 100.0, np.full_like(t, 100.0)])
        left = np.column_stack([np.zeros_like(t), (1.0 - t) * 100.0])
        polygon = np.vstack([bottom, right, top, left])

        simplified = simplify_closed_polygon(polygon, epsilon=2.0)
        ordered = order_quad_corners(simplified[:4]) if len(simplified) >= 4 else simplified

        self.assertEqual(len(simplified), 4, f"Phải rút gọn còn đúng 4 đỉnh, nhận được {len(simplified)}")
        np.testing.assert_allclose(
            np.sort(ordered, axis=0),
            np.sort(np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32), axis=0),
            atol=1e-4
        )

    def test_collect_boundary_extremes_keeps_corners(self):
        """Lấy mẫu biên cực trị theo hàng/cột phải chứa đúng 4 đỉnh của tứ giác."""
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[5:55, 9:71] = 255

        pts = collect_boundary_extremes(mask)

        self.assertGreaterEqual(len(pts), 4)
        self.assertEqual(pts[:, 0].min(), 9.0, "Phải giữ được cạnh trái x=9")
        self.assertEqual(pts[:, 0].max(), 70.0, "Phải giữ được cạnh phải x=70")
        self.assertEqual(pts[:, 1].min(), 5.0, "Phải giữ được cạnh trên y=5")
        self.assertEqual(pts[:, 1].max(), 54.0, "Phải giữ được cạnh dưới y=54")

    def test_otsu_threshold_matches_opencv(self):
        """Ngưỡng Otsu tự cài đặt phải khớp cv2.threshold(THRESH_OTSU)."""
        rng = np.random.default_rng(7)
        img = np.concatenate([
            rng.integers(10, 90, size=4000),
            rng.integers(170, 250, size=6000),
        ]).astype(np.uint8).reshape(100, 100)

        t_sc = otsu_threshold_scratch(img)
        t_cv, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.assertLessEqual(abs(t_sc - int(t_cv)), 1, f"Ngưỡng Otsu lệch quá 1 mức xám: {t_sc} vs {t_cv}")

    def test_detect_quad_from_hough_lines_rectangle(self):
        """Hough + Cramer phải dựng đúng 4 góc của khung chữ nhật dày."""
        edge_img = np.zeros((300, 250), dtype=np.uint8)
        # Khung chữ nhật dày 3 px để Canny/Hough có đủ điểm biên
        edge_img[40:44, 30:221] = 255
        edge_img[256:260, 30:221] = 255
        edge_img[40:260, 30:34] = 255
        edge_img[40:260, 217:221] = 255

        lines, _ = hough_lines_standard(edge_img, rho_res=1.0, theta_res=np.pi / 180.0, threshold=100)
        quad = detect_quad_from_hough_lines(lines, edge_img, edge_img.shape)

        self.assertIsNotNone(quad, "Phải dựng được tứ giác từ 4 cạnh mạnh nhất")
        expected = np.array([[32.0, 42.0], [219.0, 42.0], [219.0, 258.0], [32.0, 258.0]], dtype=np.float32)
        err = float(np.mean(np.linalg.norm(quad - expected, axis=1)))
        self.assertLess(err, 6.0, f"Sai số góc quá lớn: {err:.2f} px, quad={quad.tolist()}")

    def test_apply_parameter_space_nms_merges_duplicates(self):
        """NMS tham số phải gộp các đường gần trùng và giữ đường mạnh nhất."""
        lines = [
            (100.0, np.pi / 2),
            (102.0, np.pi / 2 + np.deg2rad(1.0)),   # gần trùng -> bị gộp
            (300.0, 0.0),                            # khác hẳn -> giữ
        ]
        kept = apply_parameter_space_nms(lines)
        self.assertEqual(len(kept), 2, f"Phải gộp còn 2 đường, nhận được {len(kept)}")
        self.assertAlmostEqual(kept[0][0], 100.0, places=3)


if __name__ == "__main__":
    unittest.main()
