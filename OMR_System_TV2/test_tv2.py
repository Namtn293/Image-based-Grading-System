"""
Script kiểm thử độc lập (Unit Test & Benchmark) cho Module TV2:
So sánh đối chứng từng hàm tự code (NumPy from Scratch) với OpenCV (cv2).
"""

import time
import numpy as np
import cv2

# Import module tự code from scratch của TV2
from tv2_perspective import (
    order_corners,
    solve_homography_svd,
    warp_perspective_bilinear,
    match_template_ncc,
    detect_markers_and_correct_rotation,
    process_perspective_pipeline,
)


def compute_metrics(img_scratch: np.ndarray, img_cv: np.ndarray):
    """Tính MSE và PSNR giữa hai ảnh."""
    diff = img_scratch.astype(np.float64) - img_cv.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    return mse, psnr


def test_order_corners():
    print("=" * 60)
    print("1. KIỂM THỬ: order_corners()")

    # Giả lập 4 đỉnh bị xáo trộn thứ tự: [BR, TL, BL, TR]
    scrambled = np.array([
        [480, 640],  # BR
        [20, 30],    # TL
        [25, 620],   # BL
        [490, 40],   # TR
    ], dtype=np.float32)

    ordered, (w_dst, h_dst) = order_corners(scrambled)

    print(f"  Điểm đầu vào (xáo trộn):\n{scrambled}")
    print(f"  Kết quả sắp xếp [TL, TR, BR, BL]:\n{ordered}")
    print(f"  Kích thước ước tính: W_dst = {w_dst}, H_dst = {h_dst}")

    # Kiểm tra
    assert np.allclose(ordered[0], [20, 30]), "Lỗi: TL không đúng!"
    assert np.allclose(ordered[1], [490, 40]), "Lỗi: TR không đúng!"
    assert np.allclose(ordered[2], [480, 640]), "Lỗi: BR không đúng!"
    assert np.allclose(ordered[3], [25, 620]), "Lỗi: BL không đúng!"
    print("  --> [PASS] order_corners() hoạt động chính xác 100%!")


def test_solve_homography():
    print("=" * 60)
    print("2. KIỂM THỬ: solve_homography_svd() vs cv2.getPerspectiveTransform()")

    # 4 điểm nguồn (tứ giác nghiêng)
    src = np.array([
        [50.0, 60.0],
        [450.0, 30.0],
        [470.0, 580.0],
        [40.0, 550.0]
    ], dtype=np.float32)

    # 4 điểm đích (hình chữ nhật chuẩn)
    dst = np.array([
        [0.0, 0.0],
        [400.0, 0.0],
        [400.0, 500.0],
        [0.0, 500.0]
    ], dtype=np.float32)

    # 1. Chạy Scratch
    t0 = time.perf_counter()
    H_scratch = solve_homography_svd(src, dst)
    t_scratch = (time.perf_counter() - t0) * 1000

    # 2. Chạy OpenCV
    t0 = time.perf_counter()
    H_cv = cv2.getPerspectiveTransform(src, dst)
    t_cv = (time.perf_counter() - t0) * 1000

    max_diff = np.max(np.abs(H_scratch - H_cv))
    print(f"  H (Scratch):\n{np.round(H_scratch, 6)}")
    print(f"  H (OpenCV):\n{np.round(H_cv, 6)}")
    print(f"  Sai số cực đại (Max Diff): {max_diff:.2e}")
    print(f"  Thời gian chạy: Scratch = {t_scratch:.3f} ms | OpenCV = {t_cv:.3f} ms")

    assert max_diff < 1e-4, f"Lỗi: Sai số Homography quá lớn ({max_diff})!"
    print("  --> [PASS] Ma trận Homography khớp hoàn toàn với OpenCV!")


def test_warp_perspective():
    print("=" * 60)
    print("3. KIỂM THỬ: warp_perspective_bilinear() vs cv2.warpPerspective()")

    # Tạo ảnh mẫu tổng hợp: lưới kẻ ca-rô + gradient + hình tròn
    h_in, w_in = 600, 500
    img = np.zeros((h_in, w_in), dtype=np.uint8)
    for r in range(0, h_in, 30):
        img[r : r + 15, :] = 200
    for c in range(0, w_in, 30):
        img[:, c : c + 15] = 150
    # Vẽ các hình tròn kiểm tra độ mượt của nội suy
    cv2.circle(img, (250, 300), 80, 255, -1)
    cv2.circle(img, (250, 300), 50, 50, -1)

    src = np.array([[60, 50], [420, 80], [450, 520], [40, 500]], dtype=np.float32)
    out_size = (400, 500)
    dst = np.array([[0, 0], [out_size[0], 0], [out_size[0], out_size[1]], [0, out_size[1]]], dtype=np.float32)

    H = solve_homography_svd(src, dst)

    # 1. Chạy Scratch
    t0 = time.perf_counter()
    warped_scratch = warp_perspective_bilinear(img, H, out_size)
    t_scratch = (time.perf_counter() - t0) * 1000

    # 2. Chạy OpenCV
    t0 = time.perf_counter()
    warped_cv = cv2.warpPerspective(img, H, out_size, flags=cv2.INTER_LINEAR)
    t_cv = (time.perf_counter() - t0) * 1000

    mse, psnr = compute_metrics(warped_scratch, warped_cv)
    max_diff = np.max(np.abs(warped_scratch.astype(int) - warped_cv.astype(int)))

    print(f"  Kích thước ảnh đích: {out_size[0]} x {out_size[1]}")
    print(f"  MSE  = {mse:.4f}")
    print(f"  PSNR = {psnr:.2f} dB (Tiêu chuẩn: > 48.0 dB)")
    print(f"  Sai khác mức xám cực đại: {max_diff} mức xám")
    print(f"  Thời gian: Scratch = {t_scratch:.2f} ms | OpenCV = {t_cv:.2f} ms")

    assert psnr > 48.0, f"Lỗi: PSNR quá thấp ({psnr:.2f} dB)!"
    print("  --> [PASS] Thuật toán Inverse Warping + Bilinear đạt chuẩn xuất sắc!")


def test_match_template():
    print("=" * 60)
    print("4. KIỂM THỬ: match_template_ncc() vs cv2.matchTemplate()")

    # Tạo ảnh ngẫu nhiên có độ mượt
    np.random.seed(42)
    img = np.random.randint(50, 200, (150, 150), dtype=np.uint8)

    # Tạo template một ô vuông đen đặc viền trắng
    template = np.zeros((20, 20), dtype=np.uint8)
    template[4:16, 4:16] = 255  # Khối vuông trắng giữa nền đen

    # Dán template vào vị trí (y=40, x=70) trên ảnh
    img[40:60, 70:90] = template

    # 1. Chạy Scratch
    t0 = time.perf_counter()
    ncc_scratch = match_template_ncc(img, template)
    t_scratch = (time.perf_counter() - t0) * 1000

    # 2. Chạy OpenCV
    t0 = time.perf_counter()
    ncc_cv = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    t_cv = (time.perf_counter() - t0) * 1000

    max_diff = np.max(np.abs(ncc_scratch - ncc_cv))
    peak_scratch = np.unravel_index(np.argmax(ncc_scratch), ncc_scratch.shape)
    peak_cv = np.unravel_index(np.argmax(ncc_cv), ncc_cv.shape)

    print(f"  Kích thước NCC map: {ncc_scratch.shape}")
    print(f"  Vị trí cực đại tìm được: Scratch = {peak_scratch} | OpenCV = {peak_cv}")
    print(f"  Giá trị NCC tại cực đại: Scratch = {ncc_scratch[peak_scratch]:.4f} | OpenCV = {ncc_cv[peak_cv]:.4f}")
    print(f"  Sai số cực đại giữa 2 ma trận NCC: {max_diff:.2e}")
    print(f"  Thời gian: Scratch = {t_scratch:.2f} ms | OpenCV = {t_cv:.2f} ms")

    assert max_diff < 1e-4, f"Lỗi: Sai số NCC quá lớn ({max_diff})!"
    assert peak_scratch == (40, 70), "Lỗi: Không tìm đúng vị trí template!"
    print("  --> [PASS] Thuật toán Template Matching NCC khớp 100% với OpenCV!")


def test_rotation_correction():
    print("=" * 60)
    print("5. KIỂM THỬ: detect_markers_and_correct_rotation()")

    # Tạo ảnh giả lập phiếu thi phẳng (kích thước 200 x 200)
    base_sheet = np.ones((200, 200), dtype=np.uint8) * 240

    # Tạo Marker đặc biệt (chữ L hoặc chữ thập)
    anchor = np.zeros((16, 16), dtype=np.uint8)
    anchor[4:12, 4:12] = 255
    anchor[6:10, 6:10] = 50

    # Đặt Marker đặc biệt tại góc Top-Right (chuẩn 0 độ)
    # y: 10..26, x: 174..190
    base_sheet[10:26, 174:190] = anchor

    # Kiểm thử cho cả 4 góc xoay theo chiều kim đồng hồ: 0, 90, 180, 270
    angles_to_test = [0, 90, 180, 270]
    for expected_angle in angles_to_test:
        if expected_angle == 0:
            test_img = base_sheet.copy()
        elif expected_angle == 90:
            # Xoay thuận chiều kim đồng hồ 90 độ (np.rot90 với k=3 hoặc k=-1)
            test_img = np.rot90(base_sheet, k=3)
        elif expected_angle == 180:
            test_img = np.rot90(base_sheet, k=2)
        elif expected_angle == 270:
            # Xoay thuận chiều kim đồng hồ 270 độ (tức ngược 90 độ, k=1)
            test_img = np.rot90(base_sheet, k=1)

        corrected, detected_angle, scores = detect_markers_and_correct_rotation(test_img, anchor)
        print(f"  Ảnh xoay thuận {expected_angle:3d}° --> Nhận diện được: {detected_angle:3d}°")
        assert detected_angle == expected_angle, f"Lỗi: Nhận diện sai góc xoay {expected_angle}!"
        # Sau khi sửa phải về lại đúng vị trí TR
        _, final_angle, _ = detect_markers_and_correct_rotation(corrected, anchor)
        assert final_angle == 0, f"Lỗi: Sau khi nắn chưa về 0 độ (vẫn còn {final_angle} độ)!"

    print("  --> [PASS] Nhận diện và tự động xoay chuẩn hướng 0° thành công cho cả 4 góc!")


def test_pipeline_wrapper():
    print("=" * 60)
    print("6. KIỂM THỬ: process_perspective_pipeline() (Hàm tích hợp toàn bộ)")

    # Giả lập ảnh gốc và 4 đỉnh từ TV1
    img = np.ones((500, 400), dtype=np.uint8) * 180
    raw_corners = np.array([[350, 420], [50, 60], [40, 410], [360, 50]], dtype=np.float32)

    # Chạy pipeline tổng của TV2
    warped_sheet, info = process_perspective_pipeline(img, raw_corners, target_size=(300, 400))

    print(f"  Shape ảnh sau pipeline: {warped_sheet.shape}")
    print(f"  Góc TL sau sắp xếp: {info['ordered_corners'][0]}")
    print(f"  Shape ma trận H: {info['homography_matrix'].shape}")

    assert warped_sheet.shape == (400, 300), f"Lỗi: Kích thước đầu ra sai ({warped_sheet.shape})!"
    assert info["homography_matrix"].shape == (3, 3), "Lỗi: Ma trận H sai shape!"
    print("  --> [PASS] Hàm tích hợp process_perspective_pipeline() hoạt động hoàn hảo!")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  BẮT ĐẦU CHẠY KIỂM THỬ TOÀN DIỆN MODULE THÀNH VIÊN 2 (TV2)")
    print("#" * 60 + "\n")

    test_order_corners()
    test_solve_homography()
    test_warp_perspective()
    test_match_template()
    test_rotation_correction()
    test_pipeline_wrapper()

    print("\n" + "=" * 60)
    print("🎉 TẤT CẢ 5 BÀI KIỂM THỬ ĐỀU ĐẠT CHUẨN XUẤT SẮC!")
    print("=" * 60 + "\n")
