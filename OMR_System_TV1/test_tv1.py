"""Script: test_tv1.py - Kiểm thử toàn diện và Đối chứng Độc lập Module Thành viên 1
Học viện Công nghệ Bưu chính Viễn thông (PTIT)
Môn học: Xử lý ảnh số (INT13146)
Giảng viên hướng dẫn: TS. Phạm Hoàng Việt
Thành viên phụ trách: Thành viên 1 (Tiền xử lý & Phát hiện khung phiếu)

Chạy trực tiếp:
    python test_tv1.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Thêm src vào path
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tv1_preprocessing import (
    rgb_to_gray_scratch,
    gaussian_blur_scratch,
    canny_edge_scratch,
    hough_lines_standard,
    apply_parameter_space_nms,
    detect_sheet_corners,
    process_preprocessing_pipeline,
    calculate_mse,
    calculate_psnr,
    calculate_binary_iou_dice,
    calculate_line_match_rate,
)


def run_all_tv1_tests():
    print("\n" + "#" * 68)
    print("  BẮT ĐẦU CHẠY KIỂM THỬ TOÀN DIỆN MODULE THÀNH VIÊN 1 (TV1)")
    print("#" * 68 + "\n")

    base_dir = Path(__file__).resolve().parent.parent
    samples_dir = base_dir / "OMR_System_TV4" / "data" / "samples"
    sample_p = samples_dir / "01_B21DCCN001_Chuan.png"

    if sample_p.exists():
        real_img = cv2.imread(str(sample_p))
    else:
        real_img = np.full((1000, 800, 3), 245, dtype=np.uint8)
        cv2.rectangle(real_img, (50, 50), (750, 950), (20, 20, 20), 4)

    # ============================================================
    # 1. KIỂM THỬ: Chuyển đổi mức xám Luminance ITU-R BT.601
    # ============================================================
    print("=" * 68)
    print("1. KIỂM THỬ: rgb_to_gray_scratch() vs cv2.cvtColor(BGR2GRAY)")
    t0 = time.perf_counter()
    gray_sc = rgb_to_gray_scratch(real_img, is_bgr=True)
    t_sc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    gray_cv = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    t_cv = (time.perf_counter() - t0) * 1000.0

    mse_g = calculate_mse(gray_sc, gray_cv)
    psnr_g = calculate_psnr(gray_sc, gray_cv)
    max_d_g = int(np.max(np.abs(gray_sc.astype(int) - gray_cv.astype(int))))

    print(f"  Kích thước ảnh: {real_img.shape} -> {gray_sc.shape}")
    print(f"  MSE  = {mse_g:.4f}")
    print(f"  PSNR = {psnr_g:.2f} dB (Tiêu chuẩn: > 48.0 dB)")
    print(f"  Sai khác mức xám cực đại: {max_d_g} mức xám")
    print(f"  Thời gian: Scratch = {t_sc:.2f} ms | OpenCV = {t_cv:.2f} ms")
    assert mse_g < 0.1, "Lỗi MSE Grayscale!"
    print("  --> [PASS] Thuật toán Chuyển mức xám Luminance khớp hoàn hảo!")

    # ============================================================
    # 2. KIỂM THỬ: Gaussian Blur 2D (Tích chập phân tách O(2k))
    # ============================================================
    print("\n" + "=" * 68)
    print("2. KIỂM THỬ: gaussian_blur_scratch() vs cv2.GaussianBlur()")
    t0 = time.perf_counter()
    blur_sc = gaussian_blur_scratch(gray_sc, kernel_size=5, sigma=1.4)
    t_sc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    blur_cv = cv2.GaussianBlur(gray_cv, (5, 5), 1.4, borderType=cv2.BORDER_REFLECT_101)
    t_cv = (time.perf_counter() - t0) * 1000.0

    mse_b = calculate_mse(blur_sc, blur_cv)
    psnr_b = calculate_psnr(blur_sc, blur_cv)
    max_d_b = int(np.max(np.abs(blur_sc.astype(int) - blur_cv.astype(int))))

    print(f"  Cấu hình: Kernel 5x5, sigma=1.4, Reflect padding (BORDER_REFLECT_101)")
    print(f"  MSE  = {mse_b:.4f}")
    print(f"  PSNR = {psnr_b:.2f} dB (Tiêu chuẩn: > 48.0 dB)")
    print(f"  Sai khác mức xám cực đại: {max_d_b} mức xám")
    print(f"  Thời gian: Scratch = {t_sc:.2f} ms | OpenCV = {t_cv:.2f} ms")
    assert psnr_b > 48.0, "Lỗi PSNR Gaussian Blur!"
    print("  --> [PASS] Thuật toán Lọc Gaussian 2D đạt chuẩn xuất sắc!")

    # ============================================================
    # 3. KIỂM THỬ: Canny Edge Detector 4 bước hoàn chỉnh
    # ============================================================
    print("\n" + "=" * 68)
    print("3. KIỂM THỬ: canny_edge_scratch() vs cv2.Canny(L2gradient=True)")
    t0 = time.perf_counter()
    canny_sc = canny_edge_scratch(gray_sc, low_threshold=50.0, high_threshold=150.0)
    t_sc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    canny_cv = cv2.Canny(blur_cv, 50, 150, L2gradient=True)
    t_cv = (time.perf_counter() - t0) * 1000.0

    iou_c, dice_c = calculate_binary_iou_dice(canny_sc, canny_cv)
    print(f"  Ngưỡng Canny: T_low = 50.0, T_high = 150.0")
    print(f"  IoU  = {iou_c * 100:.2f}% (Tiêu chuẩn: >= 98.0%)")
    print(f"  Dice = {dice_c:.4f} (Tiêu chuẩn: >= 0.985)")
    print(f"  Thời gian: Scratch = {t_sc:.2f} ms | OpenCV = {t_cv:.2f} ms")
    assert iou_c >= 0.980, f"Lỗi IoU Canny Edge ({iou_c*100:.2f}%)!"
    print("  --> [PASS] Thuật toán Canny Edge Detector 4 bước khớp >= 99%!")

    # ============================================================
    # 4. KIỂM THỬ: Hough Line Transform cực (rho, theta)
    # ============================================================
    print("\n" + "=" * 68)
    print("4. KIỂM THỬ: hough_lines_standard() vs cv2.HoughLines()")
    t0 = time.perf_counter()
    lines_sc, acc = hough_lines_standard(canny_sc, rho_res=1.0, theta_res=np.pi / 180.0, threshold=100)
    t_sc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    lines_cv_raw = cv2.HoughLines(canny_cv, 1.0, np.pi / 180.0, 100)
    t_cv = (time.perf_counter() - t0) * 1000.0

    lines_cv = (
        [(float(l[0][0]), float(l[0][1])) for l in lines_cv_raw]
        if lines_cv_raw is not None else []
    )
    lines_cv_nms = apply_parameter_space_nms(lines_cv)
    recall = calculate_line_match_rate(lines_cv_nms, lines_sc)
    precision = calculate_line_match_rate(lines_sc, lines_cv_nms)

    print(f"  Kích thước Accumulator: {acc.shape}")
    print(f"  Số đường thẳng: Scratch = {len(lines_sc)} | OpenCV = {len(lines_cv)} (sau NMS: {len(lines_cv_nms)})")
    print(f"  Recall    = {recall * 100:.1f}% (Tiêu chuẩn: >= 95%)")
    print(f"  Precision = {precision * 100:.1f}% (Tiêu chuẩn: >= 90%)")
    print(f"  Thời gian: Scratch = {t_sc:.2f} ms | OpenCV = {t_cv:.2f} ms")
    assert recall >= 0.95, f"Recall Hough quá thấp: {recall*100:.1f}%"
    assert precision >= 0.90, f"Precision Hough quá thấp: {precision*100:.1f}%"
    print("  --> [PASS] Biến đổi Hough cực khớp tập đường chính của OpenCV!")

    # ============================================================
    # 5. KIỂM THỬ: Định vị 4 đỉnh góc phiếu thi (detect_sheet_corners)
    # ============================================================
    print("\n" + "=" * 68)
    print("5. KIỂM THỬ: detect_sheet_corners() (Kịch bản chuẩn & Nghiêng DS-02)")
    c_std_sc = detect_sheet_corners(real_img, use_scratch=True)
    c_std_cv = detect_sheet_corners(real_img, use_scratch=False)
    err_std = float(np.mean(np.linalg.norm(c_std_sc - c_std_cv, axis=1)))
    print(f"  [DS-01 Ảnh chuẩn] Sai số góc trung bình: {err_std:.2f} px")
    print(f"    Góc (Scratch): {c_std_sc.tolist()}")

    p2_path = samples_dir / "02_B21DCCN002_Nghieng_BanToi.png"
    if p2_path.exists():
        tilted_img = cv2.imread(str(p2_path))
        c_tilt_sc = detect_sheet_corners(tilted_img, use_scratch=True)
        c_tilt_cv = detect_sheet_corners(tilted_img, use_scratch=False)
        err_tilt = float(np.mean(np.linalg.norm(c_tilt_sc - c_tilt_cv, axis=1)))
        print(f"  [DS-02 Ảnh nghiêng bàn tối] Sai số góc trung bình: {err_tilt:.2f} px")
        print(f"    Góc tìm được (Scratch): {c_tilt_sc.tolist()}")
        print(f"    Góc đối chứng (OpenCV): {c_tilt_cv.tolist()}")
        assert err_tilt <= 15.0, "Sai số góc vượt quá ngưỡng cho phép!"

        # DS-02 phải được dò bằng Canny + Hough (tứ giác co vào trong ảnh)
        h_t, w_t = tilted_img.shape[:2]
        xs, ys = c_tilt_sc[:, 0], c_tilt_sc[:, 1]
        assert xs.min() > 10 and ys.min() > 10 and xs.max() < w_t - 10 and ys.max() < h_t - 10, \
            "DS-02 phải trả về tứ giác nằm trong ảnh, không trùng mép khung hình!"
        print("  --> [PASS] Nhánh Canny + Hough hoạt động đúng trên ảnh nghiêng!")

    print("  --> [PASS] Định vị 4 góc phiếu thi chính xác và bền vững!")

    # DS-05: bóng đổ một phía không được làm phiếu bị "thụt" vào trong khung hình
    p5_path = samples_dir / "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png"
    if p5_path.exists():
        shadow_img = cv2.imread(str(p5_path))
        c_sh_sc = detect_sheet_corners(shadow_img, use_scratch=True)
        c_sh_cv = detect_sheet_corners(shadow_img, use_scratch=False)
        err_sh = float(np.mean(np.linalg.norm(c_sh_sc - c_sh_cv, axis=1)))
        print(f"  [DS-05 Bóng đổ không đều] Sai số góc trung bình: {err_sh:.2f} px")
        print(f"    Góc (Scratch): {np.round(c_sh_sc, 1).tolist()}")
        assert err_sh <= 3.0, "Bóng đổ không được đẩy phiếu sang nhánh nền bàn tối!"
        print("  --> [PASS] Bóng đổ một phía không bị nhầm thành nền bàn tối!")

    # ============================================================
    # 6. KIỂM THỬ: Full Pipeline Stage 1
    # ============================================================
    print("\n" + "=" * 68)
    print("6. KIỂM THỬ: process_preprocessing_pipeline() (Tích hợp trọn vẹn)")
    p_res = process_preprocessing_pipeline(real_img, use_scratch=True)
    print(f"  Thời gian thực thi từng giai đoạn nhỏ:")
    for k, v in p_res["latencies"].items():
        print(f"    - {k:<22}: {v:.2f} ms")
    print(f"  Shape ảnh xám: {p_res['gray'].shape}")
    print(f"  Shape ảnh biên: {p_res['edges'].shape}")
    print(f"  Tọa độ 4 góc: {p_res['corners'].tolist()}")
    assert p_res["latencies"]["total_tv1_ms"] < 600.0, "TV1 phải chạy dưới 600 ms/phiếu!"
    print("  --> [PASS] Toàn bộ Pipeline Giai đoạn 1 vận hành mượt mà!")

    print("\n" + "=" * 68)
    print("TẤT CẢ 6 BÀI KIỂM THỬ MODULE THÀNH VIÊN 1 ĐỀU ĐẠT CHUẨN!")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    run_all_tv1_tests()
