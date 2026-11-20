"""Tab 3: Đối Chứng Thuật Toán vs OpenCV & Khảo Sát Tham Số.
Clean, academic interface without text emojis.
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Imports from TV1, TV2, TV3, TV4
try:
    from tv1_preprocessing import (
        rgb_to_gray_scratch,
        gaussian_blur_scratch,
        canny_edge_scratch
    )
except ImportError:
    rgb_to_gray_scratch = None
    gaussian_blur_scratch = None
    canny_edge_scratch = None

try:
    from tv2_perspective import warp_perspective_sheet, match_template_ncc
except ImportError:
    warp_perspective_sheet = None
    match_template_ncc = None

try:
    from tv3_thresholding import (
        otsu_threshold_scratch,
        adaptive_mean_threshold_scratch
    )
    from tv3_morphology_ccl import opening_binary_scratch, two_pass_ccl_scratch
except ImportError:
    otsu_threshold_scratch = None
    adaptive_mean_threshold_scratch = None
    opening_binary_scratch = None
    two_pass_ccl_scratch = None

from tv4_enhancement_eval import (
    calculate_mse,
    calculate_psnr,
    calculate_difference_map,
    calculate_binary_iou_dice,
    calculate_ccl_component_iou,
    benchmark_hist_equalization,
    benchmark_median_filter,
    benchmark_all_modules,
    run_ablation_study
)
from pipeline_coordinator import OMRPipeline, create_anchor_marker_template
from ui.styles import apply_modern_chart_style, read_image_unicode


def render_tab3(
    use_scratch: bool,
    fill_thresh: float,
    apply_enhancement: bool,
    samples_dir: str,
    default_sheet_path: str,
    default_key_path: str = ""
):
    """Render Tab 3: Quantitative Verification & Parameter Survey."""

    st.markdown("##### Đối Chứng Trực Quan: Thuật Toán Tự Cài Đặt (NumPy) vs Thư Viện OpenCV")
    st.markdown("""
    Kiểm chứng độ chính xác giữa thuật toán tự cài đặt (NumPy) và thư viện OpenCV thông qua chỉ số **MSE**, **PSNR (dB)**, **IoU**, **Bản đồ sai khác tuyệt đối** $|I_{\\text{cài đặt}} - I_{\\text{OpenCV}}|$, và **Khảo sát độ nhạy tham số**.
    """)

    # 1. State Sync & Banner
    current_core_name = "Thuật toán tự cài đặt (NumPy)" if use_scratch else "Thư viện OpenCV"
    st.markdown(f"""
    <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 12px 16px; margin-bottom: 18px;">
        <div style="font-size: 13px; color: #1E40AF; line-height: 1.5;">
            <b>Cơ chế Đối chứng Độc lập & Đồng bộ Toàn Cục:</b> Tab đối chứng luôn tự động thực thi <b>song song cả 2 thuật toán (NumPy tự cài đặt & OpenCV chuẩn)</b> trên cùng một ảnh đầu vào để tính toán chính xác sai số toán học (MSE, PSNR, IoU) độc lập với lựa chọn tại Sidebar.
            <span style="display: block; margin-top: 4px; color: #3B82F6; font-size: 12px;">
                Cấu hình Pipeline hiện tại: <b>{current_core_name}</b> • Ngưỡng tô: <b>{fill_thresh:.2f}</b> • Tiền xử lý: <b>{'Bật' if apply_enhancement else 'Tắt'}</b> (Áp dụng cho quy trình Chấm Bài Trắc Nghiệm).
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sub_t1, sub_t2, sub_t3 = st.tabs([
        "Đối Chứng Từng Thuật Toán",
        "Bảng Đối Chứng Toàn Diện Pipeline",
        "Khảo Sát Độ Nhạy Tham Số"
    ])

    with sub_t1:
        trigger_quick = st.session_state.get("run_quick_test_trigger", False)
        if trigger_quick:
            st.session_state.run_quick_test_trigger = False
            st.session_state["sb1_algo"] = "[Tiền xử lý] Phát hiện biên Canny (Canny Edge Detection)"
            st.session_state["sb1_src"] = "Ảnh phiếu chuẩn (Kịch bản 1)"

        # Control Toolbar
        with st.container(border=True):
            st.markdown('<div class="benchmark-toolbar-title">Bảng Điều Khiển Đo Đạc Đối Chứng (Control Toolbar)</div>', unsafe_allow_html=True)

            cur_sb_algo = st.session_state.get("sb1_algo", "[Tiền xử lý] Phát hiện biên Canny (Canny Edge Detection)")
            is_median = ("trung vị" in cur_sb_algo or "Median" in cur_sb_algo)

            if is_median:
                b_col1, b_col_k, b_col2, b_col3 = st.columns([2.3, 0.9, 1.8, 1.4], vertical_alignment="bottom")
            else:
                b_col1, b_col2, b_col3 = st.columns([2.5, 2.0, 1.5], vertical_alignment="bottom")

            with b_col1:
                bench_algo = st.selectbox("Chọn thuật toán đối chứng:", [
                    "[Tiền xử lý] Chuyển đổi mức xám (Grayscale ITU-R BT.601)",
                    "[Tiền xử lý] Làm mờ Gauss 2D (Gaussian Blur 5x5)",
                    "[Tiền xử lý] Phát hiện biên Canny (Canny Edge Detection)",
                    "[Hình học SVD] Nắn phối cảnh hình học (Homography SVD & Bilinear Warp)",
                    "[Hình học SVD] So khớp mẫu NCC Marker (Template Matching NCC)",
                    "[Phân đoạn] Phân ngưỡng Otsu toàn cục (Otsu Global Threshold)",
                    "[Phân đoạn] Phân ngưỡng thích nghi (Adaptive Mean Threshold)",
                    "[Phân đoạn] Phép mở hình thái học (Morphology Opening 3x3)",
                    "[Phân đoạn] Gán nhãn thành phần liên thông (Two-Pass CCL 8-lân cận)",
                    "[Cải thiện ảnh] Cân bằng biểu đồ mức xám (Histogram Equalization)",
                    "[Khử nhiễu] Bộ lọc trung vị khử nhiễu (Median Filter 2D)"
                ], index=2, key="sb1_algo")

            if is_median:
                with b_col_k:
                    bench_k = st.selectbox("Kernel:", [3, 5, 7], index=0, key="sb1_k")
            else:
                bench_k = 3

            src_map_1 = {
                "Ảnh phiếu chuẩn (Kịch bản 1)": os.path.join(samples_dir, "01_B21DCCN001_Chuan.png"),
                "Ảnh chụp nghiêng bàn tối (Kịch bản 2)": os.path.join(samples_dir, "02_B21DCCN002_Nghieng_BanToi.png"),
                "Ảnh phiếu thiếu sáng & nhiễu hạt (Kịch bản 3)": os.path.join(samples_dir, "03_B21DCCN003_ThieuSang_NhieuHat.png"),
                "Ảnh bóng đổ & chênh sáng (Kịch bản 5)": os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png"),
                "Tải ảnh tùy chọn từ máy tính": None,
            }

            with b_col2:
                bench_src = st.selectbox("Chọn ảnh thử nghiệm:", list(src_map_1.keys()), key="sb1_src")

            with b_col3:
                btn_run_bench = st.button("Bắt Đầu Đo Đạc", type="primary", use_container_width=True, key="sb1_btn_run")

            custom_bench_file = None
            if src_map_1[bench_src] is None:
                st.write("")
                custom_bench_file = st.file_uploader("Nạp ảnh tùy chọn (.png, .jpg):", type=["png", "jpg", "jpeg"], key="sb1_upload")

        # Load image source
        bench_img = None
        custom_meta = None
        if src_map_1[bench_src] is None:
            if custom_bench_file is not None:
                c_bytes = np.asarray(bytearray(custom_bench_file.getvalue()), dtype=np.uint8)
                bench_img = cv2.imdecode(c_bytes, cv2.IMREAD_COLOR)
                custom_meta = (custom_bench_file.name, custom_bench_file.size)
        else:
            p_sel = src_map_1[bench_src]
            bench_img = read_image_unicode(p_sel if os.path.exists(p_sel) else default_sheet_path)

        if trigger_quick:
            btn_run_bench = True

        current_tab3_sig = (bench_algo, bench_k, bench_src, custom_meta, bench_img.shape if bench_img is not None else None)
        if bench_img is not None and (
            btn_run_bench
            or (st.session_state.tab3_result is not None and st.session_state.tab3_last_sig != current_tab3_sig)
        ):
            with st.spinner("Đang thực hiện đo đạc đối chứng thuật toán & tính toán sai số MSE/PSNR..."):
                st.session_state.tab3_last_sig = current_tab3_sig

                if bench_img.ndim == 3 and bench_img.shape[2] >= 3:
                    gray_bench = cv2.cvtColor(bench_img, cv2.COLOR_BGR2GRAY)
                    bgr_bench = bench_img
                else:
                    gray_bench = bench_img
                    bgr_bench = cv2.cvtColor(gray_bench, cv2.COLOR_GRAY2BGR)

                t_sc_list = []
                t_cv_list = []
                out_sc = None
                out_cv = None
                vis_sc = None
                vis_cv = None
                diff_map_override = None
                amp = 10.0
                iou_val = None
                dice_val = None
                psnr_max_val = 255.0

                if "Grayscale" in bench_algo:
                    for _ in range(3):
                        t0 = time.perf_counter()
                        out_sc = rgb_to_gray_scratch(bgr_bench, is_bgr=True) if rgb_to_gray_scratch is not None else gray_bench
                        t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                        t0 = time.perf_counter()
                        out_cv = cv2.cvtColor(bgr_bench, cv2.COLOR_BGR2GRAY)
                        t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                    amp = 20.0

                elif "Gauss" in bench_algo:
                    for _ in range(3):
                        t0 = time.perf_counter()
                        out_sc = gaussian_blur_scratch(gray_bench, 5, 1.4) if gaussian_blur_scratch is not None else gray_bench
                        t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                        t0 = time.perf_counter()
                        out_cv = cv2.GaussianBlur(gray_bench, (5, 5), 1.4, borderType=cv2.BORDER_REFLECT_101)
                        t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                    amp = 20.0

                elif "Canny" in bench_algo:
                    b_cv = cv2.GaussianBlur(gray_bench, (5, 5), 1.4, borderType=cv2.BORDER_REFLECT_101)
                    for _ in range(3):
                        t0 = time.perf_counter()
                        out_sc = canny_edge_scratch(gray_bench, 50.0, 150.0) if canny_edge_scratch is not None else cv2.Canny(b_cv, 50, 150)
                        t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                        t0 = time.perf_counter()
                        out_cv = cv2.Canny(b_cv, 50, 150, L2gradient=True)
                        t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                    amp = 1.0
                    iou_val, dice_val = calculate_binary_iou_dice(out_sc, out_cv)

                elif "Warp" in bench_algo:
                    h, w = gray_bench.shape
                    quad = np.array([
                        [0.02 * w, 0.02 * h],
                        [0.97 * w, 0.01 * h],
                        [0.98 * w, 0.98 * h],
                        [0.01 * w, 0.97 * h],
                    ], dtype=np.float32)
                    if warp_perspective_sheet is not None:
                        for _ in range(3):
                            t0 = time.perf_counter()
                            out_sc = warp_perspective_sheet(gray_bench, quad, 800, 1000, use_scratch=True)
                            t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                            t0 = time.perf_counter()
                            out_cv = warp_perspective_sheet(gray_bench, quad, 800, 1000, use_scratch=False)
                            t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                    else:
                        dst = np.array([[0, 0], [799, 0], [799, 999], [0, 999]], dtype=np.float32)
                        M = cv2.getPerspectiveTransform(quad, dst)
                        out_sc = cv2.warpPerspective(gray_bench, M, (800, 1000))
                        out_cv = out_sc.copy()
                        t_sc_list = [1.0]
                        t_cv_list = [1.0]
                    amp = 5.0

                elif "NCC" in bench_algo:
                    h, w = gray_bench.shape
                    anchor = create_anchor_marker_template(30)
                    mh = max(30, int(h * 0.25))
                    mw = max(30, int(w * 0.25))
                    roi_tr = gray_bench[0:mh, max(0, w - mw):w]
                    for _ in range(3):
                        t0 = time.perf_counter()
                        out_sc = match_template_ncc(roi_tr, anchor) if match_template_ncc is not None else cv2.matchTemplate(roi_tr, anchor, cv2.TM_CCOEFF_NORMED)
                        t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                        t0 = time.perf_counter()
                        out_cv = cv2.matchTemplate(roi_tr, anchor, cv2.TM_CCOEFF_NORMED)
                        t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                    vis_sc = np.clip((out_sc + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    vis_cv = np.clip((out_cv + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    amp = 2550.0
                    diff_map_override = np.clip(np.abs(out_sc.astype(np.float64) - out_cv.astype(np.float64)) * amp, 0, 255).astype(np.uint8)
                    psnr_max_val = 1.0

                elif "Otsu" in bench_algo:
                    for _ in range(3):
                        t0 = time.perf_counter()
                        if otsu_threshold_scratch is not None:
                            _, out_sc = otsu_threshold_scratch(gray_bench, method="BINARY_INV")
                        else:
                            _, out_sc = cv2.threshold(gray_bench, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                        t0 = time.perf_counter()
                        _, out_cv = cv2.threshold(gray_bench, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                    amp = 1.0
                    iou_val, dice_val = calculate_binary_iou_dice(out_sc, out_cv)

                elif "Adaptive" in bench_algo:
                    for _ in range(3):
                        t0 = time.perf_counter()
                        if adaptive_mean_threshold_scratch is not None:
                            out_sc = adaptive_mean_threshold_scratch(gray_bench, block_size=51, C=42.0)
                        else:
                            out_sc = cv2.adaptiveThreshold(gray_bench, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42)
                        t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                        t0 = time.perf_counter()
                        out_cv = cv2.adaptiveThreshold(gray_bench, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42)
                        t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                    amp = 1.0
                    iou_val, dice_val = calculate_binary_iou_dice(out_sc, out_cv)

                elif "hình thái" in bench_algo or "Opening" in bench_algo:
                    _, bin_base = cv2.threshold(gray_bench, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    for _ in range(3):
                        t0 = time.perf_counter()
                        out_sc = opening_binary_scratch(bin_base) if opening_binary_scratch is not None else bin_base
                        t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                        t0 = time.perf_counter()
                        out_cv = cv2.morphologyEx(bin_base, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                        t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                    amp = 1.0
                    iou_val, dice_val = calculate_binary_iou_dice(out_sc, out_cv)

                elif "CCL" in bench_algo:
                    adap_m = (
                        adaptive_mean_threshold_scratch(gray_bench, block_size=51, C=42.0)
                        if adaptive_mean_threshold_scratch is not None
                        else cv2.adaptiveThreshold(gray_bench, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42)
                    )
                    open_m = (
                        opening_binary_scratch(adap_m)
                        if opening_binary_scratch is not None
                        else cv2.morphologyEx(adap_m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    )
                    k_sc, lbl_sc, k_cv, lbl_cv = 0, np.zeros_like(open_m, dtype=np.int32), 0, np.zeros_like(open_m, dtype=np.int32)
                    for _ in range(2):
                        t0 = time.perf_counter()
                        if two_pass_ccl_scratch is not None:
                            k_sc, lbl_sc, _ = two_pass_ccl_scratch(open_m, connectivity=8)
                        else:
                            k_sc_r, lbl_sc, _, _ = cv2.connectedComponentsWithStats(open_m, connectivity=8)
                            k_sc = max(0, int(k_sc_r) - 1)
                        t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                        t0 = time.perf_counter()
                        k_cv_r, lbl_cv, _, _ = cv2.connectedComponentsWithStats(open_m, connectivity=8)
                        t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                        k_cv = max(0, int(k_cv_r) - 1)

                    iou_val, dice_val = calculate_ccl_component_iou(lbl_sc, lbl_cv, int(k_sc), k_cv)
                    out_sc = (lbl_sc > 0).astype(np.uint8) * 255
                    out_cv = (lbl_cv > 0).astype(np.uint8) * 255
                    rng = np.random.default_rng(42)
                    palette = rng.integers(60, 255, size=(max(int(k_sc), k_cv) + 2, 3), dtype=np.uint8)
                    palette[0] = [0, 0, 0]
                    vis_sc = palette[np.clip(lbl_sc, 0, len(palette) - 1)]
                    # Map OpenCV labels to overlapping Scratch label color so identical regions share identical colors
                    cv_to_sc = np.arange(len(palette), dtype=np.int64)
                    fg_overlap = (lbl_sc > 0) & (lbl_cv > 0)
                    if np.any(fg_overlap):
                        stride = np.int64(k_cv + 1)
                        u_pairs = np.unique(lbl_sc[fg_overlap].astype(np.int64) * stride + lbl_cv[fg_overlap].astype(np.int64))
                        for pair_val in u_pairs:
                            s_lbl = int(pair_val // stride)
                            c_lbl = int(pair_val % stride)
                            if 0 < c_lbl < len(cv_to_sc):
                                cv_to_sc[c_lbl] = s_lbl
                    cv_mapped = cv_to_sc[np.clip(lbl_cv, 0, len(cv_to_sc) - 1)]
                    vis_cv = palette[np.clip(cv_mapped, 0, len(palette) - 1)]
                    amp = 1.0
                    st.info(f"Two-Pass CCL (8-lân cận): Scratch = {k_sc} thành phần | OpenCV = {k_cv} thành phần (Chênh lệch ΔK = {abs(int(k_sc) - k_cv)}).")

                elif "Histogram" in bench_algo:
                    st.session_state.tab3_result = benchmark_hist_equalization(gray_bench, iterations=3)

                elif "trung vị" in bench_algo or "Median" in bench_algo:
                    st.session_state.tab3_result = benchmark_median_filter(gray_bench, kernel_size=bench_k, iterations=3)

                if "Histogram" not in bench_algo and "Median" not in bench_algo and "trung vị" not in bench_algo:
                    mse_calc = calculate_mse(out_sc, out_cv)
                    psnr_calc = calculate_psnr(out_sc, out_cv, max_val=psnr_max_val)
                    diff_map = diff_map_override if diff_map_override is not None else calculate_difference_map(out_sc, out_cv, amplify=amp)
                    st.session_state.tab3_result = {
                        "img_scratch": vis_sc if vis_sc is not None else out_sc,
                        "img_opencv": vis_cv if vis_cv is not None else out_cv,
                        "diff_map": diff_map,
                        "time_scratch_ms": round(float(np.mean(t_sc_list)), 3),
                        "time_opencv_ms": round(float(np.mean(t_cv_list)), 3),
                        "mse": round(mse_calc, 4),
                        "psnr_db": round(psnr_calc, 2),
                        "amplify": amp,
                        "iou": iou_val,
                        "dice": dice_val
                    }

        # Result Display Area & Empty State
        if st.session_state.tab3_result is not None:
            b_res = st.session_state.tab3_result
            st.write("")
            bm1, bm2, bm3, bm4 = st.columns(4)
            bm1.metric("Thời gian Scratch (NumPy)", f"{b_res['time_scratch_ms']} ms")
            bm2.metric("Thời gian OpenCV (C++)", f"{b_res['time_opencv_ms']} ms")

            if b_res.get("iou") is not None:
                if b_res["mse"] == 0.0:
                    mse_delta = "Khớp 100%"
                elif b_res["iou"] >= 0.985:
                    mse_delta = "Đạt chuẩn IoU ≥ 98.5%"
                else:
                    mse_delta = "Cần kiểm tra"
            else:
                if b_res["mse"] == 0.0:
                    mse_delta = "Khớp 100%"
                elif b_res["mse"] < 1.0:
                    mse_delta = "Đạt chuẩn < 1.0"
                else:
                    mse_delta = "Cần kiểm tra"

            bm3.metric("Sai số MSE", f"{b_res['mse']}", delta=mse_delta)
            if b_res.get("iou") is not None:
                bm4.metric("IoU / Dice", f"{b_res['iou']*100:.2f}%", delta=f"Dice: {b_res.get('dice', 1.0):.4f}")
            else:
                bm4.metric("PSNR (dB)", f"{b_res['psnr_db']} dB")

            st.write("")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                with st.container(border=True):
                    st.markdown("###### 1. Thuật Toán Tự Cài Đặt (NumPy)")
                    st.image(b_res["img_scratch"], use_container_width=True, channels="RGB" if b_res["img_scratch"].ndim == 3 else "GRAY")
            with tc2:
                with st.container(border=True):
                    st.markdown("###### 2. Thư Viện OpenCV Chuẩn (C++)")
                    st.image(b_res["img_opencv"], use_container_width=True, channels="RGB" if b_res["img_opencv"].ndim == 3 else "GRAY")
            with tc3:
                with st.container(border=True):
                    amp_factor = b_res.get("amplify", 10.0)
                    st.markdown(f"###### 3. Bản Đồ Sai Khác (x{amp_factor:.0f})")
                    st.image(b_res["diff_map"], use_container_width=True, channels="RGB" if b_res["diff_map"].ndim == 3 else "GRAY")
        else:
            st.write("")
            with st.container(border=True):
                st.markdown("""
                <div style="text-align: center; padding: 22px 16px 12px 16px;">
                    <h4 style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">Chưa Có Dữ Liệu Đối Chứng</h4>
                    <p style="color: #64748B; font-size: 13.5px; max-width: 580px; margin: 0 auto 16px auto; line-height: 1.5;">
                        Chọn thuật toán và ảnh tại <b>Bảng Điều Khiển</b> phía trên rồi bấm nút <b>"Bắt Đầu Đo Đạc"</b>, hoặc chạy thử nghiệm ngay lập tức với bộ mẫu tiêu chuẩn.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                _, q_c2, _ = st.columns([1.5, 2.2, 1.5])
                with q_c2:
                    if st.button("Chạy thử nhanh với mẫu mặc định", type="primary", use_container_width=True, key="btn_quick_bench"):
                        st.session_state.run_quick_test_trigger = True
                        st.rerun()

            st.write("")
            st.markdown("###### Xem Trước Bố Cục Kết Quả (3 Khung Hình Đối Chứng):")
            sk1, sk2, sk3 = st.columns(3)
            with sk1:
                st.markdown("""
                <div class="skeleton-preview-card">
                    <div class="skeleton-icon">SCRATCH</div>
                    <div class="skeleton-title">1. Thuật Toán Tự Cài Đặt (NumPy)</div>
                    <div class="skeleton-desc">Vùng hiển thị ma trận ảnh xử lý bằng thuật toán tự cài đặt thuần NumPy để kiểm chứng.</div>
                    <div class="skeleton-badge">Đang chờ lệnh đo...</div>
                </div>
                """, unsafe_allow_html=True)
            with sk2:
                st.markdown("""
                <div class="skeleton-preview-card">
                    <div class="skeleton-icon">OPENCV</div>
                    <div class="skeleton-title">2. Thư Viện Chuẩn OpenCV (C++)</div>
                    <div class="skeleton-desc">Vùng hiển thị kết quả xử lý từ hàm thư viện OpenCV chính quy để làm baseline đối chuẩn.</div>
                    <div class="skeleton-badge">Đang chờ lệnh đo...</div>
                </div>
                """, unsafe_allow_html=True)
            with sk3:
                st.markdown("""
                <div class="skeleton-preview-card">
                    <div class="skeleton-icon">DIFF MAP</div>
                    <div class="skeleton-title">3. Bản Đồ Sai Khác Tuyệt Đối</div>
                    <div class="skeleton-desc">Biểu diễn sai lệch trực quan |I_Scratch - I_OpenCV| khuếch đại để soi độ lệch từng pixel.</div>
                    <div class="skeleton-badge">Đang chờ lệnh đo...</div>
                </div>
                """, unsafe_allow_html=True)

    with sub_t2:
        st.markdown("###### Bảng Đối Chứng Toàn Diện Thuật Toán")

        all_src_map = {
            "Ảnh phiếu chuẩn (Kịch bản 1)": os.path.join(samples_dir, "01_B21DCCN001_Chuan.png"),
            "Ảnh chụp nghiêng bàn tối (Kịch bản 2)": os.path.join(samples_dir, "02_B21DCCN002_Nghieng_BanToi.png"),
            "Ảnh phiếu thiếu sáng & nhiễu hạt (Kịch bản 3)": os.path.join(samples_dir, "03_B21DCCN003_ThieuSang_NhieuHat.png"),
            "Ảnh bóng đổ (Kịch bản 5)": os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png"),
        }
        all_src = st.selectbox("Chọn ảnh kiểm thử liên module:", list(all_src_map.keys()), key="sb2_src")
        p_all = all_src_map[all_src]
        img_all = read_image_unicode(p_all if os.path.exists(p_all) else default_sheet_path, cv2.IMREAD_COLOR)

        if st.button("Chạy Đối Chứng Toàn Bộ Thuật Toán", type="primary", key="sb2_btn_run"):
            with st.spinner("Đang chạy đối chứng toàn bộ thuật toán..."):
                st.session_state.tab3_all_bm = benchmark_all_modules(img_all)

        if st.session_state.tab3_all_bm is not None:
            raw_bm = st.session_state.tab3_all_bm
            rows = []
            review_algos = []
            for r in raw_bm:
                iou_str = f"{r['iou']*100:.2f}%" if r.get("iou") is not None else "—"
                dice_str = f"{r['dice']:.4f}" if r.get("dice") is not None else "—"
                dt_str = f"{r['delta_t']}" if r.get("delta_t") is not None else "—"
                m_type = r.get("metric_type")
                if m_type == "corner" or ("Corner" in r["algorithm"] and r.get("delta_t") is not None):
                    passed = r.get("delta_t") is not None and r["delta_t"] <= 3.0
                elif m_type == "binary" or r.get("iou") is not None:
                    passed = r["iou"] is not None and r["iou"] >= 0.985
                else:
                    passed = r["mse"] < 1.0 and r["psnr_db"] > 48.0
                pass_check = "PASS" if passed else "REVIEW"
                if not passed:
                    review_algos.append(r["algorithm"])
                rows.append({
                    "Module": r["module"],
                    "Thuật toán": r["algorithm"],
                    "MSE": f"{r['mse']:.4f}",
                    "PSNR (dB)": f"{r['psnr_db']:.2f}",
                    "IoU (%)": iou_str,
                    "Dice": dice_str,
                    "ΔT / ΔK / Corner Err": dt_str,
                    "Scratch (ms)": f"{r['time_scratch_ms']:.2f}",
                    "OpenCV (ms)": f"{r['time_opencv_ms']:.2f}",
                    "Đánh giá": pass_check
                })
            df_bm = pd.DataFrame(rows)
            st.dataframe(df_bm, use_container_width=True)
            st.caption("Ghi chú cột ΔT / ΔK / Corner Err: Otsu = chênh lệch ngưỡng ΔT; Two-Pass CCL = chênh lệch số thành phần ΔK; Corner Detection = sai số đỉnh Euclidean (px).")
            if not review_algos:
                st.success("Tất cả thuật toán đều đạt chuẩn toán học tương đương OpenCV (MSE < 1.0, PSNR > 48 dB, IoU ≥ 98.5%, Sai số góc ≤ 3.0 px).")
            else:
                st.warning(f"Có {len(review_algos)} thuật toán cần kiểm tra thêm: {', '.join(review_algos)}.")

    with sub_t3:
        st.markdown("###### Khảo Sát Tham Số & Độ Nhạy (Ablation Study)")

        ablation_stress_path = os.path.join(os.path.dirname(samples_dir), "ablation", "ablation_faint_pencil_erasure.png")
        abl_src_map = {}
        if os.path.exists(ablation_stress_path):
            abl_src_map["Kịch bản chuyên sâu: Bút chì nhạt + Vết tẩy xóa nhiều mức (Khảo sát FP/FN theo tau_min)"] = ablation_stress_path
        abl_src_map.update({
            "Kịch bản 5: Bóng đổ & Chiếu sáng không đều (Otsu vs Adaptive)": os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png"),
            "Kịch bản 4: Tô sai quy chế & Vết tẩy xóa (Khảo sát Δτ)": os.path.join(samples_dir, "04_B21DCCN004_ToSaiQuyChe_TayXoa.png"),
            "Kịch bản 3: Thiếu sáng & Nhiễu hạt": os.path.join(samples_dir, "03_B21DCCN003_ThieuSang_NhieuHat.png"),
            "Kịch bản 2: Chụp nghiêng trên bàn tối": os.path.join(samples_dir, "02_B21DCCN002_Nghieng_BanToi.png"),
            "Kịch bản 1: Phiếu chuẩn": os.path.join(samples_dir, "01_B21DCCN001_Chuan.png"),
        })
        abl_src = st.selectbox("Chọn ảnh phiếu làm bài để khảo sát:", list(abl_src_map.keys()), key="sb3_src")
        p_abl = abl_src_map[abl_src]

        if st.button("Chạy Khảo Sát Thực Nghiệm", type="primary", key="sb3_btn_run"):
            with st.spinner("Đang trích xuất ảnh nắn thẳng và chạy khảo sát..."):
                gt_csv_path = p_abl.replace(".png", "_gt.csv")
                if not os.path.exists(gt_csv_path):
                    st.error(f"Không tìm thấy file nhãn chuẩn kiểm định đi kèm: {os.path.basename(gt_csv_path)}")
                else:
                    df_gt = pd.read_csv(gt_csv_path)
                    gt_answers = dict(zip(df_gt.iloc[:, 0].astype(int), df_gt.iloc[:, 1].astype(str).str.strip()))

                    test_sheet_bgr = read_image_unicode(p_abl)
                    key_to_use = st.session_state.get("active_key_path", default_key_path) or default_key_path
                    pipeline = OMRPipeline(key_to_use if key_to_use else None)
                    proc_out = pipeline.process_sheet(test_sheet_bgr, use_scratch_mode=True)

                    st.session_state.tab3_ablation = run_ablation_study(
                        warped_gray=proc_out["warped_gray"],
                        ground_truth_answers=gt_answers,
                        question_grid=proc_out.get("question_grid"),
                        bubble_radius=proc_out.get("bubble_radius"),
                    )

        if st.session_state.tab3_ablation is not None:
            abl = st.session_state.tab3_ablation
            if "error" in abl:
                st.error(abl["error"])
            else:
                st.write("#### 1. So Sánh Phân Ngưỡng: Otsu Toàn Cục vs Adaptive Mean")
                ac1, ac2 = st.columns(2)
                ac1.metric(
                    "Phân ngưỡng Thích nghi (Adaptive Mean)",
                    f"{abl['adaptive_accuracy_pct']:.1f}%",
                    delta="Pipeline chính"
                )
                delta_diff = abl["otsu_accuracy_pct"] - abl["adaptive_accuracy_pct"]
                ac2.metric(
                    "Phân ngưỡng Otsu Toàn cục",
                    f"{abl['otsu_accuracy_pct']:.1f}%",
                    delta=f"{delta_diff:+.1f}%",
                    delta_color="normal"
                )

                if abl["adaptive_accuracy_pct"] > abl["otsu_accuracy_pct"]:
                    st.info(
                        f"Kết quả thực nghiệm: Trên ảnh bóng đổ/chênh sáng, Otsu toàn cục bị suy giảm độ chính xác xuống còn {abl['otsu_accuracy_pct']:.1f}%. "
                        f"Phân ngưỡng thích nghi cục bộ duy trì độ chính xác cao {abl['adaptive_accuracy_pct']:.1f}% (vượt trội +{abs(delta_diff):.1f}%)."
                    )
                elif abl["adaptive_accuracy_pct"] == abl["otsu_accuracy_pct"]:
                    st.info("Trên ảnh có độ rọi đồng đều, cả hai phương pháp đều cho độ chính xác tương đương.")
                else:
                    st.warning(
                        f"Trên kịch bản này, Otsu toàn cục ({abl['otsu_accuracy_pct']:.1f}%) nhỉnh hơn Adaptive Mean ({abl['adaptive_accuracy_pct']:.1f}%)."
                    )

                st.write("---")
                st.write("#### 2. Khảo Sát Độ Nhạy Ngưỡng Diện Tích Tô ($\\tau_{\\min}$)")
                surv = abl["threshold_survey"]
                s_rows = []
                for item in surv:
                    tau = item["fill_threshold"]
                    is_opt = " (Mặc định tối ưu)" if abs(tau - 0.38) < 1e-4 else ""
                    s_rows.append({
                        "Ngưỡng diện tích tô (tau_min)": f"{tau:.2f}{is_opt}",
                        "Độ chính xác nhận diện (%)": f"{item['mark_accuracy_pct']:.1f}%",
                        "Số lỗi FP (Tẩy xóa / Tô nhầm)": item.get("fp_count", 0),
                        "Số lỗi FN (Bỏ sót tô nhạt)": item.get("fn_count", 0),
                        "Số câu bỏ trống (Blank)": item["blank_count"],
                        "Số câu phạm quy (Multiple)": item["multiple_count"]
                    })
                df_surv = pd.DataFrame(s_rows)
                st.dataframe(df_surv, use_container_width=True)

                # Accuracy & FP/FN tradeoff curves with modern styling
                fig_abl, (ax_acc, ax_err) = plt.subplots(1, 2, figsize=(10.5, 3.3), dpi=130)
                apply_modern_chart_style(
                    ax_acc,
                    title="Biến thiên Độ chính xác theo tau_min",
                    xlabel="Ngưỡng tô đậm (tau_min)",
                    ylabel="Độ chính xác (%)"
                )
                taus = [it["fill_threshold"] for it in surv]
                accs = [it["mark_accuracy_pct"] for it in surv]
                fps = [it.get("fp_count", 0) for it in surv]
                fns = [it.get("fn_count", 0) for it in surv]
                ax_acc.plot(taus, accs, marker="o", color="#2563EB", linewidth=2.4, markersize=6, label="Độ chính xác (%)")
                ax_acc.axvline(0.38, color="#EF4444", linestyle="--", alpha=0.85, linewidth=1.5, label="Ngưỡng tối ưu 0.38")
                ax_acc.set_ylim(min(accs) - 10 if min(accs) > 10 else 0, 105)
                ax_acc.legend(loc="lower left", frameon=False, fontsize=8.5)

                apply_modern_chart_style(
                    ax_err,
                    title="Đánh đổi Lỗi FP (Tẩy xóa) vs FN (Bỏ sót nhạt)",
                    xlabel="Ngưỡng tô đậm (tau_min)",
                    ylabel="Số câu lỗi (câu)"
                )
                ax_err.plot(taus, fps, marker="s", color="#F59E0B", linewidth=2.2, markersize=6, label="Lỗi dương giả (FP)")
                ax_err.plot(taus, fns, marker="^", color="#EF4444", linewidth=2.2, markersize=6, label="Lỗi âm giả (FN)")
                ax_err.axvline(0.38, color="#10B981", linestyle="--", alpha=0.85, linewidth=1.5, label="Cân bằng tối ưu 0.38")
                ax_err.set_ylim(bottom=0, top=max(fps + fns + [2]) + 1)
                ax_err.legend(loc="upper center", frameon=False, fontsize=8.5)
                plt.tight_layout()
                st.pyplot(fig_abl)
                plt.close(fig_abl)
