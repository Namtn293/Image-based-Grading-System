"""Tab 2: Kiểm Thử Thuật Toán Từng Phân Hệ (Pipeline Stepper).
Clean, academic interface without text emojis.
"""

import os
import io
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Imports for algorithms across modules
try:
    from tv1_preprocessing import (
        rgb_to_gray_scratch,
        gaussian_blur_scratch,
        canny_edge_scratch,
        detect_sheet_corners
    )
except ImportError:
    rgb_to_gray_scratch = None
    gaussian_blur_scratch = None
    canny_edge_scratch = None
    detect_sheet_corners = None

try:
    from tv2_perspective import (
        order_corners,
        destination_corners,
        solve_homography_svd,
        warp_perspective_bilinear,
        detect_markers_and_correct_rotation,
        corners_for_angle,
    )
except ImportError:
    order_corners = None
    destination_corners = None
    solve_homography_svd = None
    warp_perspective_bilinear = None
    detect_markers_and_correct_rotation = None
    corners_for_angle = None

try:
    from tv3_thresholding import (
        otsu_threshold_scratch,
        adaptive_mean_threshold_scratch
    )
    from tv3_morphology_ccl import (
        opening_binary_scratch,
        closing_binary_scratch,
        morphology_binary_scratch,
        two_pass_ccl_scratch
    )
except ImportError:
    otsu_threshold_scratch = None
    adaptive_mean_threshold_scratch = None
    opening_binary_scratch = None
    closing_binary_scratch = None
    morphology_binary_scratch = None
    two_pass_ccl_scratch = None

from tv4_enhancement_eval import (
    histogram_equalization_scratch,
    median_filter_scratch,
    load_matrix_from_csv
)
from pipeline_coordinator import (
    create_anchor_marker_template,
    _detect_orientation_cv,
    _detect_sheet_corners_cv,
)
from ui.styles import apply_modern_chart_style, read_image_unicode


def _to_gray_for_mode(img: np.ndarray, use_scratch: bool) -> np.ndarray:
    """Convert BGR or grayscale image to 2D uint8 grayscale using Scratch or OpenCV."""
    if img.ndim == 2:
        return img.copy()
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0].copy()
    if use_scratch and rgb_to_gray_scratch is not None:
        return rgb_to_gray_scratch(img, is_bgr=True)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def render_tab2(
    use_scratch: bool,
    samples_dir: str,
    default_sheet_path: str,
    default_csv_15_path: str,
    default_csv_low_path: str,
    default_csv_60_path: str
):
    """Render Tab 2: 4-Stage Interactive Pipeline Stepper."""

    mode_badge = "Thuật toán tự cài đặt (NumPy Scratch)" if use_scratch else "Thư viện chuẩn OpenCV (Baseline)"
    st.markdown("##### Kiểm Thử Thuật Toán Từng Phân Hệ (Pipeline Stepper)")
    st.caption(f"Lõi tính toán hiện tại (theo Sidebar): **{mode_badge}**")

    step_items = [
        {"num": 1, "title": "1. Tiền Xử Lý & Khung", "sub": "Lọc Gauss, Canny & 4 Góc Khung"},
        {"num": 2, "title": "2. Nắn Phẳng SVD & NCC", "sub": "DLT Homography 3x3 & Chuẩn Hóa Góc Xoay"},
        {"num": 3, "title": "3. Phân Đoạn & CCL", "sub": "Otsu, Adaptive, Hình Thái & CCL"},
        {"num": 4, "title": "4. Cải Thiện & CSV", "sub": "Lọc Median, Histogram & Ma Trận CSV"},
    ]

    cur_step = st.session_state.get("pipeline_step_idx", 0)
    if cur_step >= len(step_items) or cur_step < 0:
        cur_step = 0
        st.session_state.pipeline_step_idx = 0

    # Stepper Navigation Buttons (Direct Interactive Stepper)
    step_cols = st.columns(4)
    for idx, (col, item) in enumerate(zip(step_cols, step_items)):
        with col:
            is_active = (cur_step == idx)
            prefix = "[Đang chọn] " if is_active else ""
            if st.button(
                f"{prefix}{item['title']}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                key=f"stepper_btn_{idx}"
            ):
                st.session_state.pipeline_step_idx = idx
                st.rerun()

    progress_ratio = (cur_step + 1) / 4.0
    st.progress(
        progress_ratio,
        text=f"Tiến trình Pipeline: Bước {cur_step + 1}/4 ({int(progress_ratio * 100)}%) — {step_items[cur_step]['sub']}"
    )
    st.write("")

    # -----------------------------------------------------
    # BƯỚC 1: TIỀN XỬ LÝ & ĐỊNH VỊ KHUNG PHIẾU
    # -----------------------------------------------------
    if cur_step == 0:
        st.markdown("###### Giai Đoạn 1: Tiền Xử Lý Ảnh & Định Vị Khung Phiếu Thi")

        m1_col_ctrl, m1_col_view = st.columns([1.1, 1.9], gap="large")
        with m1_col_ctrl:
            with st.container(border=True):
                st.markdown("##### Cấu Hình Tiền Xử Lý")
                m1_src_map = {
                    "Phiếu chuẩn (01_B21DCCN001_Chuan.png)": os.path.join(samples_dir, "01_B21DCCN001_Chuan.png"),
                    "Chụp nghiêng bàn tối (02_B21DCCN002_Nghieng_BanToi.png)": os.path.join(samples_dir, "02_B21DCCN002_Nghieng_BanToi.png"),
                    "Thiếu sáng + Nhiễu hạt (03_B21DCCN003_ThieuSang_NhieuHat.png)": os.path.join(samples_dir, "03_B21DCCN003_ThieuSang_NhieuHat.png"),
                    "Bóng đổ chéo (05_B21DCCN005_BongDo_ChieuSangKhongDeu.png)": os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png"),
                    "Tải ảnh từ máy tính": None,
                }
                m1_src_opt = st.selectbox("Chọn ảnh đầu vào:", list(m1_src_map.keys()), key="m1_src_opt")

                m1_img_bgr = None
                if m1_src_map[m1_src_opt] is None:
                    up_m1 = st.file_uploader("Nạp ảnh (.png, .jpg):", type=["png", "jpg", "jpeg"], key="up_m1")
                    if up_m1 is not None:
                        fb = np.asarray(bytearray(up_m1.getvalue()), dtype=np.uint8)
                        m1_img_bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                else:
                    p = m1_src_map[m1_src_opt]
                    m1_img_bgr = read_image_unicode(p if os.path.exists(p) else default_sheet_path)

                m1_algo = st.selectbox(
                    "Chọn thuật toán:",
                    [
                        "1. Chuyển đổi mức xám (Grayscale ITU-R BT.601)",
                        "2. Làm mờ Gauss 2D (Gaussian Blur)",
                        "3. Phát hiện biên Canny 4 bước (Canny Edge Detection)",
                        "4. Định vị 4 đỉnh góc khung phiếu (Sheet Corner Detection)"
                    ],
                    key="m1_algo"
                )
                m1_params = {}
                if "Gauss" in m1_algo:
                    gk_col, gs_col = st.columns(2)
                    with gk_col:
                        m1_params["ksize"] = st.selectbox("Kernel Size:", [3, 5, 7], index=1, key="m1_gk")
                    with gs_col:
                        m1_params["sigma"] = st.slider("Sigma:", 0.5, 3.0, 1.4, 0.1, key="m1_gs")
                elif "Canny" in m1_algo:
                    cl_col, ch_col = st.columns(2)
                    with cl_col:
                        m1_params["low"] = st.slider("Low Threshold:", 10.0, 100.0, 50.0, 5.0, key="m1_cl")
                    with ch_col:
                        m1_params["high"] = st.slider("High Threshold:", 50.0, 250.0, 150.0, 5.0, key="m1_ch")

                btn_run_m1 = st.button("Thực Thi Tiền Xử Lý", type="primary", use_container_width=True, key="btn_run_m1")

                if m1_img_bgr is not None and btn_run_m1:
                    t0 = time.perf_counter()
                    gray = _to_gray_for_mode(m1_img_bgr, use_scratch=use_scratch)
                    out_vis = None
                    extra_info = None
                    used_core = "Scratch (NumPy)" if use_scratch else "OpenCV Baseline"

                    if "Grayscale" in m1_algo:
                        out_vis = gray
                    elif "Gauss" in m1_algo:
                        k_g = m1_params.get("ksize", 5)
                        sig = m1_params.get("sigma", 1.4)
                        if use_scratch and gaussian_blur_scratch is not None:
                            out_vis = gaussian_blur_scratch(gray, kernel_size=k_g, sigma=sig)
                        else:
                            out_vis = cv2.GaussianBlur(gray, (k_g, k_g), sig, borderType=cv2.BORDER_REFLECT_101)
                    elif "Canny" in m1_algo:
                        low_t = m1_params.get("low", 50.0)
                        high_t = m1_params.get("high", 150.0)
                        if use_scratch and canny_edge_scratch is not None:
                            out_vis = canny_edge_scratch(gray, low_thresh=low_t, high_thresh=high_t)
                        else:
                            b_cv = cv2.GaussianBlur(gray, (5, 5), 1.4, borderType=cv2.BORDER_REFLECT_101)
                            out_vis = cv2.Canny(b_cv, int(low_t), int(high_t), L2gradient=True)
                    else:
                        if detect_sheet_corners is not None:
                            corners = detect_sheet_corners(gray, use_scratch=use_scratch)
                        else:
                            corners = _detect_sheet_corners_cv(gray)

                        vis_corners = m1_img_bgr.copy() if m1_img_bgr.ndim == 3 else cv2.cvtColor(m1_img_bgr, cv2.COLOR_GRAY2BGR)
                        corner_colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
                        corner_names = ["TL (Top-Left)", "TR (Top-Right)", "BR (Bottom-Right)", "BL (Bottom-Left)"]
                        pts_poly = corners.astype(np.int32)
                        cv2.polylines(vis_corners, [pts_poly], isClosed=True, color=(0, 255, 0), thickness=3)
                        for pt, col, name in zip(corners, corner_colors, corner_names):
                            cx, cy = int(pt[0]), int(pt[1])
                            cv2.circle(vis_corners, (cx, cy), 12, col, -1)
                            cv2.circle(vis_corners, (cx, cy), 14, (255, 255, 255), 2)
                            cv2.putText(vis_corners, name.split()[0], (cx + 14, cy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                        out_vis = vis_corners
                        extra_info = {
                            "corners": corners,
                            "names": corner_names
                        }

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    st.session_state.tab2_m1_result = {
                        "in_bgr": m1_img_bgr,
                        "out_vis": out_vis,
                        "algo": m1_algo,
                        "mode": used_core,
                        "elapsed_ms": elapsed_ms,
                        "extra": extra_info
                    }

                if st.session_state.tab2_m1_result is not None:
                    r1 = st.session_state.tab2_m1_result
                    st.divider()
                    st.metric("Thời gian thực thi", f"{r1['elapsed_ms']:.2f} ms", delta=r1.get("mode", ""))
                    ok_png, png_buf = cv2.imencode(".png", r1["out_vis"])
                    if ok_png:
                        st.download_button(
                            label="Tải Ảnh Kết Quả (.png)",
                            data=png_buf.tobytes(),
                            file_name="ket_qua_tien_xu_ly.png",
                            mime="image/png",
                            use_container_width=True,
                            key="dl_btn_tv1"
                        )

        with m1_col_view:
            with st.container(border=True):
                st.markdown("##### Không Gian Trực Quan & Kết Quả")
                if st.session_state.tab2_m1_result is not None:
                    r1 = st.session_state.tab2_m1_result
                    st.success(f"Đã thực thi {r1['algo']} ({r1.get('mode', '')}) trong {r1['elapsed_ms']:.2f} ms.")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("###### Ảnh Đầu Vào")
                        st.image(cv2.cvtColor(r1["in_bgr"], cv2.COLOR_BGR2RGB) if r1["in_bgr"].ndim == 3 else r1["in_bgr"], use_container_width=True)
                    with c2:
                        st.markdown("###### Kết Quả Xử Lý")
                        if r1["out_vis"].ndim == 2:
                            st.image(r1["out_vis"], use_container_width=True, channels="GRAY")
                        else:
                            st.image(cv2.cvtColor(r1["out_vis"], cv2.COLOR_BGR2RGB), use_container_width=True)

                    if r1["extra"] and "corners" in r1["extra"]:
                        st.markdown("###### Tọa Độ 4 Đỉnh Góc Khung Phiếu Phát Hiện Được:")
                        c_rows = []
                        for name, pt in zip(r1["extra"]["names"], r1["extra"]["corners"]):
                            c_rows.append({
                                "Đỉnh": name,
                                "Tọa độ X (pixel)": f"{pt[0]:.1f}",
                                "Tọa độ Y (pixel)": f"{pt[1]:.1f}"
                            })
                        st.table(pd.DataFrame(c_rows))
                else:
                    if m1_img_bgr is not None:
                        st.image(cv2.cvtColor(m1_img_bgr, cv2.COLOR_BGR2RGB), caption=f"Ảnh đầu vào hiện tại ({m1_img_bgr.shape[1]}x{m1_img_bgr.shape[0]})", use_container_width=True)
                        st.info("Bấm nút 'Thực Thi Tiền Xử Lý' ở cột điều khiển bên trái để bắt đầu chạy.")

    # -----------------------------------------------------
    # BƯỚC 2: NẮN PHẲNG PHỐI CẢNH SVD & ĐỊNH VỊ NCC
    # -----------------------------------------------------
    elif cur_step == 1:
        st.markdown("###### Giai Đoạn 2: Nắn Phẳng Phối Cảnh SVD & Chuẩn Hóa Hướng Phiếu NCC")

        m2_col_ctrl, m2_col_view = st.columns([1.1, 1.9], gap="large")
        with m2_col_ctrl:
            with st.container(border=True):
                st.markdown("##### Cấu Hình Nắn Phẳng SVD & NCC")
                m2_src_map = {
                    "Ảnh chụp nghiêng trên bàn tối (02_B21DCCN002_Nghieng_BanToi.png - Khuyến nghị)": os.path.join(samples_dir, "02_B21DCCN002_Nghieng_BanToi.png"),
                    "Ảnh phiếu chuẩn (01_B21DCCN001_Chuan.png)": os.path.join(samples_dir, "01_B21DCCN001_Chuan.png"),
                    "Ảnh bóng đổ chéo (05_B21DCCN005_BongDo_ChieuSangKhongDeu.png)": os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png"),
                    "Tải ảnh tùy chọn từ máy tính": None,
                }
                m2_src_opt = st.selectbox("Chọn ảnh kiểm thử nắn phẳng:", list(m2_src_map.keys()), key="m2_src_opt")

                m2_img_bgr = None
                if m2_src_map[m2_src_opt] is None:
                    up_m2 = st.file_uploader("Nạp ảnh nghiêng (.png, .jpg):", type=["png", "jpg", "jpeg"], key="up_m2")
                    if up_m2 is not None:
                        fb = np.asarray(bytearray(up_m2.getvalue()), dtype=np.uint8)
                        m2_img_bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                else:
                    p = m2_src_map[m2_src_opt]
                    m2_img_bgr = read_image_unicode(p if os.path.exists(p) else default_sheet_path)

                rot_opt = st.selectbox(
                    "Mô phỏng góc xoay đầu vào (Kiểm thử NCC):",
                    ["0° (Gốc)", "90° (Xoay phải)", "180° (Lật ngược)", "270° (Xoay trái)"],
                    index=0,
                    key="m2_rot_opt"
                )
                rot_k_map = {"0° (Gốc)": 0, "90° (Xoay phải)": 3, "180° (Lật ngược)": 2, "270° (Xoay trái)": 1}
                rot_k = rot_k_map[rot_opt]

                enable_ncc = st.checkbox(
                    "Tự động định vị Anchor Marker (NCC) & chuẩn hóa hướng 0°",
                    value=True,
                    key="m2_enable_ncc"
                )

                target_size_label = st.selectbox(
                    "Kích thước chuẩn hóa (W x H):",
                    ["800 x 1000 (Chuẩn hệ thống OMR)", "600 x 800", "1000 x 1200"],
                    index=0,
                    key="m2_size"
                )
                w_tgt, h_tgt = (800, 1000) if "800 x 1000" in target_size_label else ((600, 800) if "600" in target_size_label else (1000, 1200))

                btn_run_m2 = st.button("Thực Thi Nắn Phẳng Phối Cảnh SVD", type="primary", use_container_width=True, key="btn_run_m2")

                if m2_img_bgr is not None and btn_run_m2:
                    m2_work = np.rot90(m2_img_bgr, rot_k).copy() if rot_k else m2_img_bgr.copy()
                    t0 = time.perf_counter()
                    gray = _to_gray_for_mode(m2_work, use_scratch=use_scratch)

                    if detect_sheet_corners is not None:
                        src_corners = detect_sheet_corners(gray, use_scratch=use_scratch)
                    else:
                        src_corners = _detect_sheet_corners_cv(gray)

                    warped_out = None
                    H_mat = None
                    final_src = None
                    detected_angle = 0
                    marker_scores = []
                    used_core = "Scratch (NumPy)" if use_scratch else "OpenCV Baseline"

                    try:
                        if order_corners is not None:
                            ordered_src, auto_size = order_corners(src_corners)
                        else:
                            ordered_src = src_corners.astype(np.float32)
                            tl, tr, br, bl = ordered_src.astype(np.float64)
                            w_est = max(int(round(max(np.hypot(*(tr - tl)), np.hypot(*(br - bl))))), 1)
                            h_est = max(int(round(max(np.hypot(*(bl - tl)), np.hypot(*(br - tr))))), 1)
                            auto_size = (w_est, h_est)

                        final_src = ordered_src
                        if enable_ncc:
                            same_orient = (auto_size[0] > auto_size[1]) == (800 > 1000)
                            preview_size = (800, 1000) if same_orient else (1000, 800)
                            if use_scratch and solve_homography_svd is not None and warp_perspective_bilinear is not None and destination_corners is not None:
                                H_prev = solve_homography_svd(ordered_src, destination_corners(preview_size))
                                preview_gray = warp_perspective_bilinear(gray, H_prev, preview_size)
                            else:
                                pw, ph = preview_size
                                dst_prev = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
                                M_prev = cv2.getPerspectiveTransform(ordered_src.astype(np.float32), dst_prev)
                                preview_gray = cv2.warpPerspective(gray, M_prev, (pw, ph), flags=cv2.INTER_LINEAR)

                            anchor_tmpl = create_anchor_marker_template(30)
                            if use_scratch and detect_markers_and_correct_rotation is not None:
                                _, detected_angle, marker_scores = detect_markers_and_correct_rotation(preview_gray, anchor_tmpl)
                            else:
                                detected_angle, marker_scores = _detect_orientation_cv(preview_gray, anchor_tmpl)

                            sorted_sc = sorted((s for _, s in marker_scores), reverse=True)
                            top1 = sorted_sc[0] if len(sorted_sc) >= 1 else -1.0
                            top2 = sorted_sc[1] if len(sorted_sc) >= 2 else -1.0
                            if top1 < 0.72 or (top1 - top2) < 0.15:
                                detected_angle = 0

                            if detected_angle != 0:
                                if corners_for_angle is not None:
                                    final_src = corners_for_angle(ordered_src, detected_angle)
                                else:
                                    final_src = np.roll(ordered_src, -(detected_angle // 90), axis=0)

                        dst_corners = np.array([
                            [0, 0],
                            [w_tgt - 1, 0],
                            [w_tgt - 1, h_tgt - 1],
                            [0, h_tgt - 1]
                        ], dtype=np.float32)

                        if use_scratch and solve_homography_svd is not None and warp_perspective_bilinear is not None:
                            dst_for_svd = destination_corners((w_tgt, h_tgt)) if destination_corners is not None else dst_corners
                            H_mat = solve_homography_svd(final_src, dst_for_svd)
                            warped_out = warp_perspective_bilinear(m2_work, H_mat, (w_tgt, h_tgt))
                        else:
                            H_mat = cv2.getPerspectiveTransform(final_src.astype(np.float32), dst_corners)
                            warped_out = cv2.warpPerspective(m2_work, H_mat, (w_tgt, h_tgt), flags=cv2.INTER_LINEAR)
                    except ValueError as exc:
                        st.error(f"Không thể nắn phẳng phối cảnh: {exc}")
                        st.session_state.tab2_m2_result = None

                    if warped_out is not None and H_mat is not None and final_src is not None:
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        vis_src = m2_work.copy() if m2_work.ndim == 3 else cv2.cvtColor(m2_work, cv2.COLOR_GRAY2BGR)
                        pts_poly = final_src.astype(np.int32)
                        cv2.polylines(vis_src, [pts_poly], isClosed=True, color=(0, 255, 0), thickness=3)
                        corner_names = ["TL", "TR", "BR", "BL"]
                        for pt, name in zip(final_src, corner_names):
                            cx, cy = int(pt[0]), int(pt[1])
                            cv2.circle(vis_src, (cx, cy), 10, (0, 0, 255), -1)
                            cv2.putText(vis_src, name, (cx + 12, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        st.session_state.tab2_m2_result = {
                            "vis_src": vis_src,
                            "warped": warped_out,
                            "H_mat": H_mat,
                            "src_corners": final_src,
                            "detected_angle": detected_angle,
                            "marker_scores": marker_scores,
                            "mode": used_core,
                            "elapsed_ms": elapsed_ms,
                            "out_size": (w_tgt, h_tgt)
                        }

                if st.session_state.tab2_m2_result is not None:
                    r2 = st.session_state.tab2_m2_result
                    st.divider()
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Thời gian thực thi", f"{r2['elapsed_ms']:.2f} ms")
                    mc2.metric("Góc xoay NCC phát hiện", f"{r2.get('detected_angle', 0)}°")
                    st.markdown("###### Ma Trận Homography $\\mathbf{H}_{3 \\times 3}$:")
                    h_df = pd.DataFrame(r2["H_mat"], columns=["h_x", "h_y", "h_w"], index=["row 1", "row 2", "row 3"])
                    st.dataframe(h_df.style.format("{:.5f}"), use_container_width=True)
                    ok_png, png_buf = cv2.imencode(".png", r2["warped"])
                    if ok_png:
                        st.download_button(
                            label="Tải Ảnh Đã Nắn Phẳng (.png)",
                            data=png_buf.tobytes(),
                            file_name="anh_nan_phang.png",
                            mime="image/png",
                            use_container_width=True,
                            key="dl_btn_tv2"
                        )

        with m2_col_view:
            with st.container(border=True):
                st.markdown("##### Không Gian Trực Quan & So Sánh Phối Cảnh")
                if st.session_state.tab2_m2_result is not None:
                    r2 = st.session_state.tab2_m2_result
                    st.success(
                        f"Đã giải hệ Homography ({r2.get('mode', '')}) & nắn thẳng bài thi trong {r2['elapsed_ms']:.2f} ms "
                        f"(Góc NCC phát hiện: {r2.get('detected_angle', 0)}°)."
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("###### Ảnh Đầu Vào (4 Đỉnh Chuẩn Hóa)")
                        st.image(cv2.cvtColor(r2["vis_src"], cv2.COLOR_BGR2RGB) if r2["vis_src"].ndim == 3 else r2["vis_src"], use_container_width=True)
                    with c2:
                        st.markdown(f"###### Ảnh Sau Nắn Phẳng ({r2['out_size'][0]}x{r2['out_size'][1]})")
                        st.image(cv2.cvtColor(r2["warped"], cv2.COLOR_BGR2RGB) if r2["warped"].ndim == 3 else r2["warped"], use_container_width=True)

                    if r2.get("marker_scores"):
                        st.markdown("###### Điểm Tương Quan NCC Anchor Marker Tại 4 Góc:")
                        ncc_rows = [{"Góc": c, "Điểm NCC": f"{s:.4f}"} for c, s in r2["marker_scores"]]
                        st.table(pd.DataFrame(ncc_rows))
                else:
                    if m2_img_bgr is not None:
                        preview_in = np.rot90(m2_img_bgr, rot_k).copy() if rot_k else m2_img_bgr
                        st.image(
                            cv2.cvtColor(preview_in, cv2.COLOR_BGR2RGB),
                            caption=f"Ảnh đầu vào hiện tại ({preview_in.shape[1]}x{preview_in.shape[0]}, xoay mô phỏng {rot_opt})",
                            use_container_width=True
                        )
                        st.info("Bấm nút 'Thực Thi Nắn Phẳng Phối Cảnh SVD' ở cột bên trái để nắn thẳng tờ phiếu.")

    # -----------------------------------------------------
    # BƯỚC 3: PHÂN ĐOẠN NHỊ PHÂN & ĐÁNH NHÃN CCL
    # -----------------------------------------------------
    elif cur_step == 2:
        st.markdown("###### Giai Đoạn 3: Phân Đoạn Nhị Phân, Hình Thái Học & Đánh Nhãn CCL")

        m3_col_ctrl, m3_col_view = st.columns([1.1, 1.9], gap="large")
        with m3_col_ctrl:
            with st.container(border=True):
                st.markdown("##### Cấu Hình Phân Đoạn & CCL")
                m3_src_map = {
                    "Ảnh phiếu chuẩn nắn thẳng (sample_sheet.png)": default_sheet_path,
                    "Ảnh thiếu sáng + nhiễu hạt (03_B21DCCN003_ThieuSang_NhieuHat.png)": os.path.join(samples_dir, "03_B21DCCN003_ThieuSang_NhieuHat.png"),
                    "Ảnh bóng đổ chéo (05_B21DCCN005_BongDo_ChieuSangKhongDeu.png)": os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png"),
                    "Ảnh tô sai quy chế & tẩy xóa (04_B21DCCN004_ToSaiQuyChe_TayXoa.png)": os.path.join(samples_dir, "04_B21DCCN004_ToSaiQuyChe_TayXoa.png"),
                    "Tải ảnh tùy chọn từ máy tính": None,
                }
                m3_src_opt = st.selectbox("Chọn ảnh đầu vào:", list(m3_src_map.keys()), key="m3_src_opt")

                m3_img = None
                if m3_src_map[m3_src_opt] is None:
                    up_m3 = st.file_uploader("Nạp ảnh (.png, .jpg):", type=["png", "jpg", "jpeg"], key="up_m3")
                    if up_m3 is not None:
                        fb = np.asarray(bytearray(up_m3.getvalue()), dtype=np.uint8)
                        m3_img = cv2.imdecode(fb, cv2.IMREAD_GRAYSCALE)
                else:
                    p = m3_src_map[m3_src_opt]
                    m3_img = read_image_unicode(p if os.path.exists(p) else default_sheet_path, cv2.IMREAD_GRAYSCALE)

                m3_algo = st.selectbox(
                    "Chọn thuật toán phân đoạn:",
                    [
                        "1. Phân ngưỡng Otsu toàn cục (Otsu's Thresholding)",
                        "2. Phân ngưỡng thích nghi cục bộ (Adaptive Mean Thresholding)",
                        "3. Phép toán hình thái học (Morphology Opening / Closing)",
                        "4. Gán nhãn thành phần liên thông (Two-Pass CCL & Disjoint-Set)"
                    ],
                    key="m3_algo"
                )
                m3_params = {}
                if "Adaptive" in m3_algo:
                    bs_col, c_col = st.columns(2)
                    with bs_col:
                        m3_params["block_size"] = st.selectbox("Block Size:", [31, 51, 71], index=1, key="m3_bs")
                    with c_col:
                        m3_params["C"] = st.slider("Hằng số C:", 10.0, 60.0, 42.0, 2.0, key="m3_c")
                elif "hình thái" in m3_algo:
                    m3_params["base"] = st.selectbox("Mặt nạ nhị phân nền:", ["Adaptive Mean (51, 42)", "Otsu toàn cục"], key="m3_morph_base")
                    m3_params["op"] = st.selectbox(
                        "Toán tử hình thái:",
                        ["Mở (Opening - Khử nhiễu trắng)", "Đóng (Closing - Lấp đầy lỗ hổng)", "Co (Erosion)", "Giãn (Dilation)"],
                        key="m3_op"
                    )
                elif "CCL" in m3_algo:
                    m3_params["base"] = st.selectbox("Mặt nạ nhị phân nền:", ["Adaptive Mean (51, 42)", "Otsu toàn cục"], key="m3_ccl_base")
                    m3_params["connectivity"] = st.selectbox("Kiểu lân cận CCL:", [8, 4], index=0, key="m3_conn")

                btn_run_m3 = st.button("Thực Thi Phân Đoạn & CCL", type="primary", use_container_width=True, key="btn_run_m3")

                def _get_base_binary(img_g: np.ndarray, base_label: str) -> np.ndarray:
                    if "Adaptive" in base_label:
                        if use_scratch and adaptive_mean_threshold_scratch is not None:
                            return adaptive_mean_threshold_scratch(img_g, block_size=51, C=42.0)
                        return cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 42)
                    if use_scratch and otsu_threshold_scratch is not None:
                        _, b_out = otsu_threshold_scratch(img_g, method="BINARY_INV")
                        return b_out
                    _, b_out = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    return b_out

                if m3_img is not None and btn_run_m3:
                    t0 = time.perf_counter()
                    out_bin = None
                    extra_data = None
                    used_core = "Scratch (NumPy)" if use_scratch else "OpenCV Baseline"

                    if "Otsu" in m3_algo:
                        if use_scratch and otsu_threshold_scratch is not None:
                            t_val, out_bin = otsu_threshold_scratch(m3_img, method="BINARY_INV")
                        else:
                            t_val, out_bin = cv2.threshold(m3_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        extra_data = {"otsu_t": t_val}
                    elif "Adaptive" in m3_algo:
                        bs_val = int(m3_params.get("block_size", 51))
                        c_val = float(m3_params.get("C", 42.0))
                        if use_scratch and adaptive_mean_threshold_scratch is not None:
                            out_bin = adaptive_mean_threshold_scratch(m3_img, block_size=bs_val, C=c_val)
                        else:
                            out_bin = cv2.adaptiveThreshold(m3_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, bs_val, c_val)
                    elif "hình thái" in m3_algo:
                        bin_base = _get_base_binary(m3_img, m3_params.get("base", "Adaptive"))
                        op_choice = m3_params.get("op", "Mở")
                        k_cv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        if "Mở" in op_choice:
                            out_bin = opening_binary_scratch(bin_base) if (use_scratch and opening_binary_scratch is not None) else cv2.morphologyEx(bin_base, cv2.MORPH_OPEN, k_cv)
                        elif "Đóng" in op_choice:
                            out_bin = closing_binary_scratch(bin_base) if (use_scratch and closing_binary_scratch is not None) else cv2.morphologyEx(bin_base, cv2.MORPH_CLOSE, k_cv)
                        elif "Co" in op_choice:
                            out_bin = morphology_binary_scratch(bin_base, op="erode") if (use_scratch and morphology_binary_scratch is not None) else cv2.erode(bin_base, k_cv)
                        else:
                            out_bin = morphology_binary_scratch(bin_base, op="dilate") if (use_scratch and morphology_binary_scratch is not None) else cv2.dilate(bin_base, k_cv)
                    else:
                        bin_base = _get_base_binary(m3_img, m3_params.get("base", "Adaptive"))
                        conn = int(m3_params.get("connectivity", 8))
                        if use_scratch and two_pass_ccl_scratch is not None:
                            num_comp, labels, stats = two_pass_ccl_scratch(bin_base, connectivity=conn)
                        else:
                            num_comp, labels, cv_stats, _ = cv2.connectedComponentsWithStats(bin_base, connectivity=conn)
                            num_comp -= 1
                            stats = []
                            for li in range(1, num_comp + 1):
                                stats.append({
                                    "label": li,
                                    "area": int(cv_stats[li, cv2.CC_STAT_AREA]),
                                    "bbox": (int(cv_stats[li, cv2.CC_STAT_LEFT]), int(cv_stats[li, cv2.CC_STAT_TOP]), int(cv_stats[li, cv2.CC_STAT_WIDTH]), int(cv_stats[li, cv2.CC_STAT_HEIGHT]))
                                })

                        rng = np.random.default_rng(42)
                        colors = rng.integers(50, 255, size=(num_comp + 1, 3), dtype=np.uint8)
                        colors[0] = [0, 0, 0]
                        out_bin = colors[labels]
                        extra_data = {"num_comp": num_comp, "stats": stats}

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    st.session_state.tab2_m3_result = {
                        "in_gray": m3_img,
                        "out_bin": out_bin,
                        "algo": m3_algo,
                        "mode": used_core,
                        "elapsed_ms": elapsed_ms,
                        "extra": extra_data
                    }

                if st.session_state.tab2_m3_result is not None:
                    r3 = st.session_state.tab2_m3_result
                    st.divider()
                    st.metric("Thời gian thực thi", f"{r3['elapsed_ms']:.2f} ms", delta=r3.get("mode", ""))
                    png_input = r3["out_bin"] if r3["out_bin"].ndim == 2 else cv2.cvtColor(r3["out_bin"], cv2.COLOR_RGB2BGR)
                    ok_png, png_buf = cv2.imencode(".png", png_input)
                    if ok_png:
                        st.download_button(
                            label="Tải Mặt Nạ Nhị Phân (.png)",
                            data=png_buf.tobytes(),
                            file_name="mat_na_nhi_phan.png",
                            mime="image/png",
                            use_container_width=True,
                            key="dl_btn_tv3"
                        )

        with m3_col_view:
            with st.container(border=True):
                st.markdown("##### Không Gian Trực Quan & Mặt Nạ Nhị Phân")
                if st.session_state.tab2_m3_result is not None:
                    r3 = st.session_state.tab2_m3_result
                    st.success(f"Đã thực thi thành công {r3['algo']} ({r3.get('mode', '')}) trong {r3['elapsed_ms']:.2f} ms.")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("###### Ảnh Xám Đầu Vào")
                        st.image(r3["in_gray"], use_container_width=True, channels="GRAY")
                    with c2:
                        st.markdown("###### Mặt Nạ Nhị Phân / Nhãn CCL")
                        st.image(r3["out_bin"], use_container_width=True, channels="RGB" if r3["out_bin"].ndim == 3 else "GRAY")

                    if r3["extra"] and "otsu_t" in r3["extra"]:
                        st.info(f"Ngưỡng tối ưu toàn cục tính được bởi Otsu: T* = {r3['extra']['otsu_t']}")
                    elif r3["extra"] and "num_comp" in r3["extra"]:
                        st.markdown(f"###### Thống Kê {r3['extra']['num_comp']} Thành Phần Liên Thông (CCL):")
                        df_ccl = pd.DataFrame(r3["extra"]["stats"])
                        st.dataframe(df_ccl.head(15), use_container_width=True, height=220)
                else:
                    if m3_img is not None:
                        st.image(m3_img, caption="Ảnh xám đầu vào hiện tại", use_container_width=True, channels="GRAY")
                        st.info("Bấm nút 'Thực Thi Phân Đoạn & CCL' ở cột bên trái để chạy.")

    # -----------------------------------------------------
    # BƯỚC 4: CÂN BẰNG SÁNG, KHỬ NHIỄU & MA TRẬN CSV
    # -----------------------------------------------------
    else:
        st.markdown("###### Giai Đoạn 4: Cân Bằng Sáng, Khử Nhiễu & Ma Trận Mức Xám CSV")

        m4_col_ctrl, m4_col_view = st.columns([1.1, 1.9], gap="large")
        with m4_col_ctrl:
            with st.container(border=True):
                st.markdown("##### Nguồn Dữ Liệu & Thuật Toán")
                source_type = st.radio(
                    "Nguồn dữ liệu kiểm thử:",
                    [
                        "Ma trận mức xám dạng File CSV",
                        "Ảnh chụp phiếu thi (.png / .jpg)"
                    ],
                    key="m4_src_type"
                )

                test_matrix = None
                source_name = ""

                if "File CSV" in source_type:
                    csv_mode = st.selectbox(
                        "Chọn file CSV kiểm thử:",
                        [
                            "Mẫu 1: Ma trận 15x15 có nhiễu muối tiêu (sample_matrix_15x15.csv)",
                            "Mẫu 2: Ma trận 15x15 tương phản thấp dải [55-95] (sample_matrix_low_contrast.csv)",
                            "Mẫu 3: Ma trận 60x60 có 4 điểm nhiễu (sample_matrix.csv)",
                            "Tải file CSV bất kỳ từ máy tính"
                        ],
                        key="m4_csv_mode"
                    )
                    if "Mẫu 1" in csv_mode:
                        if os.path.exists(default_csv_15_path):
                            test_matrix = load_matrix_from_csv(default_csv_15_path)
                            source_name = "sample_matrix_15x15.csv"
                    elif "Mẫu 2" in csv_mode:
                        if os.path.exists(default_csv_low_path):
                            test_matrix = load_matrix_from_csv(default_csv_low_path)
                            source_name = "sample_matrix_low_contrast.csv"
                    elif "Mẫu 3" in csv_mode:
                        if os.path.exists(default_csv_60_path):
                            test_matrix = load_matrix_from_csv(default_csv_60_path)
                            source_name = "sample_matrix.csv"
                    else:
                        up_csv = st.file_uploader("Nạp file ma trận CSV (.csv, .txt):", type=["csv", "txt"], key="m4_up_csv")
                        if up_csv is not None:
                            try:
                                raw_text = up_csv.getvalue().decode("utf-8-sig")
                                test_matrix = load_matrix_from_csv(io.StringIO(raw_text))
                                source_name = up_csv.name
                            except Exception as e:
                                st.error(f"Lỗi đọc file CSV: {e}")
                else:
                    m4_img_map = {
                        "Ảnh mẫu thiếu sáng + nhiễu muối tiêu (03_B21DCCN003_ThieuSang_NhieuHat.png)": os.path.join(samples_dir, "03_B21DCCN003_ThieuSang_NhieuHat.png"),
                        "Ảnh phiếu mẫu chuẩn (sample_sheet.png)": default_sheet_path,
                        "Tải ảnh từ máy tính": None,
                    }
                    img_mode = st.selectbox("Chọn ảnh kiểm thử:", list(m4_img_map.keys()), key="m4_img_mode")
                    if m4_img_map[img_mode] is None:
                        up_img = st.file_uploader("Nạp ảnh thật (.png, .jpg):", type=["png", "jpg", "jpeg"], key="m4_up_img")
                        if up_img is not None:
                            fbytes = np.asarray(bytearray(up_img.getvalue()), dtype=np.uint8)
                            test_matrix = cv2.imdecode(fbytes, cv2.IMREAD_GRAYSCALE)
                            source_name = up_img.name
                    else:
                        p_m4 = m4_img_map[img_mode]
                        test_matrix = read_image_unicode(p_m4 if os.path.exists(p_m4) else default_sheet_path, cv2.IMREAD_GRAYSCALE)
                        source_name = os.path.basename(p_m4 if os.path.exists(p_m4) else default_sheet_path)

                m4_algo = st.selectbox(
                    "Chọn thuật toán cải thiện:",
                    [
                        "1. Bộ lọc trung vị khử nhiễu muối tiêu (Median Filter 2D)",
                        "2. Cân bằng biểu đồ mức xám (Histogram Equalization)"
                    ],
                    key="m4_algo"
                )
                k_size = 3
                if "trung vị" in m4_algo or "Median" in m4_algo:
                    k_size = st.selectbox("Kích thước cửa sổ lọc (Kernel size):", [3, 5, 7], index=0, key="m4_ksize")

                btn_run_m4 = st.button("Thực Thi Thuật Toán", type="primary", use_container_width=True, key="btn_run_m4")

                if test_matrix is not None and btn_run_m4:
                    t0 = time.perf_counter()
                    used_core = "Scratch (NumPy)" if use_scratch else "OpenCV Baseline"
                    if "trung vị" in m4_algo or "Median" in m4_algo:
                        if use_scratch:
                            out_matrix = median_filter_scratch(test_matrix, kernel_size=k_size)
                        else:
                            out_matrix = cv2.medianBlur(test_matrix, k_size)
                    else:
                        if use_scratch:
                            out_matrix = histogram_equalization_scratch(test_matrix)
                        else:
                            out_matrix = cv2.equalizeHist(test_matrix)

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    st.session_state.tab2_m4_result = {
                        "in_matrix": test_matrix,
                        "out_matrix": out_matrix,
                        "algo": m4_algo,
                        "mode": used_core,
                        "elapsed_ms": elapsed_ms,
                        "source_name": source_name
                    }

                if st.session_state.tab2_m4_result is not None:
                    r4 = st.session_state.tab2_m4_result
                    st.divider()
                    st.metric("Thời gian thực thi", f"{r4['elapsed_ms']:.2f} ms", delta=r4.get("mode", ""))
                    base_fname = os.path.splitext(r4["source_name"])[0]
                    csv_buf = io.StringIO()
                    np.savetxt(csv_buf, r4["out_matrix"], fmt="%d", delimiter=",")
                    st.download_button(
                        label="Tải Ma Trận Kết Quả (.csv)",
                        data=csv_buf.getvalue(),
                        file_name=f"ket_qua_{base_fname}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="m4_dl_csv"
                    )
                    ok_png, png_buf = cv2.imencode(".png", r4["out_matrix"])
                    if ok_png:
                        st.download_button(
                            label="Tải Ảnh Kết Quả (.png)",
                            data=png_buf.tobytes(),
                            file_name=f"ket_qua_{base_fname}.png",
                            mime="image/png",
                            use_container_width=True,
                            key="m4_dl_png"
                        )

        with m4_col_view:
            with st.container(border=True):
                st.markdown("##### Không Gian Ma Trận Số Học & Biểu Đồ")
                if st.session_state.tab2_m4_result is not None:
                    r4 = st.session_state.tab2_m4_result
                    st.success(f"Đã thực thi thành công {r4['algo']} ({r4.get('mode', '')}) trên {r4['source_name']} trong {r4['elapsed_ms']:.2f} ms.")
                    c1, c2 = st.columns(2)
                    in_mat = r4["in_matrix"]
                    out_mat = r4["out_matrix"]
                    in_large = in_mat.shape[0] > 100 or in_mat.shape[1] > 100
                    out_large = out_mat.shape[0] > 100 or out_mat.shape[1] > 100
                    with c1:
                        st.markdown("###### Đầu Vào (Trước Xử Lý)")
                        st.image(in_mat, caption=f"Ảnh mức xám ({in_mat.shape[0]}x{in_mat.shape[1]})", use_container_width=True, channels="GRAY")
                        if in_large:
                            st.caption(f"Hiển thị lát cắt 30x30 góc trên-trái (Kích thước gốc: {in_mat.shape[0]}x{in_mat.shape[1]})")
                            st.dataframe(pd.DataFrame(in_mat[:30, :30]), use_container_width=True, height=220)
                        else:
                            st.dataframe(pd.DataFrame(in_mat), use_container_width=True, height=220)
                    with c2:
                        st.markdown("###### Kết Quả (Sau Xử Lý)")
                        st.image(out_mat, caption=f"Sau xử lý ({out_mat.shape[0]}x{out_mat.shape[1]})", use_container_width=True, channels="GRAY")
                        if out_large:
                            st.caption(f"Hiển thị lát cắt 30x30 góc trên-trái (Kích thước gốc: {out_mat.shape[0]}x{out_mat.shape[1]})")
                            st.dataframe(pd.DataFrame(out_mat[:30, :30]), use_container_width=True, height=220)
                        else:
                            st.dataframe(pd.DataFrame(out_mat), use_container_width=True, height=220)

                    if "Histogram" in r4["algo"]:
                        fig_h, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(8, 2.8), dpi=120)
                        apply_modern_chart_style(ax_h1, title="Histogram Trước Cân Bằng", xlabel="Mức xám [0-255]", ylabel="Tần suất")
                        apply_modern_chart_style(ax_h2, title="Histogram Sau Cân Bằng", xlabel="Mức xám [0-255]", ylabel="Tần suất")
                        ax_h1.hist(in_mat.ravel(), bins=64, range=(0, 256), color="#64748B", alpha=0.85)
                        ax_h2.hist(out_mat.ravel(), bins=64, range=(0, 256), color="#2563EB", alpha=0.85)
                        plt.tight_layout()
                        st.pyplot(fig_h)
                        plt.close(fig_h)
                else:
                    if test_matrix is not None:
                        st.markdown("###### Dữ Liệu Ma Trận Số Học Đầu Vào (Pixel Values):")
                        if test_matrix.shape[0] > 100 or test_matrix.shape[1] > 100:
                            st.caption(f"Hiển thị lát cắt 30x30 góc trên-trái (Kích thước gốc: {test_matrix.shape[0]}x{test_matrix.shape[1]})")
                            st.dataframe(pd.DataFrame(test_matrix[:30, :30]), use_container_width=True, height=260)
                        else:
                            st.dataframe(pd.DataFrame(test_matrix), use_container_width=True, height=260)
                        st.info("Bấm nút 'Thực Thi Thuật Toán' ở cột bên trái để bắt đầu chạy.")

    # Stepper Footer Navigation
    st.write("")
    nav_c1, _, nav_c3 = st.columns([1.5, 4, 1.5])
    with nav_c1:
        if cur_step > 0:
            if st.button(f"Quay Lại Bước {cur_step}", use_container_width=True, key=f"btn_prev_{cur_step}"):
                st.session_state.pipeline_step_idx = cur_step - 1
                st.rerun()
    with nav_c3:
        if cur_step < 3:
            if st.button(f"Tiếp Bước {cur_step + 2}", type="primary", use_container_width=True, key=f"btn_next_{cur_step}"):
                st.session_state.pipeline_step_idx = cur_step + 1
                st.rerun()
