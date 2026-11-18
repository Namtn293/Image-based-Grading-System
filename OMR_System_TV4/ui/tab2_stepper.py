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
        solve_homography_svd,
        warp_perspective_bilinear
    )
except ImportError:
    order_corners = None
    solve_homography_svd = None
    warp_perspective_bilinear = None

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


def render_tab2(
    use_scratch: bool,
    samples_dir: str,
    default_sheet_path: str,
    default_csv_15_path: str,
    default_csv_low_path: str,
    default_csv_60_path: str
):
    """Render Tab 2: 4-Stage Interactive Pipeline Stepper."""

    st.markdown("##### Kiểm Thử Thuật Toán Từng Phân Hệ (Pipeline Stepper)")

    step_items = [
        {"num": 1, "title": "1. Tiền Xử Lý & Khung", "sub": "Lọc Gauss, Canny & 4 Góc Khung"},
        {"num": 2, "title": "2. Nắn Phẳng SVD", "sub": "DLT & Ma Trận Homography 3x3"},
        {"num": 3, "title": "3. Phân Đoạn & CCL", "sub": "Otsu & Phân Tích Bong Bóng"},
        {"num": 4, "title": "4. Cải Thiện & CSV", "sub": "Lọc Median & Ma Trận Mức Xám CSV"},
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
                m1_src_opt = st.selectbox(
                    "Chọn ảnh đầu vào:",
                    [
                        "Phiếu chuẩn (01_B21DCCN001_Chuan.png)",
                        "Chụp nghiêng bàn tối (02_B21DCCN002_Nghieng_BanToi.png)",
                        "Thiếu sáng + Nhiễu hạt (03_B21DCCN003_ThieuSang_NhieuHat.png)",
                        "Bóng đổ chéo (05_B21DCCN005_BongDo_ChieuSangKhongDeu.png)",
                        "Tải ảnh từ máy tính"
                    ],
                    key="m1_src_opt"
                )
                m1_img_bgr = None
                if "Tải ảnh" in m1_src_opt:
                    up_m1 = st.file_uploader("Nạp ảnh (.png, .jpg):", type=["png", "jpg", "jpeg"], key="up_m1")
                    if up_m1 is not None:
                        fb = np.asarray(bytearray(up_m1.read()), dtype=np.uint8)
                        m1_img_bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                elif "Chụp nghiêng" in m1_src_opt:
                    p = os.path.join(samples_dir, "02_B21DCCN002_Nghieng_BanToi.png")
                    m1_img_bgr = cv2.imread(p if os.path.exists(p) else default_sheet_path)
                elif "Thiếu sáng" in m1_src_opt:
                    p = os.path.join(samples_dir, "03_B21DCCN003_ThieuSang_NhieuHat.png")
                    m1_img_bgr = cv2.imread(p if os.path.exists(p) else default_sheet_path)
                elif "Bóng đổ" in m1_src_opt:
                    p = os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
                    m1_img_bgr = cv2.imread(p if os.path.exists(p) else default_sheet_path)
                else:
                    p = os.path.join(samples_dir, "01_B21DCCN001_Chuan.png")
                    m1_img_bgr = cv2.imread(p if os.path.exists(p) else default_sheet_path)

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
                    gray = cv2.cvtColor(m1_img_bgr, cv2.COLOR_BGR2GRAY) if m1_img_bgr.ndim == 3 else m1_img_bgr
                    out_vis = None
                    extra_info = None

                    if "Grayscale" in m1_algo:
                        if rgb_to_gray_scratch is not None:
                            out_vis = rgb_to_gray_scratch(m1_img_bgr, is_bgr=True)
                        else:
                            out_vis = gray
                    elif "Gauss" in m1_algo:
                        if gaussian_blur_scratch is not None:
                            out_vis = gaussian_blur_scratch(gray, kernel_size=m1_params.get("ksize", 5), sigma=m1_params.get("sigma", 1.4))
                        else:
                            out_vis = cv2.GaussianBlur(gray, (m1_params.get("ksize", 5), m1_params.get("ksize", 5)), m1_params.get("sigma", 1.4))
                    elif "Canny" in m1_algo:
                        if canny_edge_scratch is not None:
                            out_vis = canny_edge_scratch(gray, low_thresh=m1_params.get("low", 50.0), high_thresh=m1_params.get("high", 150.0))
                        else:
                            out_vis = cv2.Canny(gray, int(m1_params.get("low", 50.0)), int(m1_params.get("high", 150.0)))
                    else:
                        if detect_sheet_corners is not None:
                            corners = detect_sheet_corners(gray, use_scratch=True)
                        else:
                            h, w = gray.shape[:2]
                            corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)

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
                        "elapsed_ms": elapsed_ms,
                        "extra": extra_info
                    }

                if st.session_state.tab2_m1_result is not None:
                    r1 = st.session_state.tab2_m1_result
                    st.divider()
                    st.metric("Thời gian thực thi", f"{r1['elapsed_ms']:.2f} ms")
                    ok_png, png_buf = cv2.imencode(".png", r1["out_vis"] if r1["out_vis"].ndim == 2 else cv2.cvtColor(r1["out_vis"], cv2.COLOR_RGB2BGR))
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
                    st.success(f"Đã thực thi {r1['algo']} trong {r1['elapsed_ms']:.2f} ms.")
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
    # BƯỚC 2: NẮN PHẲNG PHỐI CẢNH SVD
    # -----------------------------------------------------
    elif cur_step == 1:
        st.markdown("###### Giai Đoạn 2: Nắn Phẳng Phối Cảnh & Định Vị Marker (Homography SVD)")

        m2_col_ctrl, m2_col_view = st.columns([1.1, 1.9], gap="large")
        with m2_col_ctrl:
            with st.container(border=True):
                st.markdown("##### Cấu Hình Nắn Phẳng SVD")
                m2_src_opt = st.selectbox(
                    "Chọn ảnh kiểm thử nắn phẳng:",
                    [
                        "Ảnh chụp nghiêng trên bàn tối (02_B21DCCN002_Nghieng_BanToi.png - Khuyến nghị)",
                        "Ảnh phiếu chuẩn (01_B21DCCN001_Chuan.png)",
                        "Ảnh bóng đổ chéo (05_B21DCCN005_BongDo_ChieuSangKhongDeu.png)",
                        "Tải ảnh tùy chọn từ máy tính"
                    ],
                    key="m2_src_opt"
                )
                m2_img_bgr = None
                if "Tải ảnh" in m2_src_opt:
                    up_m2 = st.file_uploader("Nạp ảnh nghiêng (.png, .jpg):", type=["png", "jpg", "jpeg"], key="up_m2")
                    if up_m2 is not None:
                        fb = np.asarray(bytearray(up_m2.read()), dtype=np.uint8)
                        m2_img_bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                elif "Ảnh chuẩn" in m2_src_opt:
                    p = os.path.join(samples_dir, "01_B21DCCN001_Chuan.png")
                    m2_img_bgr = cv2.imread(p if os.path.exists(p) else default_sheet_path)
                elif "Ảnh bóng đổ" in m2_src_opt:
                    p = os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
                    m2_img_bgr = cv2.imread(p if os.path.exists(p) else default_sheet_path)
                else:
                    p = os.path.join(samples_dir, "02_B21DCCN002_Nghieng_BanToi.png")
                    m2_img_bgr = cv2.imread(p if os.path.exists(p) else default_sheet_path)

                target_size_label = st.selectbox(
                    "Kích thước chuẩn hóa (W x H):",
                    ["800 x 1000 (Chuẩn hệ thống OMR)", "600 x 800", "1000 x 1200"],
                    index=0,
                    key="m2_size"
                )
                w_tgt, h_tgt = (800, 1000) if "800 x 1000" in target_size_label else ((600, 800) if "600" in target_size_label else (1000, 1200))

                btn_run_m2 = st.button("Thực Thi Nắn Phẳng Phối Cảnh SVD", type="primary", use_container_width=True, key="btn_run_m2")

                if m2_img_bgr is not None and btn_run_m2:
                    t0 = time.perf_counter()
                    gray = cv2.cvtColor(m2_img_bgr, cv2.COLOR_BGR2GRAY) if m2_img_bgr.ndim == 3 else m2_img_bgr

                    if detect_sheet_corners is not None:
                        src_corners = detect_sheet_corners(gray, use_scratch=True)
                    else:
                        h_in, w_in = gray.shape[:2]
                        src_corners = np.array([[0, 0], [w_in - 1, 0], [w_in - 1, h_in - 1], [0, h_in - 1]], dtype=np.float32)

                    if order_corners is not None:
                        ordered_src, _ = order_corners(src_corners)
                    else:
                        ordered_src = src_corners.astype(np.float32)

                    dst_corners = np.array([
                        [0, 0],
                        [w_tgt - 1, 0],
                        [w_tgt - 1, h_tgt - 1],
                        [0, h_tgt - 1]
                    ], dtype=np.float32)

                    warped_out = None
                    try:
                        if solve_homography_svd is not None:
                            H_mat = solve_homography_svd(ordered_src, dst_corners)
                        else:
                            H_mat, _ = cv2.findHomography(ordered_src, dst_corners)

                        if H_mat is None:
                            raise ValueError("Không tìm được ma trận Homography từ 4 góc đã phát hiện.")

                        if warp_perspective_bilinear is not None:
                            warped_out = warp_perspective_bilinear(m2_img_bgr, H_mat, (w_tgt, h_tgt))
                        else:
                            warped_out = cv2.warpPerspective(m2_img_bgr, H_mat, (w_tgt, h_tgt))
                    except ValueError as exc:
                        st.error(f"Không thể nắn phẳng phối cảnh: {exc}")
                        st.session_state.tab2_m2_result = None

                if m2_img_bgr is not None and btn_run_m2 and warped_out is not None:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    vis_src = m2_img_bgr.copy() if m2_img_bgr.ndim == 3 else cv2.cvtColor(m2_img_bgr, cv2.COLOR_GRAY2BGR)
                    pts_poly = ordered_src.astype(np.int32)
                    cv2.polylines(vis_src, [pts_poly], isClosed=True, color=(0, 255, 0), thickness=3)
                    corner_names = ["TL", "TR", "BR", "BL"]
                    for pt, name in zip(ordered_src, corner_names):
                        cx, cy = int(pt[0]), int(pt[1])
                        cv2.circle(vis_src, (cx, cy), 10, (0, 0, 255), -1)
                        cv2.putText(vis_src, name, (cx + 12, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    st.session_state.tab2_m2_result = {
                        "vis_src": vis_src,
                        "warped": warped_out,
                        "H_mat": H_mat,
                        "src_corners": ordered_src,
                        "elapsed_ms": elapsed_ms,
                        "out_size": (w_tgt, h_tgt)
                    }

                if st.session_state.tab2_m2_result is not None:
                    r2 = st.session_state.tab2_m2_result
                    st.divider()
                    st.metric("Thời gian giải SVD & Nắn phẳng", f"{r2['elapsed_ms']:.2f} ms")
                    st.markdown("###### Ma Trận Homography $\\mathbf{H}_{3 \\times 3}$ (SVD):")
                    h_df = pd.DataFrame(r2["H_mat"], columns=["h_x", "h_y", "h_w"], index=["row 1", "row 2", "row 3"])
                    st.dataframe(h_df.style.format("{:.5f}"), use_container_width=True)
                    ok_png, png_buf = cv2.imencode(".png", r2["warped"] if r2["warped"].ndim == 2 else cv2.cvtColor(r2["warped"], cv2.COLOR_RGB2BGR))
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
                    st.success(f"Đã giải hệ ma trận SVD & nắn thẳng bài thi thành công trong {r2['elapsed_ms']:.2f} ms.")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("###### Ảnh Gốc (4 Đỉnh Nguồn SVD)")
                        st.image(cv2.cvtColor(r2["vis_src"], cv2.COLOR_BGR2RGB) if r2["vis_src"].ndim == 3 else r2["vis_src"], use_container_width=True)
                    with c2:
                        st.markdown(f"###### Ảnh Sau Nắn Phẳng ({r2['out_size'][0]}x{r2['out_size'][1]})")
                        st.image(cv2.cvtColor(r2["warped"], cv2.COLOR_BGR2RGB) if r2["warped"].ndim == 3 else r2["warped"], use_container_width=True)
                else:
                    if m2_img_bgr is not None:
                        st.image(cv2.cvtColor(m2_img_bgr, cv2.COLOR_BGR2RGB), caption=f"Ảnh chụp nghiêng hiện tại ({m2_img_bgr.shape[1]}x{m2_img_bgr.shape[0]})", use_container_width=True)
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
                m3_src_opt = st.selectbox(
                    "Chọn ảnh đầu vào:",
                    [
                        "Ảnh phiếu chuẩn nắn thẳng (sample_sheet.png)",
                        "Ảnh thiếu sáng + nhiễu hạt (03_B21DCCN003_ThieuSang_NhieuHat.png)",
                        "Ảnh bóng đổ chéo (05_B21DCCN005_BongDo_ChieuSangKhongDeu.png)",
                        "Ảnh tô sai quy chế & tẩy xóa (04_B21DCCN004_ToSaiQuyChe_TayXoa.png)",
                        "Tải ảnh tùy chọn từ máy tính"
                    ],
                    key="m3_src_opt"
                )
                m3_img = None
                if "Tải ảnh" in m3_src_opt:
                    up_m3 = st.file_uploader("Nạp ảnh (.png, .jpg):", type=["png", "jpg", "jpeg"], key="up_m3")
                    if up_m3 is not None:
                        fb = np.asarray(bytearray(up_m3.read()), dtype=np.uint8)
                        m3_img = cv2.imdecode(fb, cv2.IMREAD_GRAYSCALE)
                elif "thiếu sáng" in m3_src_opt:
                    p = os.path.join(samples_dir, "03_B21DCCN003_ThieuSang_NhieuHat.png")
                    m3_img = cv2.imread(p if os.path.exists(p) else default_sheet_path, cv2.IMREAD_GRAYSCALE)
                elif "bóng đổ" in m3_src_opt:
                    p = os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
                    m3_img = cv2.imread(p if os.path.exists(p) else default_sheet_path, cv2.IMREAD_GRAYSCALE)
                elif "tô sai quy chế" in m3_src_opt:
                    p = os.path.join(samples_dir, "04_B21DCCN004_ToSaiQuyChe_TayXoa.png")
                    m3_img = cv2.imread(p if os.path.exists(p) else default_sheet_path, cv2.IMREAD_GRAYSCALE)
                else:
                    m3_img = cv2.imread(default_sheet_path, cv2.IMREAD_GRAYSCALE)

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
                    m3_params["op"] = st.selectbox("Toán tử hình thái:", ["Mở (Opening - Khử nhiễu trắng)", "Đóng (Closing - Lấp đầy lỗ hổng)", "Co (Erosion)", "Giãn (Dilation)"], key="m3_op")
                elif "CCL" in m3_algo:
                    m3_params["connectivity"] = st.selectbox("Kiểu lân cận CCL:", [8, 4], index=0, key="m3_conn")

                btn_run_m3 = st.button("Thực Thi Phân Đoạn & CCL", type="primary", use_container_width=True, key="btn_run_m3")

                if m3_img is not None and btn_run_m3:
                    t0 = time.perf_counter()
                    out_bin = None
                    extra_data = None

                    if "Otsu" in m3_algo:
                        if otsu_threshold_scratch is not None:
                            t_val, out_bin = otsu_threshold_scratch(m3_img, method="BINARY_INV")
                        else:
                            t_val, out_bin = cv2.threshold(m3_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        extra_data = {"otsu_t": t_val}
                    elif "Adaptive" in m3_algo:
                        if adaptive_mean_threshold_scratch is not None:
                            out_bin = adaptive_mean_threshold_scratch(m3_img, block_size=m3_params.get("block_size", 51), C=m3_params.get("C", 42.0))
                        else:
                            out_bin = cv2.adaptiveThreshold(m3_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, int(m3_params.get("block_size", 51)), int(m3_params.get("C", 42.0)))
                    elif "hình thái" in m3_algo:
                        if otsu_threshold_scratch is not None:
                            _, bin_base = otsu_threshold_scratch(m3_img, method="BINARY_INV")
                        else:
                            _, bin_base = cv2.threshold(m3_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        op_choice = m3_params.get("op", "Mở")
                        if "Mở" in op_choice:
                            out_bin = opening_binary_scratch(bin_base) if opening_binary_scratch is not None else cv2.morphologyEx(bin_base, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                        elif "Đóng" in op_choice:
                            out_bin = closing_binary_scratch(bin_base) if closing_binary_scratch is not None else cv2.morphologyEx(bin_base, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                        elif "Co" in op_choice:
                            out_bin = morphology_binary_scratch(bin_base, op="erode") if morphology_binary_scratch is not None else cv2.erode(bin_base, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                        else:
                            out_bin = morphology_binary_scratch(bin_base, op="dilate") if morphology_binary_scratch is not None else cv2.dilate(bin_base, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                    else:
                        if otsu_threshold_scratch is not None:
                            _, bin_base = otsu_threshold_scratch(m3_img, method="BINARY_INV")
                        else:
                            _, bin_base = cv2.threshold(m3_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                        if two_pass_ccl_scratch is not None:
                            num_comp, labels, stats = two_pass_ccl_scratch(bin_base, connectivity=m3_params.get("connectivity", 8))
                        else:
                            num_comp, labels, cv_stats, centroids = cv2.connectedComponentsWithStats(bin_base)
                            num_comp -= 1
                            stats = []
                            for li in range(1, num_comp + 1):
                                stats.append({
                                    "label": li,
                                    "area": int(cv_stats[li, cv2.CC_STAT_AREA]),
                                    "bbox": (int(cv_stats[li, cv2.CC_STAT_LEFT]), int(cv_stats[li, cv2.CC_STAT_TOP]), int(cv_stats[li, cv2.CC_STAT_WIDTH]), int(cv_stats[li, cv2.CC_STAT_HEIGHT]))
                                })

                        np.random.seed(42)
                        colors = np.random.randint(50, 255, size=(num_comp + 1, 3), dtype=np.uint8)
                        colors[0] = [0, 0, 0]
                        colored_labels = colors[labels]
                        out_bin = colored_labels
                        extra_data = {"num_comp": num_comp, "stats": stats}

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    st.session_state.tab2_m3_result = {
                        "in_gray": m3_img,
                        "out_bin": out_bin,
                        "algo": m3_algo,
                        "elapsed_ms": elapsed_ms,
                        "extra": extra_data
                    }

                if st.session_state.tab2_m3_result is not None:
                    r3 = st.session_state.tab2_m3_result
                    st.divider()
                    st.metric("Thời gian thực thi", f"{r3['elapsed_ms']:.2f} ms")
                    ok_png, png_buf = cv2.imencode(".png", r3["out_bin"])
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
                    st.success(f"Đã thực thi thành công {r3['algo']} trong {r3['elapsed_ms']:.2f} ms.")
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
                    img_mode = st.selectbox(
                        "Chọn ảnh kiểm thử:",
                        [
                            "Ảnh mẫu thiếu sáng + nhiễu muối tiêu (03_B21DCCN003_ThieuSang_NhieuHat.png)",
                            "Ảnh phiếu mẫu chuẩn (sample_sheet.png)",
                            "Tải ảnh từ máy tính"
                        ],
                        key="m4_img_mode"
                    )
                    if "Ảnh mẫu thiếu sáng" in img_mode:
                        p3 = os.path.join(samples_dir, "03_B21DCCN003_ThieuSang_NhieuHat.png")
                        test_matrix = cv2.imread(p3 if os.path.exists(p3) else default_sheet_path, cv2.IMREAD_GRAYSCALE)
                        source_name = "03_B21DCCN003_ThieuSang_NhieuHat.png"
                    elif "Ảnh phiếu mẫu chuẩn" in img_mode:
                        test_matrix = cv2.imread(default_sheet_path, cv2.IMREAD_GRAYSCALE)
                        source_name = "sample_sheet.png"
                    else:
                        up_img = st.file_uploader("Nạp ảnh thật (.png, .jpg):", type=["png", "jpg", "jpeg"], key="m4_up_img")
                        if up_img is not None:
                            fbytes = np.asarray(bytearray(up_img.read()), dtype=np.uint8)
                            test_matrix = cv2.imdecode(fbytes, cv2.IMREAD_GRAYSCALE)
                            source_name = up_img.name

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
                    out_matrix = None
                    if "trung vị" in m4_algo or "Median" in m4_algo:
                        out_matrix = median_filter_scratch(test_matrix, kernel_size=k_size)
                    else:
                        out_matrix = histogram_equalization_scratch(test_matrix)

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    st.session_state.tab2_m4_result = {
                        "in_matrix": test_matrix,
                        "out_matrix": out_matrix,
                        "algo": m4_algo,
                        "elapsed_ms": elapsed_ms,
                        "source_name": source_name
                    }

                if st.session_state.tab2_m4_result is not None:
                    r4 = st.session_state.tab2_m4_result
                    st.divider()
                    st.metric("Thời gian thực thi", f"{r4['elapsed_ms']:.2f} ms")
                    base_fname = os.path.splitext(r4['source_name'])[0]
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
                    st.success(f"Đã thực thi thành công {r4['algo']} trên {r4['source_name']} trong {r4['elapsed_ms']:.2f} ms.")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("###### Đầu Vào (Trước Xử Lý)")
                        st.image(r4["in_matrix"], caption=f"Ảnh mức xám ({r4['in_matrix'].shape[0]}x{r4['in_matrix'].shape[1]})", use_container_width=True, channels="GRAY")
                        st.dataframe(pd.DataFrame(r4["in_matrix"]), use_container_width=True, height=220)
                    with c2:
                        st.markdown("###### Kết Quả (Sau Xử Lý)")
                        st.image(r4["out_matrix"], caption=f"Sau xử lý ({r4['out_matrix'].shape[0]}x{r4['out_matrix'].shape[1]})", use_container_width=True, channels="GRAY")
                        st.dataframe(pd.DataFrame(r4["out_matrix"]), use_container_width=True, height=220)
                else:
                    if test_matrix is not None:
                        st.markdown("###### Dữ Liệu Ma Trận Số Học Đầu Vào (Pixel Values):")
                        st.dataframe(pd.DataFrame(test_matrix), use_container_width=True, height=260)
                        st.info("Bấm nút 'Thực Thi Thuật Toán' ở cột bên trái để bắt đầu chạy.")

    # Stepper Footer Navigation
    st.write("")
    nav_c1, nav_c2, nav_c3 = st.columns([1.5, 4, 1.5])
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
