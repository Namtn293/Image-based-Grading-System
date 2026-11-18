"""Tab 1: Chấm Bài Trắc Nghiệm & Human-in-the-Loop Override.
Clean, professional UI without text emojis.
"""

import os
import io
import time
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from tv4_enhancement_eval import OMRGrader
from pipeline_coordinator import OMRPipeline


def render_tab1(
    use_scratch: bool,
    apply_enhancement: bool,
    fill_thresh: float,
    data_dir: str,
    samples_dir: str,
    default_sheet_path: str,
    default_key_path: str
):
    """Render Tab 1: Automated Grading and Review Workspace."""

    # 1. Quản Lý File Đáp Án Chuẩn (Answer Key CSV)
    with st.expander("Cấu Hình Bộ Đáp Án Chuẩn (Answer Key CSV)", expanded=False):
        ak_col1, ak_col2, ak_col3 = st.columns([2, 1.2, 1], vertical_alignment="bottom")
        with ak_col1:
            up_key = st.file_uploader("Nạp file đáp án chuẩn CSV mới:", type=["csv"], key="tab1_key_uploader")
            if up_key is not None:
                try:
                    raw_bytes = up_key.getvalue()
                    val_g = OMRGrader()
                    val_g.load_answer_key(io.StringIO(raw_bytes.decode("utf-8-sig")))
                    saved_path = os.path.join(data_dir, "uploaded_key.csv")
                    with open(saved_path, "wb") as f:
                        f.write(raw_bytes)
                    st.session_state.active_key_path = saved_path
                    st.success(f"Đã nạp thành công bộ đáp án: {len(val_g.answer_key)} câu hỏi.")
                except Exception as e:
                    st.error(f"File CSV không hợp lệ: {e}")
        with ak_col2:
            try:
                curr_grader = OMRGrader(st.session_state.active_key_path)
                curr_q_count = len(curr_grader.answer_key)
            except Exception:
                curr_q_count = 20
            st.markdown(f"**Quy mô hiện tại:** `{curr_q_count} câu hỏi`")
            if st.session_state.active_key_path != default_key_path:
                if st.button("Đặt Lại Mặc Định (20 câu)", use_container_width=True, key="reset_key_btn"):
                    st.session_state.active_key_path = default_key_path
                    st.rerun()
        with ak_col3:
            sample_csv_text = "question_id,correct_option\n" + "\n".join([f"{i},A" for i in range(1, 21)])
            st.download_button(
                "Tải Mẫu CSV",
                data=sample_csv_text,
                file_name="mau_dap_an_chuan.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_sample_key"
            )

    active_key_path = st.session_state.active_key_path

    # 2. Chọn Nguồn Dữ Liệu Phiếu Thi
    input_mode = st.radio(
        "Nguồn dữ liệu phiếu thi:",
        [
            "Chọn từ bộ 5 kịch bản mẫu (Chuẩn, Nghiêng bàn tối, Thiếu sáng, Tô sai quy chế, Bóng đổ)",
            "Tải ảnh phiếu thi từ máy tính (Hỗ trợ chấm 1 hoặc nhiều ảnh đồng thời)"
        ],
        horizontal=True
    )

    images_to_run: List[Tuple[str, np.ndarray]] = []
    trigger_run = False
    scen_val = None

    if "bộ 5 kịch bản mẫu" in input_mode:
        scenario_options = {
            "Kịch bản 1 (DS-01): Phiếu chuẩn (Mã SV: SV-01 — 17 đúng, 2 sai, 1 bỏ trống -> 8.5đ)": "01_B21DCCN001_Chuan.png",
            "Kịch bản 2 (DS-02): Chụp nghiêng trên mặt bàn tối (Mã SV: SV-02 -> 9.0đ)": "02_B21DCCN002_Nghieng_BanToi.png",
            "Kịch bản 3 (DS-04): Thiếu sáng + Nhiễu muối tiêu (Mã SV: SV-03 -> 8.0đ)": "03_B21DCCN003_ThieuSang_NhieuHat.png",
            "Kịch bản 4 (DS-05): Tô 2 đáp án phạm quy + Tẩy xóa (Mã SV: SV-04 -> 7.5đ)": "04_B21DCCN004_ToSaiQuyChe_TayXoa.png",
            "Kịch bản 5 (DS-03): Bóng đổ chéo & Chiếu sáng không đều (Mã SV: SV-05 -> 9.5đ)": "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png",
            "Chấm hàng loạt cả 5 kịch bản mẫu (Batch Processing 5 bài thi)": "ALL_SAMPLES"
        }

        col_scen, col_btn = st.columns([3, 1], vertical_alignment="bottom")
        with col_scen:
            selected_scen_label = st.selectbox(
                "Chọn kịch bản kiểm thử:",
                list(scenario_options.keys()),
                label_visibility="collapsed"
            )
        scen_val = scenario_options[selected_scen_label]
        current_tab1_sig = ("SAMPLE", scen_val, use_scratch, apply_enhancement, fill_thresh, active_key_path)

        with col_btn:
            btn_clicked = st.button("Bắt Đầu Chấm Điểm", type="primary", use_container_width=True)

        if btn_clicked or (st.session_state.tab1_processed_items and st.session_state.tab1_last_signature != current_tab1_sig):
            trigger_run = True
            st.session_state.tab1_last_signature = current_tab1_sig
            if scen_val == "ALL_SAMPLES":
                for fname in sorted(os.listdir(samples_dir)):
                    if fname.endswith(".png"):
                        raw_sid = fname.split("_")[1]
                        sid = raw_sid.replace("B21DCCN00", "SV-0") if "B21DCCN" in raw_sid else raw_sid
                        img = cv2.imread(os.path.join(samples_dir, fname))
                        if img is not None:
                            images_to_run.append((sid, img))
            else:
                fpath = os.path.join(samples_dir, scen_val)
                if not os.path.exists(fpath):
                    fpath = default_sheet_path
                raw_sid = scen_val.split("_")[1] if "_" in scen_val else "B21DCCN001"
                sid = raw_sid.replace("B21DCCN00", "SV-0") if "B21DCCN" in raw_sid else raw_sid
                img = cv2.imread(fpath)
                if img is not None:
                    images_to_run.append((sid, img))
    else:
        col_up, col_btn2 = st.columns([3, 1], vertical_alignment="bottom")
        with col_up:
            uploaded_sheets = st.file_uploader(
                "Tải lên 1 hoặc nhiều ảnh phiếu thi (.jpg, .png):",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
        up_names = [up.name for up in uploaded_sheets] if uploaded_sheets else []
        current_tab1_sig = ("UPLOAD", up_names, use_scratch, apply_enhancement, fill_thresh, active_key_path)

        with col_btn2:
            btn_up_clicked = st.button("Chấm Ảnh Đã Tải", type="primary", use_container_width=True)

        if btn_up_clicked or (uploaded_sheets and st.session_state.tab1_processed_items and st.session_state.tab1_last_signature != current_tab1_sig and st.session_state.tab1_last_signature and st.session_state.tab1_last_signature[0] == "UPLOAD"):
            if uploaded_sheets:
                trigger_run = True
                st.session_state.tab1_last_signature = current_tab1_sig
                for up in uploaded_sheets:
                    f_bytes = np.asarray(bytearray(up.getvalue()), dtype=np.uint8)
                    img = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
                    sid = os.path.splitext(up.name)[0].upper()
                    if img is not None:
                        images_to_run.append((sid, img))
            else:
                st.warning("Vui lòng chọn ít nhất 1 file ảnh trước khi thực hiện chấm điểm.")

    try:
        pipeline = OMRPipeline(active_key_path)
    except Exception as e:
        st.warning(f"Không thể khởi tạo bộ chấm thi với '{active_key_path}': {e}. Hệ thống sử dụng file mặc định.")
        pipeline = OMRPipeline(default_key_path)

    if trigger_run and images_to_run:
        results_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (sid, img_bgr) in enumerate(images_to_run):
            status_text.text(f"Đang xử lý bài thi {idx+1}/{len(images_to_run)}: {sid}...")
            try:
                res = pipeline.process_sheet(
                    image_bgr=img_bgr,
                    student_id=sid,
                    use_scratch_mode=use_scratch,
                    apply_enhancement=apply_enhancement,
                    fill_threshold=fill_thresh
                )
                res["raw_bgr"] = img_bgr
                results_list.append(res)
            except Exception as e:
                st.error(f"Lỗi khi xử lý bài thi {sid}: {e}")
            progress_bar.progress((idx + 1) / len(images_to_run))

        status_text.empty()
        progress_bar.empty()

        st.session_state.tab1_processed_items = results_list
        st.session_state.graded_batch_results = [r["grading_result"] for r in results_list]

    # Render persisted results OR clean Empty State
    if st.session_state.tab1_processed_items:
        results_list = st.session_state.tab1_processed_items

        if len(results_list) > 1:
            st.success(f"Đã hoàn tất chấm hàng loạt {len(results_list)} bài thi.")
            warnings_list = [r for r in results_list if r.get("warning")]
            if warnings_list:
                st.warning(f"Cảnh báo cấu trúc: Có {len(warnings_list)}/{len(results_list)} bài thi có số câu trên ảnh không khớp với file đáp án.")
            summary_rows = []
            for r in results_list:
                g = r["grading_result"]
                row_data = {
                    "Mã Sinh Viên": g["student_id"],
                    "Điểm Số (Hệ 10)": g["score_10"],
                    "Số Câu Đúng": g["correct_count"],
                    "Số Câu Sai": g["incorrect_count"],
                    "Bỏ Trống": g["blank_count"],
                    "Phạm Quy (>1 ô)": g["multiple_count"],
                    "Thời gian xử lý (ms)": f"{r['latencies']['total_pipeline_ms']:.1f}"
                }
                if r.get("warning"):
                    row_data["Cảnh báo"] = "Lệch số câu đề thi"
                summary_rows.append(row_data)
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            selected_idx = st.selectbox(
                "Chọn sinh viên để xem chi tiết bài thi & phúc khảo:",
                range(len(results_list)),
                format_func=lambda i: f"{results_list[i]['student_id']} — Điểm: {results_list[i]['grading_result']['score_10']}/10" + (" [Lệch đề]" if results_list[i].get("warning") else ""),
                key="tab1_selected_student_idx"
            )
            active_res = results_list[selected_idx]
        else:
            active_res = results_list[0]
            selected_idx = 0

        gr = active_res["grading_result"]

        if active_res.get("warning"):
            st.error(f"Cảnh báo cấu trúc: {active_res['warning']}")

        # 5 Metric Cards
        st.write("")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Điểm số (Hệ 10)", f"{gr['score_10']:.1f}", delta=f"{gr['correct_count']}/{gr['total_questions']} câu")
        m2.metric("Số câu đúng", gr["correct_count"])
        m3.metric("Số câu sai", gr["incorrect_count"])
        m4.metric("Bỏ trống", gr["blank_count"])
        m5.metric("Phạm quy (>1 ô)", gr["multiple_count"])

        # 3-Column Visual Progression Cards
        st.write("")
        img_c1, img_c2, img_c3 = st.columns(3)
        with img_c1:
            st.markdown("""
            <div class="stage-card">
                <div class="stage-header">
                    <span class="stage-title">1. Ảnh Gốc Đầu Vào</span>
                    <span class="stage-pill stage-pill-1">Input</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(cv2.cvtColor(active_res["raw_bgr"], cv2.COLOR_BGR2RGB), caption="Ảnh gốc trước khi xử lý", use_container_width=True)
        with img_c2:
            st.markdown("""
            <div class="stage-card">
                <div class="stage-header">
                    <span class="stage-title">2. Sau Nắn Phẳng & Lọc</span>
                    <span class="stage-pill stage-pill-2">Processed</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(active_res["warped_gray"], caption="Đã lọc nhiễu, cân bằng sáng & nắn phẳng", use_container_width=True, channels="GRAY")
        with img_c3:
            st.markdown("""
            <div class="stage-card">
                <div class="stage-header">
                    <span class="stage-title">3. Kết Quả Chấm Điểm</span>
                    <span class="stage-pill stage-pill-3">Annotated</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(
                cv2.cvtColor(active_res["annotated_image"], cv2.COLOR_BGR2RGB),
                caption="Xanh: Đúng | Đỏ: Sai | Vàng: Bỏ trống/Phạm quy",
                use_container_width=True
            )

        # Detailed Question & Fill Ratio Table
        st.write("")
        st.markdown("##### Bảng Chi Tiết Từng Câu & Tỷ Lệ Tô Đậm (Fill Ratio A, B, C, D)")
        ratios = active_res["fill_ratios"]
        detail_rows = []
        for d in gr["details"]:
            q = d["question"]
            r_q = ratios.get(q, {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})
            stt_str = d["status"]
            if d.get("is_erased"):
                stt_str = f"{d['status']} (Tẩy xóa Δτ)"
            detail_rows.append({
                "Câu": f"Câu {q:02d}",
                "SV Chọn": d["student_ans"],
                "Đáp Án Chuẩn": d["correct_ans"],
                "Trạng Thái": stt_str,
                "Kết Quả": "Đúng" if d["is_correct"] else "Sai / Lỗi",
                "Fill A": f"{r_q.get('A', 0.0):.3f}",
                "Fill B": f"{r_q.get('B', 0.0):.3f}",
                "Fill C": f"{r_q.get('C', 0.0):.3f}",
                "Fill D": f"{r_q.get('D', 0.0):.3f}",
                "Ghi Chú": d.get("warning", "")
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, height=260)

        # Human-in-the-loop manual review / override
        with st.expander("Kiểm duyệt & Phúc khảo thủ công (Human-in-the-Loop Override)"):
            st.caption("Nếu bài thi bị mờ hoặc tẩy xóa dẫn đến cảnh báo BLANK / MULTIPLE hoặc vết tẩy xóa Δτ, giám thị có thể xác nhận lại đáp án tại đây:")
            erased_qs = [d["question"] for d in gr["details"] if d.get("is_erased")]
            if erased_qs:
                st.info(f"Lưu ý: Câu {', '.join(str(q) for q in erased_qs)} có vết tẩy xóa đã được thuật toán tự động giải mã theo luật chênh lệch độ đậm Δτ ≥ 0.15.")
            ov_col1, ov_col2, ov_col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
            with ov_col1:
                q_to_edit = st.selectbox("Chọn câu hỏi cần chỉnh sửa:", list(range(1, gr["total_questions"] + 1)), key="tab1_override_q")
            curr_det = next((d for d in gr["details"] if d["question"] == q_to_edit), None)
            curr_choice = curr_det["student_ans"] if curr_det else "BLANK"
            opt_choices = ["A", "B", "C", "D", "BLANK", "MULTIPLE"]
            curr_idx = opt_choices.index(curr_choice) if curr_choice in opt_choices else 0
            with ov_col2:
                new_opt = st.selectbox(
                    f"Đáp án xác nhận lại (Hiện tại: {curr_choice}):",
                    opt_choices,
                    index=curr_idx,
                    key="tab1_override_opt"
                )
            with ov_col3:
                if st.button("Cập Nhật", use_container_width=True, key="tab1_override_btn"):
                    current_ans_map = {d["question"]: d["student_ans"] for d in gr["details"]}
                    current_ans_map[q_to_edit] = new_opt
                    updated_gr = pipeline.grader.grade_student(gr["student_id"], current_ans_map)
                    vis_base = cv2.cvtColor(active_res["warped_gray"], cv2.COLOR_GRAY2BGR)
                    updated_img = pipeline._annotate_sheet_results(vis_base, updated_gr, ratios, fill_thresh)
                    st.session_state.tab1_processed_items[selected_idx]["grading_result"] = updated_gr
                    st.session_state.tab1_processed_items[selected_idx]["annotated_image"] = updated_img
                    st.session_state.graded_batch_results = [
                        item["grading_result"] for item in st.session_state.tab1_processed_items
                    ]
                    st.rerun()

        with st.expander("Phân rã thời gian xử lý từng giai đoạn (Latency Breakdown)"):
            lat_df = pd.DataFrame([active_res["latencies"]]).T
            lat_df.columns = ["Thời gian (ms)"]
            st.table(lat_df)

        # Export Excel Report (in-memory buffer)
        excel_buf = io.BytesIO()
        OMRGrader.generate_excel_report(
            [r["grading_result"] for r in st.session_state.tab1_processed_items],
            excel_buf
        )
        excel_bytes = excel_buf.getvalue()

        st.download_button(
            label=f"Tải Bảng Điểm Excel ({len(st.session_state.tab1_processed_items)} Sinh Viên) (.xlsx)",
            data=excel_bytes,
            file_name="Ket_Qua_Cham_Thi_OMR.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        # Preview Card / Empty State before grading run
        if "bộ 5 kịch bản mẫu" in input_mode:
            if scen_val == "ALL_SAMPLES":
                with st.container(border=True):
                    st.markdown("""
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                        <div>
                            <h4 style="margin: 0; font-size: 16px; font-weight: 700; color: #0F172A;">Chế Độ Chấm Hàng Loạt (Batch Processing) 5 Sinh Viên Mẫu</h4>
                            <p style="margin: 2px 0 0 0; font-size: 13px; color: #64748B;">Hệ thống sẽ chạy liên tiếp 5 kịch bản kiểm thử thực tế và tổng hợp vào 1 bảng điểm chung.</p>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 14px;">
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px;">
                            <b style="color: #1E293B; font-size: 12.5px;">1. DS-01 (SV-01)</b>
                            <div style="color: #64748B; font-size: 12px; margin-top: 2px;">Phiếu chuẩn (8.5đ)</div>
                        </div>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px;">
                            <b style="color: #1E293B; font-size: 12.5px;">2. DS-02 (SV-02)</b>
                            <div style="color: #64748B; font-size: 12px; margin-top: 2px;">Nghiêng bàn tối (9.0đ)</div>
                        </div>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px;">
                            <b style="color: #1E293B; font-size: 12.5px;">3. DS-04 (SV-03)</b>
                            <div style="color: #64748B; font-size: 12px; margin-top: 2px;">Thiếu sáng + Nhiễu hạt (8.0đ)</div>
                        </div>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px;">
                            <b style="color: #1E293B; font-size: 12.5px;">4. DS-05 (SV-04)</b>
                            <div style="color: #64748B; font-size: 12px; margin-top: 2px;">Tô 2 ô + Tẩy xóa (7.5đ)</div>
                        </div>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px;">
                            <b style="color: #1E293B; font-size: 12.5px;">5. DS-03 (SV-05)</b>
                            <div style="color: #64748B; font-size: 12px; margin-top: 2px;">Bóng đổ chéo (9.5đ)</div>
                        </div>
                    </div>
                    <p style="margin: 16px 0 0 0; font-size: 12.5px; color: #2563EB; font-weight: 600;">
                        Hướng dẫn: Bấm nút <b>"Bắt Đầu Chấm Điểm"</b> ở trên để kích hoạt Pipeline xử lý toàn bộ bài thi.
                    </p>
                    """, unsafe_allow_html=True)
            else:
                fpath = os.path.join(samples_dir, scen_val) if scen_val else default_sheet_path
                if not os.path.exists(fpath):
                    fpath = default_sheet_path
                with st.container(border=True):
                    pv_c1, pv_c2 = st.columns([1, 2], vertical_alignment="center")
                    with pv_c1:
                        st.image(fpath, caption=f"Ảnh xem trước: {os.path.basename(fpath)}", use_container_width=True)
                    with pv_c2:
                        st.markdown(f"#### Xem Trước Phiếu Thi: `{os.path.basename(fpath)}`")
                        st.markdown("""
                        - **Quy chuẩn phiếu:** 20 câu trắc nghiệm, 4 đáp án (A, B, C, D).
                        - **Tọa độ định vị:** 4 điểm mốc fiducial ở 4 góc để nắn phẳng hình học qua ma trận SVD.
                        - **Trạng thái:** Phiếu thi đã được nạp sẵn, sẵn sàng đưa vào luồng chấm điểm.
                        """)
                        st.markdown("""
                        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px 14px; color: #1D4ED8; font-size: 13px; margin-top: 10px;">
                            Hướng dẫn: Bấm nút <b>Bắt Đầu Chấm Điểm</b> ở trên để hệ thống tiến hành nắn phẳng, phân đoạn nhị phân và chấm điểm.
                        </div>
                        """, unsafe_allow_html=True)
        else:
            with st.container(border=True):
                st.markdown("""
                <div style="text-align: center; padding: 24px 16px;">
                    <div style="font-size: 15px; font-weight: 700; color: #2563EB; margin-bottom: 8px;">TRẠNG THÁI CHỜ</div>
                    <h3 style="font-size: 17px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">Sẵn Sàng Nhận Diện & Chấm Điểm</h3>
                    <p style="color: #64748B; font-size: 13.5px; max-width: 540px; margin: 0 auto 16px auto;">
                        Vui lòng tải lên 1 hoặc nhiều ảnh chụp phiếu thi (.png, .jpg) ở ô phía trên, sau đó bấm <b>"Chấm Ảnh Đã Tải"</b>.
                    </p>
                    <div style="display: inline-flex; gap: 16px; text-align: left; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 16px; font-size: 12px; color: #475569;">
                        <div>• Định dạng hỗ trợ: PNG, JPG, JPEG</div>
                        <div>• Độ phân giải khuyến nghị ≥ 800×1000</div>
                        <div>• Đầy đủ 4 điểm mốc định vị góc</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
