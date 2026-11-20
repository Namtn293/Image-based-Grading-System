"""Tab 4: Thống Kê & Phổ Điểm Lớp Học.
Clean, academic interface without text emojis.
"""

import os
import io
from typing import List, Dict, Any
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

from tv4_enhancement_eval import OMRGrader
from pipeline_coordinator import OMRPipeline
from ui.styles import apply_modern_chart_style, read_image_unicode


@st.cache_data
def get_default_graded_batch(samples_dir: str, default_key_path: str, key_mtime: float = 0.0) -> List[Dict[str, Any]]:
    """Tải và chấm điểm sẵn 5 bài thi thực tế từ thư mục data/samples để hiển thị số liệu thực."""
    results = []
    p = OMRPipeline(default_key_path)
    if os.path.exists(samples_dir):
        for fname in sorted(os.listdir(samples_dir)):
            if fname.endswith(".png"):
                raw_sid = fname.split("_")[1] if "_" in fname else "B21DCCN001"
                sid = raw_sid.replace("B21DCCN00", "SV-0") if "B21DCCN" in raw_sid else raw_sid
                fpath = os.path.join(samples_dir, fname)
                img = read_image_unicode(fpath)
                if img is not None:
                    res = p.process_sheet(img, sid, use_scratch_mode=False)
                    results.append(res["grading_result"])
    return results


@st.cache_data
def generate_mock_graded_batch(num_students: int = 60, key_path: str = "", default_key_path: str = "") -> List[Dict[str, Any]]:
    """Tạo bộ dữ liệu mẫu giả lập (Mock Data) chuẩn hóa cho 60 sinh viên phục vụ kiểm thử trực quan phổ điểm."""
    grader = OMRGrader(key_path)
    if not grader.answer_key and default_key_path and os.path.exists(default_key_path):
        grader = OMRGrader(default_key_path)
    if not grader.answer_key:
        grader.set_answer_key_dict({i: "A" for i in range(1, 21)})
    ans_key = grader.answer_key
    questions = sorted(ans_key.keys())
    rng = np.random.default_rng(42)
    results = []
    options = ["A", "B", "C", "D"]
    q_difficulties = {q: 0.50 + 0.45 * (np.sin(q * 1.3) ** 2) for q in questions}
    for i in range(1, num_students + 1):
        sid = f"SV21DCCN{i:03d}"
        ability = float(rng.normal(0.2, 0.45))
        student_answers = {}
        for q in questions:
            corr_ans = ans_key[q]
            p_correct = float(np.clip(q_difficulties[q] + ability, 0.15, 0.98))
            r = float(rng.random())
            if r < p_correct:
                student_answers[q] = corr_ans
            elif r < p_correct + 0.03:
                student_answers[q] = "BLANK"
            elif r < p_correct + 0.05:
                student_answers[q] = "MULTIPLE"
            else:
                wrong_choices = [c for c in options if c != corr_ans]
                student_answers[q] = str(rng.choice(wrong_choices))
        res = grader.grade_student(sid, student_answers)
        results.append(res)
    return results


def render_tab4(samples_dir: str, default_key_path: str):
    """Render Tab 4: Class Statistics, Score Distribution and Proctor Oversight."""

    if "use_mock_data" not in st.session_state:
        st.session_state.use_mock_data = False

    actual_batch = st.session_state.graded_batch_results

    # State Banner with Mock Data Toggle
    banner_c1, banner_c2 = st.columns([3.8, 1.4], vertical_alignment="center")

    if st.session_state.use_mock_data:
        with banner_c1:
            st.info("Đang hiển thị Dữ Liệu Mẫu Giả Lập (Mock Data 60 sinh viên) — Phân phối điểm chuẩn (Bell Curve) từ 1.0 đến 10.0 điểm.")
        with banner_c2:
            if st.button("Khôi Phục Dữ Liệu Gốc", type="secondary", use_container_width=True, key="btn_reset_mock"):
                st.session_state.use_mock_data = False
                st.rerun()
        batch = generate_mock_graded_batch(60, st.session_state.active_key_path, default_key_path)
    else:
        if not actual_batch:
            with banner_c1:
                st.info("Hiển thị thống kê từ 5 bài thi mẫu (SV-01 đến SV-05) có sẵn trong hệ thống. Khi thực hiện chấm bài mới tại Tab 1, biểu đồ sẽ tự động cập nhật theo kết quả vừa chấm.")
            with banner_c2:
                if st.button("Nạp Dữ Liệu Mẫu (Mock Data)", type="primary", use_container_width=True, key="btn_load_mock_empty"):
                    st.session_state.use_mock_data = True
                    st.rerun()
            key_mtime = os.path.getmtime(default_key_path) if os.path.exists(default_key_path) else 0.0
            batch = get_default_graded_batch(samples_dir, default_key_path, key_mtime)
        else:
            with banner_c1:
                st.success(f"Đang hiển thị thống kê thực tế từ {len(actual_batch)} bài thi vừa chấm tại Tab 1.")
            with banner_c2:
                if st.button("Nạp Dữ Liệu Mẫu (Mock Data)", type="secondary", use_container_width=True, key="btn_load_mock_actual"):
                    st.session_state.use_mock_data = True
                    st.rerun()
            batch = actual_batch

    total_st = len(batch)
    if total_st == 0:
        st.warning("Chưa có dữ liệu bài thi để thống kê. Vui lòng chấm bài tại Tab 1 hoặc bật Mock Data.")
        return

    sample_scores = np.array([r["score_10"] for r in batch])
    q_keys = sorted({d["question"] for r in batch for d in r["details"]})
    q_labels = [f"Q{qk:02d}" for qk in q_keys]
    q_rates = []
    for qk in q_keys:
        cor = sum(1 for r in batch if any(d["question"] == qk and d["is_correct"] for d in r["details"]))
        q_rates.append(round((cor / total_st) * 100.0, 1))

    # KPI Summary Cards
    k1, k2, k3, k4 = st.columns(4)
    avg_score = float(np.mean(sample_scores))
    max_score = float(np.max(sample_scores))
    min_score = float(np.min(sample_scores))
    pass_count = int(np.sum(sample_scores >= 5.0))
    pass_rate = round((pass_count / max(total_st, 1)) * 100.0, 1)

    k1.metric("Tổng Số Bài Thi", f"{total_st} bài")
    k2.metric("Điểm Trung Bình", f"{avg_score:.2f} / 10.0")
    k3.metric("Điểm Cao Nhất / Thấp Nhất", f"{max_score:.1f} / {min_score:.1f}")

    # Standardized KPI Card Logic (Pass Rate)
    if total_st == 0:
        k4.metric(
            "Tỷ Lệ Đạt (>= 5.0)",
            "0.0%",
            delta="Chưa có dữ liệu",
            delta_color="off",
            delta_arrow="off"
        )
    elif pass_rate < 50.0:
        k4.metric(
            "Tỷ Lệ Đạt (>= 5.0)",
            f"{pass_rate}%",
            delta=f"{pass_count}/{total_st} sinh viên (<50%)",
            delta_color="red",
            delta_arrow="down"
        )
    else:
        k4.metric(
            "Tỷ Lệ Đạt (>= 5.0)",
            f"{pass_rate}%",
            delta=f"{pass_count}/{total_st} sinh viên (≥50%)",
            delta_color="green",
            delta_arrow="up"
        )

    st.write("")
    sc1, sc2 = st.columns([1, 1])

    with sc1:
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=130)
        apply_modern_chart_style(
            ax,
            title="Biểu đồ Phân Bố Điểm Số (Score Distribution)",
            xlabel="Điểm Số (Thang 10)",
            ylabel="Số Lượng Sinh Viên"
        )
        n, bins, patches = ax.hist(sample_scores, bins=np.arange(0, 11, 1), color="#3B82F6", edgecolor="#FFFFFF", linewidth=1.2, alpha=0.9, rwidth=0.9)
        ax.set_xlim(0, 10.5)
        ax.set_xticks(range(0, 11))
        ax.axvline(avg_score, color="#EF4444", linestyle="--", linewidth=1.8, label=f"Điểm TB: {avg_score:.2f}")
        ax.legend(frameon=False, loc="upper left", fontsize=9)

        # Fix Y-axis to Pure Integers with step_size = 1
        max_freq = int(max(n)) if len(n) > 0 and max(n) > 0 else 1
        ax.set_ylim(bottom=0, top=max_freq + (1 if max_freq < 8 else max(1, int(max_freq * 0.15))))
        if max_freq <= 8:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        st.pyplot(fig)
        plt.close(fig)

    with sc2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.2), dpi=130)
        apply_modern_chart_style(
            ax2,
            title="Tỷ Lệ Trả Lời Đúng Theo Từng Câu (%)",
            xlabel="Câu Hỏi",
            ylabel="Tỷ Lệ Trả Lời Đúng (%)",
            title_pad=22
        )
        bar_colors = []
        for rate in q_rates:
            if rate < 60.0:
                bar_colors.append("#EF4444")
            elif rate < 75.0:
                bar_colors.append("#F59E0B")
            else:
                bar_colors.append("#10B981")
        bars = ax2.bar(range(len(q_labels)), q_rates, color=bar_colors, width=0.65, edgecolor="none")
        ax2.set_ylim(0, 115)
        ax2.set_xticks(range(len(q_labels)))
        ax2.set_xticklabels(q_labels, rotation=45, fontsize=8)
        legend_elements = [
            Patch(facecolor='#10B981', label='Dễ / Đạt (≥75%)'),
            Patch(facecolor='#F59E0B', label='Trung bình (60-74%)'),
            Patch(facecolor='#EF4444', label='Khó (<60%)')
        ]
        ax2.legend(
            handles=legend_elements,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.07),
            ncol=3,
            fontsize=8.2
        )
        st.pyplot(fig2)
        plt.close(fig2)

    # Anomaly & Proctor Warning Attention List (Kế hoạch mục 9)
    anomaly_sheets = []
    for r in batch:
        sid = r.get("student_id", "N/A")
        det = r.get("details", [])
        erased_qs = [d["question"] for d in det if d.get("is_erased")]
        blank_qs = [d["question"] for d in det if d.get("status") == "BLANK"]
        multi_qs = [d["question"] for d in det if d.get("status") == "MULTIPLE"]
        sheet_w = r.get("sheet_warning")
        fb_w = r.get("fallback_warning")
        orient_w = r.get("orientation_warning")
        warn = r.get("warning")

        if erased_qs or multi_qs or blank_qs or warn or sheet_w or fb_w or orient_w:
            reasons = []
            if sheet_w:
                reasons.append(sheet_w)
            if fb_w:
                reasons.append(fb_w)
            if orient_w:
                reasons.append(orient_w)
            if warn and not sheet_w and not fb_w and not orient_w:
                reasons.append(warn)
            if erased_qs:
                reasons.append(f"Vết tẩy xóa (Câu: {', '.join(str(q) for q in erased_qs)})")
            if multi_qs:
                reasons.append(f"Tô đúp / phạm quy (Câu: {', '.join(str(q) for q in multi_qs)})")
            if blank_qs:
                reasons.append(f"Bỏ trống (Câu: {', '.join(str(q) for q in blank_qs)})")

            anomaly_sheets.append({
                "Mã Sinh Viên": sid,
                "Điểm Số": f"{r.get('score_10', 0.0):.1f}",
                "Số Câu Đúng": f"{r.get('correct_count', 0)}/{r.get('total_questions', 20)}",
                "Chi Tiết Cảnh Báo": " | ".join(reasons)
            })

    if anomaly_sheets:
        st.write("")
        with st.expander(f"Danh Sách Phiếu Cần Giám Thị Kiểm Duyệt Lại ({len(anomaly_sheets)} bài có cảnh báo)", expanded=False):
            st.caption("Các bài thi có vết tẩy xóa, tô đúp, bỏ trống hoặc cảnh báo lệch đề thi cần giám thị đối chiếu bản cứng:")
            st.dataframe(pd.DataFrame(anomaly_sheets), use_container_width=True)

    st.write("")
    # Export Excel Report (in-memory buffer)
    excel_buf = io.BytesIO()
    OMRGrader.generate_excel_report(batch, excel_buf)
    excel_bytes = excel_buf.getvalue()

    data_source_tag = "Mock 60 Sinh Viên" if st.session_state.use_mock_data else f"{len(batch)} Sinh Viên"
    st.download_button(
        label=f"Tải Báo Cáo Phổ Điểm & Bảng Điểm ({data_source_tag}) (.xlsx)",
        data=excel_bytes,
        file_name="Bao_Cao_Thong_Ke_OMR.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
