"""Hệ thống Chấm bài Trắc nghiệm Tự động OMR và Đánh giá Thị giác Máy tính
Production Dashboard & Evaluation Platform.
Modularized UI Architecture (TV4).
"""

import os
import sys
import streamlit as st

# Setup paths and cross-module integration (TV1, TV2, TV3, TV4)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
TV4_SRC = os.path.join(BASE_DIR, "src")
TV1_SRC = os.path.join(ROOT_DIR, "OMR_System_TV1", "src")
TV2_DIR = os.path.join(ROOT_DIR, "OMR_System_TV2")
TV3_SRC = os.path.join(ROOT_DIR, "OMR_System_TV3", "src")

for p in [TV4_SRC, BASE_DIR, TV1_SRC, TV2_DIR, TV3_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tv4_enhancement_eval import OMRGrader
from ui.styles import inject_custom_styles
from ui.tab1_grading import render_tab1
from ui.tab2_stepper import render_tab2
from ui.tab3_benchmark import render_tab3
from ui.tab4_statistics import render_tab4

# Page configuration
st.set_page_config(
    page_title="Hệ Thống Chấm Bài OMR Tự Động",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS Stylesheet
inject_custom_styles()

# Common Directory Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
DEFAULT_KEY_PATH = os.path.join(DATA_DIR, "answer_key.csv")
DEFAULT_SHEET_PATH = os.path.join(DATA_DIR, "sample_sheet.png")
DEFAULT_CSV_15_PATH = os.path.join(DATA_DIR, "sample_matrix_15x15.csv")
DEFAULT_CSV_LOW_PATH = os.path.join(DATA_DIR, "sample_matrix_low_contrast.csv")
DEFAULT_CSV_60_PATH = os.path.join(DATA_DIR, "sample_matrix.csv")

# Initialize Session State
if "graded_batch_results" not in st.session_state:
    st.session_state.graded_batch_results = []
if "tab1_processed_items" not in st.session_state:
    st.session_state.tab1_processed_items = []
if "tab1_last_signature" not in st.session_state:
    st.session_state.tab1_last_signature = None
if "tab2_m1_result" not in st.session_state:
    st.session_state.tab2_m1_result = None
if "tab2_m2_result" not in st.session_state:
    st.session_state.tab2_m2_result = None
if "tab2_m3_result" not in st.session_state:
    st.session_state.tab2_m3_result = None
if "tab2_m4_result" not in st.session_state:
    st.session_state.tab2_m4_result = None
if "tab2_result" not in st.session_state:
    st.session_state.tab2_result = None
if "tab2_last_sig" not in st.session_state:
    st.session_state.tab2_last_sig = None
if "tab3_result" not in st.session_state:
    st.session_state.tab3_result = None
if "tab3_last_sig" not in st.session_state:
    st.session_state.tab3_last_sig = None
if "tab3_all_bm" not in st.session_state:
    st.session_state.tab3_all_bm = None
if "tab3_ablation" not in st.session_state:
    st.session_state.tab3_ablation = None
if "active_key_path" not in st.session_state:
    st.session_state.active_key_path = DEFAULT_KEY_PATH
if "pipeline_step_idx" not in st.session_state:
    st.session_state.pipeline_step_idx = 0

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="sidebar-logo-icon">OMR</div>
        <div>
            <div class="sidebar-brand-title">OMR Vision</div>
            <div class="sidebar-brand-sub">Hệ Thống Chấm Thi Thị Giác</div>
        </div>
    </div>
    <div class="sidebar-status-badge">
        <span class="status-dot-pulse"></span> Pipeline Sẵn Sàng
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group-title">Chế độ thuật toán</div>', unsafe_allow_html=True)
    mode_option = st.radio(
        "Chọn lõi tính toán:",
        ["Thuật toán tự cài đặt (NumPy)", "Thư viện OpenCV"],
        index=0,
        label_visibility="collapsed"
    )
    use_scratch = (mode_option == "Thuật toán tự cài đặt (NumPy)")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group-title">Tiền xử lý & Ngưỡng nhận diện</div>', unsafe_allow_html=True)
    apply_enhancement = st.checkbox("Bật tiền xử lý (Median + Equalization)", value=True)
    fill_thresh = st.slider(
        "Ngưỡng tô đậm (Fill Ratio):",
        0.15, 0.65, 0.38, 0.01,
        help="Tỷ lệ diện tích điểm đen tối thiểu trong ô để tính là đã tô. Mặc định 0.38 giúp loại trừ chính xác vết tẩy xóa."
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group-title">Đáp án chuẩn (Key)</div>', unsafe_allow_html=True)
    try:
        key_grader = OMRGrader(st.session_state.active_key_path)
        q_count = len(key_grader.answer_key)
    except Exception:
        q_count = 20

    is_custom_key = st.session_state.active_key_path != DEFAULT_KEY_PATH
    st.markdown(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; margin-bottom: 6px;">
        <div style="font-size: 11px; color: #64748B; font-weight: 500;">Quy mô bộ đáp án hiện tại:</div>
        <div style="font-size: 14.5px; font-weight: 700; color: #0F172A; margin-top: 2px;">
            {q_count} câu hỏi {'<span style="color:#2563EB; font-size:11px;">(Tùy biến)</span>' if is_custom_key else '<span style="color:#64748B; font-size:11px;">(Mặc định)</span>'}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Nạp & cấu hình file CSV đáp án chi tiết tại Tab Chấm Bài.")

# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="modern-header">
    <div class="header-breadcrumb">
        <span>HỆ THỐNG THỊ GIÁC MÁY TÍNH</span>
        <span class="breadcrumb-separator">/</span>
        <span>XỬ LÝ ẢNH SỐ</span>
        <span class="breadcrumb-separator">/</span>
        <span class="breadcrumb-active">OMR VISION</span>
    </div>
    <div class="header-main-row">
        <div>
            <h1 class="header-title">Hệ Thống Chấm Bài Trắc Nghiệm Tự Động OMR</h1>
            <p class="header-desc">Quy trình xử lý ảnh: Tiền xử lý • Nắn phối cảnh hình học SVD • Phân đoạn nhị phân & CCL • Chấm điểm & Đánh giá định lượng</p>
        </div>
        <div class="header-chip">
            <span class="chip-dot"></span>
            <span>4 Phân Hệ Hoàn Chỉnh</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Chấm Bài Trắc Nghiệm",
    "Kiểm Thử Thuật Toán Từng Phân Hệ",
    "Đối Chứng Thuật Toán vs OpenCV",
    "Thống Kê & Phổ Điểm"
])

with tab1:
    render_tab1(
        use_scratch=use_scratch,
        apply_enhancement=apply_enhancement,
        fill_thresh=fill_thresh,
        data_dir=DATA_DIR,
        samples_dir=SAMPLES_DIR,
        default_sheet_path=DEFAULT_SHEET_PATH,
        default_key_path=DEFAULT_KEY_PATH
    )

with tab2:
    render_tab2(
        use_scratch=use_scratch,
        samples_dir=SAMPLES_DIR,
        default_sheet_path=DEFAULT_SHEET_PATH,
        default_csv_15_path=DEFAULT_CSV_15_PATH,
        default_csv_low_path=DEFAULT_CSV_LOW_PATH,
        default_csv_60_path=DEFAULT_CSV_60_PATH
    )

with tab3:
    render_tab3(
        use_scratch=use_scratch,
        fill_thresh=fill_thresh,
        apply_enhancement=apply_enhancement,
        samples_dir=SAMPLES_DIR,
        default_sheet_path=DEFAULT_SHEET_PATH
    )

with tab4:
    render_tab4(
        samples_dir=SAMPLES_DIR,
        default_key_path=DEFAULT_KEY_PATH
    )
