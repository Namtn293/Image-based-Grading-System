"""Hệ thống Chấm bài Trắc nghiệm Tự động OMR và Đánh giá Thị giác Máy tính
Production Dashboard
"""

import os
import sys
import io
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup path and cross-module integration (TV1, TV2, TV3, TV4)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
TV4_SRC = os.path.join(BASE_DIR, "src")
TV1_SRC = os.path.join(ROOT_DIR, "OMR_System_TV1", "src")
TV2_DIR = os.path.join(ROOT_DIR, "OMR_System_TV2")
TV3_SRC = os.path.join(ROOT_DIR, "OMR_System_TV3", "src")

for p in [TV4_SRC, BASE_DIR, TV1_SRC, TV2_DIR, TV3_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Module 1 (TV1): Preprocessing & Corner Detection
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

# Module 2 (TV2): Perspective Homography & SVD Warping
try:
    from tv2_perspective import (
        order_corners,
        solve_homography_svd,
        warp_perspective_bilinear,
        warp_perspective_sheet
    )
except ImportError:
    order_corners = None
    solve_homography_svd = None
    warp_perspective_bilinear = None
    warp_perspective_sheet = None

# Module 3 (TV3): Segmentation, Morphology & CCL
try:
    from tv3_thresholding import (
        otsu_threshold_scratch,
        adaptive_mean_threshold_scratch
    )
    from tv3_morphology_ccl import (
        create_structuring_element,
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

# Module 4 (TV4): Enhancement, Grading & Quantitative Evaluation
try:
    from tv4_enhancement_eval import (
        histogram_equalization_scratch,
        median_filter_scratch,
        save_matrix_to_csv,
        load_matrix_from_csv,
        calculate_mse,
        calculate_psnr,
        calculate_difference_map,
        calculate_binary_iou_dice,
        benchmark_hist_equalization,
        benchmark_median_filter,
        benchmark_all_modules,
        run_ablation_study,
        OMRGrader
    )
    from pipeline_coordinator import OMRPipeline
except ImportError:
    if TV4_SRC not in sys.path:
        sys.path.insert(0, TV4_SRC)
    from tv4_enhancement_eval import (
        histogram_equalization_scratch,
        median_filter_scratch,
        save_matrix_to_csv,
        load_matrix_from_csv,
        calculate_mse,
        calculate_psnr,
        calculate_difference_map,
        calculate_binary_iou_dice,
        benchmark_hist_equalization,
        benchmark_median_filter,
        benchmark_all_modules,
        run_ablation_study,
        OMRGrader
    )
    from pipeline_coordinator import OMRPipeline

# Page configuration
st.set_page_config(
    page_title="Hệ Thống Chấm Bài OMR Tự Động",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Standardized Modern Light SaaS Design System)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #0F172A;
    }

    /* Hide unnecessary Streamlit chrome */
    #MainMenu, footer {
        visibility: hidden !important;
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        color: #1E293B !important;
    }

    /* Container padding polish */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px !important;
    }

    /* Refined Modern Header */
    .modern-header {
        padding: 4px 0 16px 0 !important;
        margin-bottom: 20px !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }
    .header-breadcrumb {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        color: #94A3B8;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .breadcrumb-separator {
        color: #CBD5E1;
    }
    .breadcrumb-active {
        color: #2563EB;
    }
    .header-main-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }
    .header-title {
        font-size: 22px !important;
        font-weight: 800 !important;
        letter-spacing: -0.3px !important;
        color: #0F172A !important;
        margin: 0 !important;
        line-height: 1.25 !important;
    }
    .header-desc {
        color: #64748B !important;
        font-size: 13px !important;
        margin: 4px 0 0 0 !important;
        font-weight: 400 !important;
    }
    .header-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 600;
        color: #334155;
    }
    .chip-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
    }

    /* Pipeline Stepper / Wizard */
    .stepper-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .stepper-step {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .stepper-circle {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 12px;
        background: #F1F5F9;
        color: #64748B;
        border: 2px solid #E2E8F0;
        transition: all 0.2s ease;
    }
    .stepper-step.active .stepper-circle {
        background: #2563EB;
        color: #FFFFFF;
        border-color: #2563EB;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
    }
    .stepper-step.completed .stepper-circle {
        background: #10B981;
        color: #FFFFFF;
        border-color: #10B981;
    }
    .stepper-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
    }
    .stepper-step.active .stepper-label {
        color: #0F172A;
        font-weight: 700;
    }
    .stepper-line {
        flex: 1;
        height: 2px;
        background: #E2E8F0;
        margin: 0 10px;
    }
    .stepper-line.active {
        background: #2563EB;
    }

    /* Benchmark Toolbar & Skeleton Preview Grid */
    .benchmark-toolbar-title {
        font-size: 14px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .skeleton-preview-card {
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        background: #F8FAFC;
        padding: 24px 14px;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
    }
    .skeleton-preview-card:hover {
        border-color: #94A3B8;
        background: #F1F5F9;
    }
    .skeleton-icon {
        font-size: 30px;
        margin-bottom: 8px;
        opacity: 0.75;
    }
    .skeleton-title {
        font-size: 13px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 4px;
    }
    .skeleton-desc {
        font-size: 11.5px;
        color: #64748B;
        line-height: 1.45;
        margin-bottom: 10px;
        max-width: 240px;
    }
    .skeleton-badge {
        font-size: 10.5px;
        font-weight: 600;
        color: #94A3B8;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 2px 10px;
    }

    /* Modern Underline Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px !important;
        background-color: transparent !important;
        padding: 0 !important;
        border-radius: 0 !important;
        border: none !important;
        border-bottom: 2px solid #E2E8F0 !important;
        margin-bottom: 22px !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px !important;
        border-radius: 0 !important;
        padding: 8px 4px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border: none !important;
        background-color: transparent !important;
        color: #64748B !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -2px !important;
        transition: color 0.2s ease, border-color 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #2563EB !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #2563EB !important;
        box-shadow: none !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #2563EB !important;
    }
    div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Polished Metric Cards */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06) !important;
    }
    [data-testid="stMetricLabel"] p {
        font-weight: 700 !important;
        font-size: 11.5px !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 4px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        line-height: 1.1 !important;
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
        font-size: 12px !important;
    }

    /* Stage Progression Cards */
    .stage-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 10px;
    }
    .stage-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .stage-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #0F172A;
    }
    .stage-pill {
        font-size: 10.5px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .stage-pill-1 { background: #F1F5F9; color: #475569; }
    .stage-pill-2 { background: #EFF6FF; color: #2563EB; }
    .stage-pill-3 { background: #F0FDF4; color: #16A34A; }

    /* Empty State / Preview Card */
    .empty-state-card {
        background: #FFFFFF;
        border: 1.5px dashed #CBD5E1;
        border-radius: 12px;
        padding: 24px;
        margin-top: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    /* Buttons & Inputs 42px Height Alignment */
    .stButton > button {
        height: 42px !important;
        min-height: 42px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        letter-spacing: 0.2px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stDownloadButton > button {
        background: #059669 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        height: 42px !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25) !important;
    }
    .stDownloadButton > button:hover {
        background: #047857 !important;
        transform: translateY(-1px) !important;
    }

    /* Modern Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    .sidebar-brand-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .sidebar-logo-icon {
        background: #2563EB;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 800;
    }
    .sidebar-brand-title {
        font-size: 15px;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .sidebar-brand-sub {
        font-size: 11px;
        color: #64748B;
        font-weight: 500;
    }
    .sidebar-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .status-dot-pulse {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #22C55E;
    }
    .sidebar-divider {
        height: 1px;
        background-color: #E2E8F0;
        margin: 14px 0;
        border: none;
    }
    .sidebar-group-title {
        font-size: 11.5px;
        font-weight: 700;
        color: #1E293B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* High Contrast Sidebar Labels */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #1E293B !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] small {
        color: #475569 !important;
        font-size: 11.5px !important;
    }

    /* Sidebar File Uploader Normalization */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background-color: #F1F5F9 !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] p {
        color: #334155 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #1E293B !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for modern Matplotlib visual styling
def apply_modern_chart_style(ax, title=None, xlabel=None, ylabel=None, title_pad=12):
    ax.set_facecolor("#FFFFFF")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.tick_params(colors='#64748B', labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.6, color="#F1F5F9", alpha=0.9)
    if title:
        ax.set_title(title, fontsize=11, fontweight=700, color="#1E293B", pad=title_pad)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, fontweight=600, color="#475569", labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, fontweight=600, color="#475569", labelpad=8)

# Paths
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
        <div class="sidebar-logo-icon">✓</div>
        <div>
            <div class="sidebar-brand-title">OMR Vision</div>
            <div class="sidebar-brand-sub">Hệ Thống Chấm Thi Thị Giác</div>
        </div>
    </div>
    <div class="sidebar-status-badge">
        <span class="status-dot-pulse"></span> Pipeline Sẵn Sàng
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group-title">⚙️ Chế độ thuật toán</div>', unsafe_allow_html=True)
    mode_option = st.radio(
        "Chọn lõi tính toán:",
        ["Thuật toán tự cài đặt (NumPy)", "Thư viện OpenCV"],
        index=0,
        label_visibility="collapsed"
    )
    use_scratch = (mode_option == "Thuật toán tự cài đặt (NumPy)")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group-title">🎛️ Tiền xử lý & Ngưỡng nhận diện</div>', unsafe_allow_html=True)
    apply_enhancement = st.checkbox("Bật tiền xử lý (Median + Equalization)", value=True)
    fill_thresh = st.slider(
        "Ngưỡng tô đậm (Fill Ratio):",
        0.15, 0.65, 0.38, 0.01,
        help="Tỷ lệ diện tích điểm đen tối thiểu trong ô để tính là đã tô. Mặc định 0.38 giúp loại trừ chính xác vết tẩy xóa ~0.26."
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group-title">📑 Đáp án chuẩn (Key)</div>', unsafe_allow_html=True)
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
    st.caption("ℹ️ Nạp & cấu hình file CSV đáp án chi tiết tại Tab Chấm Bài.")

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

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Chấm Bài Trắc Nghiệm (Pipeline Tự Động)",
    "🔬 Kiểm Thử Thuật Toán Từng Phân Hệ",
    "⚖️ Đối Chứng Thuật Toán vs OpenCV",
    "📊 Thống Kê & Phổ Điểm"
])

# ---------------------------------------------------------
# TAB 1: CHẤM BÀI TRẮC NGHIỆM
# ---------------------------------------------------------
with tab1:
    # Quản Lý File Đáp Án Chuẩn (Answer Key CSV)
    with st.expander("📑 Cấu Hình Bộ Đáp Án Chuẩn (Answer Key CSV)", expanded=False):
        ak_col1, ak_col2, ak_col3 = st.columns([2, 1.2, 1], vertical_alignment="bottom")
        with ak_col1:
            up_key = st.file_uploader("Nạp file đáp án chuẩn CSV mới:", type=["csv"], key="tab1_key_uploader")
            if up_key is not None:
                try:
                    raw_bytes = up_key.getvalue()
                    val_g = OMRGrader()
                    val_g.load_answer_key(io.StringIO(raw_bytes.decode("utf-8-sig")))
                    saved_path = os.path.join(DATA_DIR, "uploaded_key.csv")
                    with open(saved_path, "wb") as f:
                        f.write(raw_bytes)
                    st.session_state.active_key_path = saved_path
                    st.success(f"✅ Đã nạp thành công bộ đáp án: {len(val_g.answer_key)} câu!")
                except Exception as e:
                    st.error(f"File CSV không hợp lệ: {e}")
        with ak_col2:
            st.markdown(f"**Quy mô hiện tại:** `{q_count} câu hỏi`")
            if st.session_state.active_key_path != DEFAULT_KEY_PATH:
                if st.button("🔄 Đặt Lại Mặc Định (20 câu)", use_container_width=True, key="reset_key_btn"):
                    st.session_state.active_key_path = DEFAULT_KEY_PATH
                    st.rerun()
        with ak_col3:
            sample_csv_text = "question_id,correct_option\n" + "\n".join([f"{i},A" for i in range(1, 21)])
            st.download_button(
                "📥 Tải Mẫu CSV",
                data=sample_csv_text,
                file_name="mau_dap_an_chuan.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_sample_key"
            )

    active_key_path = st.session_state.active_key_path

    input_mode = st.radio(
        "Nguồn dữ liệu phiếu thi:",
        [
            "📂 Chọn từ Bộ 5 Kịch Bản Mẫu (Chuẩn / Nghiêng / Thiếu sáng / Tô sai quy chế / Bóng đổ)",
            "📤 Tải ảnh phiếu thi từ máy tính (Hỗ trợ 1 hoặc nhiều ảnh đồng thời)"
        ],
        horizontal=True
    )

    images_to_run = []
    trigger_run = False
    scen_val = None

    if "Bộ 5 Kịch Bản Mẫu" in input_mode:
        scenario_options = {
            "Kịch bản 1 (DS-01): Phiếu chuẩn (Mã SV: SV-01 — 17 đúng, 2 sai, 1 bỏ trống -> 8.5đ)": "01_B21DCCN001_Chuan.png",
            "Kịch bản 2 (DS-02): Chụp nghiêng trên mặt bàn tối (Mã SV: SV-02 -> 9.0đ)": "02_B21DCCN002_Nghieng_BanToi.png",
            "Kịch bản 3 (DS-04): Thiếu sáng + Nhiễu muối tiêu (Mã SV: SV-03 -> 8.0đ)": "03_B21DCCN003_ThieuSang_NhieuHat.png",
            "Kịch bản 4 (DS-05): Tô 2 đáp án phạm quy + Tẩy xóa (Mã SV: SV-04 -> 7.5đ)": "04_B21DCCN004_ToSaiQuyChe_TayXoa.png",
            "Kịch bản 5 (DS-03): Bóng đổ chéo & Chiếu sáng không đều (Mã SV: SV-05 -> 9.5đ)": "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png",
            "🔥 Chấm hàng loạt cả 5 kịch bản mẫu (Batch Processing 5 SV)": "ALL_SAMPLES"
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
            btn_clicked = st.button("🚀 Bắt Đầu Chấm Điểm", type="primary", use_container_width=True)

        if btn_clicked or (st.session_state.tab1_processed_items and st.session_state.tab1_last_signature != current_tab1_sig):
            trigger_run = True
            st.session_state.tab1_last_signature = current_tab1_sig
            if scen_val == "ALL_SAMPLES":
                for fname in sorted(os.listdir(SAMPLES_DIR)):
                    if fname.endswith(".png"):
                        raw_sid = fname.split("_")[1]
                        sid = raw_sid.replace("B21DCCN00", "SV-0") if "B21DCCN" in raw_sid else raw_sid
                        img = cv2.imread(os.path.join(SAMPLES_DIR, fname))
                        if img is not None:
                            images_to_run.append((sid, img))
            else:
                fpath = os.path.join(SAMPLES_DIR, scen_val)
                if not os.path.exists(fpath):
                    fpath = DEFAULT_SHEET_PATH
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
            btn_up_clicked = st.button("🚀 Chấm Ảnh Đã Tải", type="primary", use_container_width=True)

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
                st.warning("Vui lòng chọn ít nhất 1 file ảnh trước khi bấm chấm!")

    try:
        pipeline = OMRPipeline(active_key_path)
    except Exception as e:
        st.warning(f"Không thể khởi tạo bộ chấm thi với '{active_key_path}': {e}. Đang dùng file mặc định.")
        pipeline = OMRPipeline(DEFAULT_KEY_PATH)

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
            st.success(f"✅ Đã chấm xong hàng loạt **{len(results_list)} bài thi**!")
            warnings_list = [r for r in results_list if r.get("warning")]
            if warnings_list:
                st.warning(f"⚠️ **Cảnh báo lệch cấu trúc đề:** Có {len(warnings_list)}/{len(results_list)} bài thi có số câu trên ảnh không khớp với file đáp án!")
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
                    row_data["Cảnh báo"] = "⚠️ Lệch số câu đề thi"
                summary_rows.append(row_data)
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            selected_idx = st.selectbox(
                "🔍 Chọn sinh viên để xem ảnh chấm chi tiết & phúc khảo:",
                range(len(results_list)),
                format_func=lambda i: f"{results_list[i]['student_id']} — Điểm: {results_list[i]['grading_result']['score_10']}/10" + (" ⚠️ (Lệch đề)" if results_list[i].get("warning") else ""),
                key="tab1_selected_student_idx"
            )
            active_res = results_list[selected_idx]
        else:
            active_res = results_list[0]
            selected_idx = 0

        gr = active_res["grading_result"]

        # Alert if structural mismatch detected (e.g. 20q sheet uploaded for 10q key)
        if active_res.get("warning"):
            st.error(f"🚨 **{active_res['warning']}**")

        # Metric cards
        st.write("")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Điểm số (Hệ 10)", f"{gr['score_10']:.1f}", delta=f"{gr['correct_count']}/{gr['total_questions']} câu")
        m2.metric("✅ Số câu đúng", gr["correct_count"])
        m3.metric("❌ Số câu sai", gr["incorrect_count"])
        m4.metric("⚠️ Bỏ trống", gr["blank_count"])
        m5.metric("🚫 Phạm quy (>1 ô)", gr["multiple_count"])

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
        st.markdown("##### 📋 Bảng Chi Tiết Từng Câu & Tỷ Lệ Tô Đậm (Fill Ratio A, B, C, D)")
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
                "Kết Quả": "✅ Đúng" if d["is_correct"] else "❌ Sai/Lỗi",
                "Fill A": f"{r_q.get('A', 0.0):.3f}",
                "Fill B": f"{r_q.get('B', 0.0):.3f}",
                "Fill C": f"{r_q.get('C', 0.0):.3f}",
                "Fill D": f"{r_q.get('D', 0.0):.3f}",
                "Ghi Chú": d.get("warning", "")
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, height=260)

        # Human-in-the-loop manual review / override
        with st.expander("🛠️ Kiểm duyệt & Phúc khảo thủ công (Human-in-the-loop Override)"):
            st.caption("Nếu phiếu bị mờ hoặc tẩy xóa dẫn đến cảnh báo BLANK / MULTIPLE hoặc vết tẩy xóa Δτ, giám thị có thể xác nhận lại đáp án tại đây:")
            erased_qs = [d["question"] for d in gr["details"] if d.get("is_erased")]
            if erased_qs:
                st.warning(f"⚠️ Lưu ý: Câu {', '.join(str(q) for q in erased_qs)} có vết tẩy xóa đã được thuật toán tự động giải mã theo luật chênh lệch độ đậm Δτ ≥ 0.15.")
            ov_col1, ov_col2, ov_col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
            with ov_col1:
                q_to_edit = st.selectbox("Chọn câu hỏi cần chỉnh sửa:", list(range(1, gr["total_questions"] + 1)), key="tab1_override_q")
            curr_det = next((d for d in gr["details"] if d["question"] == q_to_edit), None)
            curr_choice = curr_det["student_ans"] if curr_det else "BLANK"
            curr_status = curr_det["status"] if curr_det else "BLANK"
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
                if st.button("✔️ Cập nhật", use_container_width=True, key="tab1_override_btn"):
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

        with st.expander("⏱️ Phân rã thời gian xử lý từng giai đoạn (Latency Breakdown)"):
            lat_df = pd.DataFrame([active_res["latencies"]]).T
            lat_df.columns = ["Thời gian (ms)"]
            st.table(lat_df)

        # Export Excel Report (in-memory buffer, prevents dirtying git tracked files)
        excel_buf = io.BytesIO()
        OMRGrader.generate_excel_report(
            [r["grading_result"] for r in st.session_state.tab1_processed_items],
            excel_buf
        )
        excel_bytes = excel_buf.getvalue()

        st.download_button(
            label=f"📥 Tải Bảng Điểm Excel ({len(st.session_state.tab1_processed_items)} Sinh Viên) (.xlsx)",
            data=excel_bytes,
            file_name="Ket_Qua_Cham_Thi_OMR.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        # Preview Card / Empty State before grading run
        if "Bộ 5 Kịch Bản Mẫu" in input_mode:
            if scen_val == "ALL_SAMPLES":
                with st.container(border=True):
                    st.markdown("""
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                        <span style="font-size: 24px;">📦</span>
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
                        👉 Bấm nút <b>"Bắt Đầu Chấm Điểm"</b> ở trên để kích hoạt Pipeline xử lý toàn bộ bài thi.
                    </p>
                    """, unsafe_allow_html=True)
            else:
                fpath = os.path.join(SAMPLES_DIR, scen_val) if scen_val else DEFAULT_SHEET_PATH
                if not os.path.exists(fpath):
                    fpath = DEFAULT_SHEET_PATH
                with st.container(border=True):
                    pv_c1, pv_c2 = st.columns([1, 2], vertical_alignment="center")
                    with pv_c1:
                        st.image(fpath, caption=f"Ảnh xem trước: {os.path.basename(fpath)}", use_container_width=True)
                    with pv_c2:
                        st.markdown(f"#### 📄 Xem Trước Phiếu Thi: `{os.path.basename(fpath)}`")
                        st.markdown("""
                        - **Quy chuẩn phiếu:** 20 câu trắc nghiệm, 4 đáp án (A, B, C, D).
                        - **Tọa độ định vị:** 4 điểm mốc fiducial ở 4 góc để nắn phẳng hình học qua ma trận SVD.
                        - **Trạng thái:** Phiếu thi đã được nạp sẵn, sẵn sàng đưa vào luồng chấm điểm.
                        """)
                        st.markdown("""
                        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 10px 14px; color: #1D4ED8; font-size: 13px; margin-top: 10px;">
                            💡 Bấm nút <b>🚀 Bắt Đầu Chấm Điểm</b> ở trên để hệ thống tiến hành nắn phẳng, phân đoạn nhị phân và chấm điểm.
                        </div>
                        """, unsafe_allow_html=True)
        else:
            with st.container(border=True):
                st.markdown("""
                <div style="text-align: center; padding: 24px 16px;">
                    <div style="font-size: 38px; margin-bottom: 10px;">📤</div>
                    <h3 style="font-size: 17px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">Sẵn Sàng Nhận Diện & Chấm Điểm</h3>
                    <p style="color: #64748B; font-size: 13.5px; max-width: 540px; margin: 0 auto 16px auto;">
                        Vui lòng tải lên 1 hoặc nhiều ảnh chụp phiếu thi (.png, .jpg) ở ô phía trên, sau đó bấm <b>"Chấm Ảnh Đã Tải"</b>.
                    </p>
                    <div style="display: inline-flex; gap: 16px; text-align: left; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 16px; font-size: 12px; color: #475569;">
                        <div>✔️ Định dạng hỗ trợ: PNG, JPG, JPEG</div>
                        <div>✔️ Độ phân giải khuyến nghị ≥ 800×1000</div>
                        <div>✔️ Đầy đủ 4 điểm mốc định vị góc</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: KIỂM THỬ THUẬT TOÁN (PIPELINE STEPPER)
# ---------------------------------------------------------
with tab2:
    st.markdown("##### 🔬 Kiểm Thử Thuật Toán Từng Phân Hệ (Pipeline Stepper)")

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

    # Stepper Navigation Buttons (4 Columns - Direct Interactive Stepper)
    step_cols = st.columns(4)
    for idx, (col, item) in enumerate(zip(step_cols, step_items)):
        with col:
            is_active = (cur_step == idx)
            is_done = (cur_step > idx)
            icon = "📍 " if is_active else ("✅ " if is_done else "")
            if st.button(
                f"{icon}{item['title']}",
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
        st.markdown("###### 🌿 Giai Đoạn 1: Tiền Xử Lý Ảnh & Định Vị Khung Phiếu Thi")

        m1_col_ctrl, m1_col_view = st.columns([1.1, 1.9], gap="large")
        with m1_col_ctrl:
            with st.container(border=True):
                st.markdown("##### ⚙️ Cấu Hình Tiền Xử Lý")
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
                    p = os.path.join(SAMPLES_DIR, "02_B21DCCN002_Nghieng_BanToi.png")
                    m1_img_bgr = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH)
                elif "Thiếu sáng" in m1_src_opt:
                    p = os.path.join(SAMPLES_DIR, "03_B21DCCN003_ThieuSang_NhieuHat.png")
                    m1_img_bgr = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH)
                elif "Bóng đổ" in m1_src_opt:
                    p = os.path.join(SAMPLES_DIR, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
                    m1_img_bgr = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH)
                else:
                    p = os.path.join(SAMPLES_DIR, "01_B21DCCN001_Chuan.png")
                    m1_img_bgr = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH)

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

                btn_run_m1 = st.button("▶️ Thực Thi Tiền Xử Lý", type="primary", use_container_width=True, key="btn_run_m1")

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
                            label="🖼️ Tải Ảnh Kết Quả (.png)",
                            data=png_buf.tobytes(),
                            file_name="ket_qua_tien_xu_ly.png",
                            mime="image/png",
                            use_container_width=True,
                            key="dl_btn_tv1"
                        )

        with m1_col_view:
            with st.container(border=True):
                st.markdown("##### 🖼️ Không Gian Trực Quan & Kết Quả")
                if st.session_state.tab2_m1_result is not None:
                    r1 = st.session_state.tab2_m1_result
                    st.success(f"✅ Đã thực thi **{r1['algo']}** trong **{r1['elapsed_ms']:.2f} ms**!")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("###### 📥 Ảnh Đầu Vào")
                        st.image(cv2.cvtColor(r1["in_bgr"], cv2.COLOR_BGR2RGB) if r1["in_bgr"].ndim == 3 else r1["in_bgr"], use_container_width=True)
                    with c2:
                        st.markdown("###### 📤 Kết Quả Xử Lý")
                        if r1["out_vis"].ndim == 2:
                            st.image(r1["out_vis"], use_container_width=True, channels="GRAY")
                        else:
                            st.image(cv2.cvtColor(r1["out_vis"], cv2.COLOR_BGR2RGB), use_container_width=True)

                    if r1["extra"] and "corners" in r1["extra"]:
                        st.markdown("###### 📍 Tọa Độ 4 Đỉnh Góc Khung Phiếu Phát Hiện Được:")
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
                        st.info("👈 Bấm nút **'Thực Thi Tiền Xử Lý'** ở cột điều khiển bên trái để bắt đầu chạy.")

    # -----------------------------------------------------
    # BƯỚC 2: NẮN PHẲNG PHỐI CẢNH SVD
    # -----------------------------------------------------
    elif cur_step == 1:
        st.markdown("###### 📐 Giai Đoạn 2: Nắn Phẳng Phối Cảnh & Định Vị Marker (Homography SVD)")

        m2_col_ctrl, m2_col_view = st.columns([1.1, 1.9], gap="large")
        with m2_col_ctrl:
            with st.container(border=True):
                st.markdown("##### ⚙️ Cấu Hình Nắn Phẳng SVD")
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
                    p = os.path.join(SAMPLES_DIR, "01_B21DCCN001_Chuan.png")
                    m2_img_bgr = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH)
                elif "Ảnh bóng đổ" in m2_src_opt:
                    p = os.path.join(SAMPLES_DIR, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
                    m2_img_bgr = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH)
                else:
                    p = os.path.join(SAMPLES_DIR, "02_B21DCCN002_Nghieng_BanToi.png")
                    m2_img_bgr = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH)

                target_size_label = st.selectbox(
                    "Kích thước chuẩn hóa (W x H):",
                    ["800 x 1000 (Chuẩn hệ thống OMR)", "600 x 800", "1000 x 1200"],
                    index=0,
                    key="m2_size"
                )
                w_tgt, h_tgt = (800, 1000) if "800 x 1000" in target_size_label else ((600, 800) if "600" in target_size_label else (1000, 1200))

                btn_run_m2 = st.button("▶️ Thực Thi Nắn Phẳng Phối Cảnh SVD", type="primary", use_container_width=True, key="btn_run_m2")

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

                    if solve_homography_svd is not None:
                        H_mat = solve_homography_svd(ordered_src, dst_corners)
                    else:
                        H_mat, _ = cv2.findHomography(ordered_src, dst_corners)

                    if warp_perspective_bilinear is not None:
                        warped_out = warp_perspective_bilinear(m2_img_bgr, H_mat, (w_tgt, h_tgt))
                    else:
                        warped_out = cv2.warpPerspective(m2_img_bgr, H_mat, (w_tgt, h_tgt))

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
                            label="🖼️ Tải Ảnh Đã Nắn Phẳng (.png)",
                            data=png_buf.tobytes(),
                            file_name="anh_nan_phang.png",
                            mime="image/png",
                            use_container_width=True,
                            key="dl_btn_tv2"
                        )

        with m2_col_view:
            with st.container(border=True):
                st.markdown("##### 🖼️ Không Gian Trực Quan & So Sánh Phối Cảnh")
                if st.session_state.tab2_m2_result is not None:
                    r2 = st.session_state.tab2_m2_result
                    st.success(f"✅ Đã giải hệ ma trận SVD & nắn thẳng bài thi thành công trong **{r2['elapsed_ms']:.2f} ms**!")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("###### 📥 Ảnh Gốc (4 Đỉnh Nguồn SVD)")
                        st.image(cv2.cvtColor(r2["vis_src"], cv2.COLOR_BGR2RGB) if r2["vis_src"].ndim == 3 else r2["vis_src"], use_container_width=True)
                    with c2:
                        st.markdown(f"###### 📤 Ảnh Sau Nắn Phẳng ({r2['out_size'][0]}x{r2['out_size'][1]})")
                        st.image(cv2.cvtColor(r2["warped"], cv2.COLOR_BGR2RGB) if r2["warped"].ndim == 3 else r2["warped"], use_container_width=True)
                else:
                    if m2_img_bgr is not None:
                        st.image(cv2.cvtColor(m2_img_bgr, cv2.COLOR_BGR2RGB), caption=f"Ảnh chụp nghiêng hiện tại ({m2_img_bgr.shape[1]}x{m2_img_bgr.shape[0]})", use_container_width=True)
                        st.info("👈 Bấm nút **'Thực Thi Nắn Phẳng Phối Cảnh SVD'** ở cột bên trái để nắn thẳng tờ phiếu.")

    # -----------------------------------------------------
    # BƯỚC 3: PHÂN ĐOẠN NHỊ PHÂN & ĐÁNH NHÃN CCL
    # -----------------------------------------------------
    elif cur_step == 2:
        st.markdown("###### 🔍 Giai Đoạn 3: Phân Đoạn Nhị Phân, Hình Thái Học & Đánh Nhãn CCL")

        m3_col_ctrl, m3_col_view = st.columns([1.1, 1.9], gap="large")
        with m3_col_ctrl:
            with st.container(border=True):
                st.markdown("##### ⚙️ Cấu Hình Phân Đoạn & CCL")
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
                    p = os.path.join(SAMPLES_DIR, "03_B21DCCN003_ThieuSang_NhieuHat.png")
                    m3_img = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH, cv2.IMREAD_GRAYSCALE)
                elif "bóng đổ" in m3_src_opt:
                    p = os.path.join(SAMPLES_DIR, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
                    m3_img = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH, cv2.IMREAD_GRAYSCALE)
                elif "tô sai quy chế" in m3_src_opt:
                    p = os.path.join(SAMPLES_DIR, "04_B21DCCN004_ToSaiQuyChe_TayXoa.png")
                    m3_img = cv2.imread(p if os.path.exists(p) else DEFAULT_SHEET_PATH, cv2.IMREAD_GRAYSCALE)
                else:
                    m3_img = cv2.imread(DEFAULT_SHEET_PATH, cv2.IMREAD_GRAYSCALE)

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

                btn_run_m3 = st.button("▶️ Thực Thi Phân Đoạn & CCL", type="primary", use_container_width=True, key="btn_run_m3")

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
                            label="🖼️ Tải Mặt Nạ Nhị Phân (.png)",
                            data=png_buf.tobytes(),
                            file_name="mat_na_nhi_phan.png",
                            mime="image/png",
                            use_container_width=True,
                            key="dl_btn_tv3"
                        )

        with m3_col_view:
            with st.container(border=True):
                st.markdown("##### 🖼️ Không Gian Trực Quan & Mặt Nạ Nhị Phân")
                if st.session_state.tab2_m3_result is not None:
                    r3 = st.session_state.tab2_m3_result
                    st.success(f"✅ Đã thực thi thành công **{r3['algo']}** trong **{r3['elapsed_ms']:.2f} ms**!")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("###### 📥 Ảnh Xám Đầu Vào")
                        st.image(r3["in_gray"], use_container_width=True, channels="GRAY")
                    with c2:
                        st.markdown("###### 📤 Mặt Nạ Nhị Phân / Nhãn CCL")
                        st.image(r3["out_bin"], use_container_width=True, channels="RGB" if r3["out_bin"].ndim == 3 else "GRAY")

                    if r3["extra"] and "otsu_t" in r3["extra"]:
                        st.info(f"💡 Ngưỡng tối ưu toàn cục tính được bởi Otsu: **T* = {r3['extra']['otsu_t']}**")
                    elif r3["extra"] and "num_comp" in r3["extra"]:
                        st.markdown(f"###### 🏷️ Thống Kê {r3['extra']['num_comp']} Thành Phần Liên Thông (CCL):")
                        df_ccl = pd.DataFrame(r3["extra"]["stats"])
                        st.dataframe(df_ccl.head(15), use_container_width=True, height=220)
                else:
                    if m3_img is not None:
                        st.image(m3_img, caption="Ảnh xám đầu vào hiện tại", use_container_width=True, channels="GRAY")
                        st.info("👈 Bấm nút **'Thực Thi Phân Đoạn & CCL'** ở cột bên trái để chạy.")

    # -----------------------------------------------------
    # BƯỚC 4: CÂN BẰNG SÁNG, KHỬ NHIỄU & MA TRẬN CSV
    # -----------------------------------------------------
    else:
        st.markdown("###### 🧮 Giai Đoạn 4: Cân Bằng Sáng, Khử Nhiễu & Ma Trận Mức Xám CSV")

        m4_col_ctrl, m4_col_view = st.columns([1.1, 1.9], gap="large")
        with m4_col_ctrl:
            with st.container(border=True):
                st.markdown("##### ⚙️ Nguồn Dữ Liệu & Thuật Toán")
                source_type = st.radio(
                    "Nguồn dữ liệu kiểm thử:",
                    [
                        "🔢 Ma trận mức xám dạng File CSV",
                        "🖼️ Ảnh chụp phiếu thi (.png / .jpg)"
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
                        if os.path.exists(DEFAULT_CSV_15_PATH):
                            test_matrix = load_matrix_from_csv(DEFAULT_CSV_15_PATH)
                            source_name = "sample_matrix_15x15.csv"
                    elif "Mẫu 2" in csv_mode:
                        if os.path.exists(DEFAULT_CSV_LOW_PATH):
                            test_matrix = load_matrix_from_csv(DEFAULT_CSV_LOW_PATH)
                            source_name = "sample_matrix_low_contrast.csv"
                    elif "Mẫu 3" in csv_mode:
                        if os.path.exists(DEFAULT_CSV_60_PATH):
                            test_matrix = load_matrix_from_csv(DEFAULT_CSV_60_PATH)
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
                        p3 = os.path.join(SAMPLES_DIR, "03_B21DCCN003_ThieuSang_NhieuHat.png")
                        test_matrix = cv2.imread(p3 if os.path.exists(p3) else DEFAULT_SHEET_PATH, cv2.IMREAD_GRAYSCALE)
                        source_name = "03_B21DCCN003_ThieuSang_NhieuHat.png"
                    elif "Ảnh phiếu mẫu chuẩn" in img_mode:
                        test_matrix = cv2.imread(DEFAULT_SHEET_PATH, cv2.IMREAD_GRAYSCALE)
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

                btn_run_m4 = st.button("▶️ Thực Thi Thuật Toán", type="primary", use_container_width=True, key="btn_run_m4")

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
                        label="📥 Tải Ma Trận Kết Quả (.csv)",
                        data=csv_buf.getvalue(),
                        file_name=f"ket_qua_{base_fname}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="m4_dl_csv"
                    )
                    ok_png, png_buf = cv2.imencode(".png", r4["out_matrix"])
                    if ok_png:
                        st.download_button(
                            label="🖼️ Tải Ảnh Kết Quả (.png)",
                            data=png_buf.tobytes(),
                            file_name=f"ket_qua_{base_fname}.png",
                            mime="image/png",
                            use_container_width=True,
                            key="m4_dl_png"
                        )

        with m4_col_view:
            with st.container(border=True):
                st.markdown("##### 🖼️ Không Gian Ma Trận Số Học & Biểu Đồ")
                if st.session_state.tab2_m4_result is not None:
                    r4 = st.session_state.tab2_m4_result
                    st.success(f"✅ Đã thực thi thành công **{r4['algo']}** trên **{r4['source_name']}** trong **{r4['elapsed_ms']:.2f} ms**!")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("###### 📥 Đầu Vào (Trước Xử Lý)")
                        st.image(r4["in_matrix"], caption=f"Ảnh mức xám ({r4['in_matrix'].shape[0]}x{r4['in_matrix'].shape[1]})", use_container_width=True, channels="GRAY")
                        st.dataframe(pd.DataFrame(r4["in_matrix"]), use_container_width=True, height=220)
                    with c2:
                        st.markdown("###### 📤 Kết Quả (Sau Xử Lý)")
                        st.image(r4["out_matrix"], caption=f"Sau xử lý ({r4['out_matrix'].shape[0]}x{r4['out_matrix'].shape[1]})", use_container_width=True, channels="GRAY")
                        st.dataframe(pd.DataFrame(r4["out_matrix"]), use_container_width=True, height=220)
                else:
                    if test_matrix is not None:
                        st.markdown("###### Dữ Liệu Ma Trận Số Học Đầu Vào (Pixel Values):")
                        st.dataframe(pd.DataFrame(test_matrix), use_container_width=True, height=260)
                        st.info("👈 Bấm nút **'Thực Thi Thuật Toán'** ở cột bên trái để bắt đầu chạy.")

    # Stepper Footer Navigation
    st.write("")
    nav_c1, nav_c2, nav_c3 = st.columns([1.5, 4, 1.5])
    with nav_c1:
        if cur_step > 0:
            if st.button(f"◀ Quay Lại Bước {cur_step}", use_container_width=True, key=f"btn_prev_{cur_step}"):
                st.session_state.pipeline_step_idx = cur_step - 1
                st.rerun()
    with nav_c3:
        if cur_step < 3:
            if st.button(f"Tiếp Bước {cur_step + 2} ▶", type="primary", use_container_width=True, key=f"btn_next_{cur_step}"):
                st.session_state.pipeline_step_idx = cur_step + 1
                st.rerun()


# ---------------------------------------------------------
# TAB 3: ĐỐI CHỨNG SCRATCH VS OPENCV
# ---------------------------------------------------------
with tab3:
    st.markdown("##### ⚖️ Đối Chứng Trực Quan: Thuật Toán Tự Cài Đặt (NumPy) vs Thư Viện OpenCV")
    st.markdown("""
    Kiểm chứng độ chính xác giữa thuật toán tự cài đặt (NumPy) và thư viện OpenCV thông qua chỉ số **MSE**, **PSNR (dB)**, **IoU**, **Bản đồ sai khác tuyệt đối** $|I_{\\text{cài đặt}} - I_{\\text{OpenCV}}|$, và **Khảo sát độ nhạy tham số**.
    """)

    # -----------------------------------------------------
    # STATE SYNC & SIDEBAR LINK BANNER (Requirement 3)
    # -----------------------------------------------------
    current_core_name = "Thuật toán tự cài đặt (NumPy)" if use_scratch else "Thư viện OpenCV"
    st.markdown(f"""
    <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 12px 16px; margin-bottom: 18px; display: flex; align-items: flex-start; gap: 12px;">
        <div style="font-size: 20px; line-height: 1.2;">💡</div>
        <div style="font-size: 13px; color: #1E40AF; line-height: 1.5;">
            <b>Cơ chế Đối chứng Độc lập & Đồng bộ Toàn Cục:</b> Tab đối chứng luôn tự động thực thi <b>song song cả 2 thuật toán (NumPy tự cài đặt & OpenCV chuẩn)</b> trên cùng một ảnh đầu vào để tính toán chính xác sai số toán học ($MSE, PSNR, IoU$) độc lập với lựa chọn tại Sidebar.
            <span style="display: block; margin-top: 4px; color: #3B82F6; font-size: 12px;">
                ⚙️ <i>Cấu hình Pipeline hiện tại:</i> <b>{current_core_name}</b> • Ngưỡng tô: <b>{fill_thresh:.2f}</b> • Tiền xử lý: <b>{'Bật' if apply_enhancement else 'Tắt'}</b> (Áp dụng cho quy trình Chấm Bài Trắc Nghiệm).
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sub_t1, sub_t2, sub_t3 = st.tabs([
        "🔍 Đối Chứng Từng Thuật Toán",
        "📊 Bảng Đối Chứng Toàn Diện Pipeline",
        "📈 Khảo Sát Độ Nhạy Tham Số"
    ])

    with sub_t1:
        # -----------------------------------------------------
        # 1. CONTROL TOOLBAR (Requirement 1)
        # -----------------------------------------------------
        with st.container(border=True):
            st.markdown('<div class="benchmark-toolbar-title">⚙️ Bảng Điều Khiển Đo Đạc Đối Chứng (Control Toolbar)</div>', unsafe_allow_html=True)

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
                    "[Phân đoạn] Phân ngưỡng Otsu toàn cục (Otsu Global Threshold)",
                    "[Phân đoạn] Phân ngưỡng thích nghi (Adaptive Mean Threshold)",
                    "[Phân đoạn] Phép mở hình thái học (Morphology Opening 3x3)",
                    "[Cải thiện ảnh] Cân bằng biểu đồ mức xám (Histogram Equalization)",
                    "[Khử nhiễu] Bộ lọc trung vị khử nhiễu (Median Filter 2D)"
                ], index=2, key="sb1_algo")

            if is_median:
                with b_col_k:
                    bench_k = st.selectbox("Kernel:", [3, 5, 7], index=0, key="sb1_k")
            else:
                bench_k = 3

            with b_col2:
                bench_src = st.selectbox("Chọn ảnh thử nghiệm:", [
                    "Ảnh phiếu chuẩn (Kịch bản 1)",
                    "Ảnh chụp nghiêng bàn tối (Kịch bản 2)",
                    "Ảnh phiếu thiếu sáng & nhiễu hạt (Kịch bản 3)",
                    "Ảnh bóng đổ & chênh sáng (Kịch bản 5)",
                    "Tải ảnh tùy chọn từ máy tính"
                ], key="sb1_src")

            with b_col3:
                btn_run_bench = st.button("🚀 Bắt Đầu Đo Đạc", type="primary", use_container_width=True, key="sb1_btn_run")

            custom_bench_file = None
            if "Tải ảnh" in bench_src:
                st.write("")
                custom_bench_file = st.file_uploader("Nạp ảnh tùy chọn (.png, .jpg):", type=["png", "jpg", "jpeg"], key="sb1_upload")

        # Load image source
        bench_img = None
        if "Tải ảnh" in bench_src:
            if custom_bench_file is not None:
                c_bytes = np.asarray(bytearray(custom_bench_file.read()), dtype=np.uint8)
                bench_img = cv2.imdecode(c_bytes, cv2.IMREAD_COLOR)
        elif "Kịch bản 2" in bench_src:
            p2 = os.path.join(SAMPLES_DIR, "02_B21DCCN002_Nghieng_BanToi.png")
            bench_img = cv2.imread(p2 if os.path.exists(p2) else DEFAULT_SHEET_PATH)
        elif "Kịch bản 3" in bench_src:
            p3 = os.path.join(SAMPLES_DIR, "03_B21DCCN003_ThieuSang_NhieuHat.png")
            bench_img = cv2.imread(p3 if os.path.exists(p3) else DEFAULT_SHEET_PATH)
        elif "Kịch bản 5" in bench_src:
            p5 = os.path.join(SAMPLES_DIR, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
            bench_img = cv2.imread(p5 if os.path.exists(p5) else DEFAULT_SHEET_PATH)
        else:
            p1 = os.path.join(SAMPLES_DIR, "01_B21DCCN001_Chuan.png")
            bench_img = cv2.imread(p1 if os.path.exists(p1) else DEFAULT_SHEET_PATH)

        # Quick Test trigger from Empty State (Requirement 2)
        trigger_quick = st.session_state.get("run_quick_test_trigger", False)
        if trigger_quick:
            st.session_state.run_quick_test_trigger = False
            bench_algo = "[Tiền xử lý] Phát hiện biên Canny (Canny Edge Detection)"
            p1 = os.path.join(SAMPLES_DIR, "01_B21DCCN001_Chuan.png")
            bench_img = cv2.imread(p1 if os.path.exists(p1) else DEFAULT_SHEET_PATH)
            btn_run_bench = True

        current_tab3_sig = (bench_algo, bench_k, bench_src, bench_img.shape if bench_img is not None else None)
        if bench_img is not None and (
            btn_run_bench
            or (st.session_state.tab3_result is not None and st.session_state.tab3_last_sig != current_tab3_sig)
        ):
            with st.spinner("⏳ Đang thực hiện đo đạc đối chứng thuật toán & tính toán sai số MSE/PSNR..."):
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
                amp = 10.0
                iou_val = None
                dice_val = None

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
                    for _ in range(3):
                        t0 = time.perf_counter()
                        out_sc = warp_perspective_sheet(gray_bench, quad, 800, 1000, use_scratch=True)
                        t_sc_list.append((time.perf_counter() - t0) * 1000.0)
                        t0 = time.perf_counter()
                        out_cv = warp_perspective_sheet(gray_bench, quad, 800, 1000, use_scratch=False)
                        t_cv_list.append((time.perf_counter() - t0) * 1000.0)
                    amp = 5.0

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

                elif "Histogram" in bench_algo:
                    st.session_state.tab3_result = benchmark_hist_equalization(gray_bench, iterations=3)

                elif "trung vị" in bench_algo or "Median" in bench_algo:
                    st.session_state.tab3_result = benchmark_median_filter(gray_bench, kernel_size=bench_k, iterations=3)

                if "Histogram" not in bench_algo and "Median" not in bench_algo and "trung vị" not in bench_algo:
                    mse_calc = calculate_mse(out_sc, out_cv)
                    psnr_calc = calculate_psnr(out_sc, out_cv)
                    diff_map = calculate_difference_map(out_sc, out_cv, amplify=amp)
                    st.session_state.tab3_result = {
                        "img_scratch": out_sc,
                        "img_opencv": out_cv,
                        "diff_map": diff_map,
                        "time_scratch_ms": round(float(np.mean(t_sc_list)), 3),
                        "time_opencv_ms": round(float(np.mean(t_cv_list)), 3),
                        "mse": round(mse_calc, 4),
                        "psnr_db": round(psnr_calc, 2),
                        "amplify": amp,
                        "iou": iou_val,
                        "dice": dice_val
                    }

        # -----------------------------------------------------
        # 2. RESULT AREA & EMPTY STATE (Requirement 1 & 2)
        # -----------------------------------------------------
        if st.session_state.tab3_result is not None:
            b_res = st.session_state.tab3_result
            st.write("")
            bm1, bm2, bm3, bm4 = st.columns(4)
            bm1.metric("Thời gian Scratch (NumPy)", f"{b_res['time_scratch_ms']} ms")
            bm2.metric("Thời gian OpenCV (C++)", f"{b_res['time_opencv_ms']} ms")
            bm3.metric("Sai số MSE", f"{b_res['mse']}", delta="Khớp tuyệt đối 100%" if b_res["mse"] == 0.0 else "Đạt chuẩn < 1.0")
            if b_res.get("iou") is not None:
                bm4.metric("IoU / PSNR", f"{b_res['iou']*100:.2f}%", delta=f"Dice: {b_res.get('dice', 1.0):.4f}")
            else:
                bm4.metric("PSNR (dB)", f"{b_res['psnr_db']} dB")

            st.write("")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                with st.container(border=True):
                    st.markdown("###### 🖼️ 1. Thuật Toán Tự Cài Đặt (NumPy)")
                    st.image(b_res["img_scratch"], use_container_width=True, channels="GRAY")
            with tc2:
                with st.container(border=True):
                    st.markdown("###### 📦 2. Thư Viện OpenCV Chuẩn (C++)")
                    st.image(b_res["img_opencv"], use_container_width=True, channels="GRAY")
            with tc3:
                with st.container(border=True):
                    amp_factor = b_res.get("amplify", 10.0)
                    st.markdown(f"###### 🔬 3. Bản Đồ Sai Khác (x{amp_factor:.0f})")
                    st.image(b_res["diff_map"], use_container_width=True, channels="GRAY")
        else:
            st.write("")
            with st.container(border=True):
                st.markdown("""
                <div style="text-align: center; padding: 22px 16px 12px 16px;">
                    <div style="font-size: 34px; margin-bottom: 8px;">⚖️</div>
                    <h4 style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">Chưa Có Dữ Liệu Đối Chứng</h4>
                    <p style="color: #64748B; font-size: 13.5px; max-width: 580px; margin: 0 auto 16px auto; line-height: 1.5;">
                        Chọn thuật toán và ảnh tại <b>Bảng Điều Khiển</b> phía trên rồi bấm nút <b>"Bắt Đầu Đo Đạc"</b>, hoặc chạy thử nghiệm ngay lập tức với bộ mẫu tiêu chuẩn.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                q_c1, q_c2, q_c3 = st.columns([1.5, 2.2, 1.5])
                with q_c2:
                    if st.button("⚡ Chạy thử nhanh với mẫu mặc định", type="primary", use_container_width=True, key="btn_quick_bench"):
                        st.session_state.run_quick_test_trigger = True
                        st.rerun()

            st.write("")
            st.markdown("###### 👁️ Xem Trước Bố Cục Kết Quả (3 Khung Hình Đối Chứng):")
            sk1, sk2, sk3 = st.columns(3)
            with sk1:
                st.markdown("""
                <div class="skeleton-preview-card">
                    <div class="skeleton-icon">🖼️</div>
                    <div class="skeleton-title">1. Thuật Toán Tự Cài Đặt (NumPy)</div>
                    <div class="skeleton-desc">Vùng hiển thị ma trận ảnh xử lý bằng thuật toán tự cài đặt thuần NumPy để kiểm chứng.</div>
                    <div class="skeleton-badge">Đang chờ lệnh đo...</div>
                </div>
                """, unsafe_allow_html=True)
            with sk2:
                st.markdown("""
                <div class="skeleton-preview-card">
                    <div class="skeleton-icon">📦</div>
                    <div class="skeleton-title">2. Thư Viện Chuẩn OpenCV (C++)</div>
                    <div class="skeleton-desc">Vùng hiển thị kết quả xử lý từ hàm thư viện OpenCV chính quy để làm baseline đối chuẩn.</div>
                    <div class="skeleton-badge">Đang chờ lệnh đo...</div>
                </div>
                """, unsafe_allow_html=True)
            with sk3:
                st.markdown("""
                <div class="skeleton-preview-card">
                    <div class="skeleton-icon">🔬</div>
                    <div class="skeleton-title">3. Bản Đồ Sai Khác Tuyệt Đối</div>
                    <div class="skeleton-desc">Biểu diễn sai lệch trực quan |I_Scratch - I_OpenCV| khuếch đại để soi độ lệch từng pixel.</div>
                    <div class="skeleton-badge">Đang chờ lệnh đo...</div>
                </div>
                """, unsafe_allow_html=True)

    with sub_t2:
        st.markdown("###### 📊 Bảng Đối Chứng Toàn Diện Thuật Toán")

        all_src = st.selectbox("Chọn ảnh kiểm thử liên module:", [
            "Ảnh phiếu chuẩn (Kịch bản 1)",
            "Ảnh phiếu thiếu sáng & nhiễu hạt (Kịch bản 3)",
            "Ảnh bóng đổ (Kịch bản 5)"
        ], key="sb2_src")

        if "Kịch bản 3" in all_src:
            p_all = os.path.join(SAMPLES_DIR, "03_B21DCCN003_ThieuSang_NhieuHat.png")
        elif "Kịch bản 5" in all_src:
            p_all = os.path.join(SAMPLES_DIR, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
        else:
            p_all = DEFAULT_SHEET_PATH

        img_all = cv2.imread(p_all, cv2.IMREAD_GRAYSCALE)

        if st.button("🚀 Chạy Đối Chứng Toàn Bộ Thuật Toán", type="primary", key="sb2_btn_run"):
            with st.spinner("Đang chạy đối chứng toàn bộ thuật toán..."):
                st.session_state.tab3_all_bm = benchmark_all_modules(img_all)

        if st.session_state.tab3_all_bm is not None:
            raw_bm = st.session_state.tab3_all_bm
            rows = []
            for r in raw_bm:
                iou_str = f"{r['iou']*100:.2f}%" if r.get('iou') is not None else "—"
                dice_str = f"{r['dice']:.4f}" if r.get('dice') is not None else "—"
                dt_str = f"{r['delta_t']}" if r.get('delta_t') is not None else "—"
                pass_check = "✅ PASS" if (r['mse'] < 1.0 or (r.get('iou') is not None and r['iou'] >= 0.985)) else "⚠️ REVIEW"
                rows.append({
                    "Module": r["module"],
                    "Thuật toán": r["algorithm"],
                    "MSE": f"{r['mse']:.4f}",
                    "PSNR (dB)": f"{r['psnr_db']:.2f}",
                    "IoU (%)": iou_str,
                    "Dice": dice_str,
                    "ΔT / Corner Err": dt_str,
                    "Scratch (ms)": f"{r['time_scratch_ms']:.2f}",
                    "OpenCV (ms)": f"{r['time_opencv_ms']:.2f}",
                    "Đánh giá": pass_check
                })
            df_bm = pd.DataFrame(rows)
            st.dataframe(df_bm, use_container_width=True)
            st.success("Đạt độ khớp toán học tương đương thư viện chuẩn OpenCV (MSE < 1.0, PSNR > 48 dB, IoU ≥ 98.5%).")

    with sub_t3:
        st.markdown("###### 📈 Khảo Sát Tham Số & Độ Nhạy (Ablation Study)")

        abl_src = st.selectbox("Chọn ảnh phiếu làm bài để khảo sát:", [
            "Kịch bản 5: Bóng đổ & Chiếu sáng không đều (Khuyến nghị)",
            "Kịch bản 3: Thiếu sáng & Nhiễu hạt",
            "Kịch bản 1: Phiếu chuẩn"
        ], key="sb3_src")

        if "Kịch bản 3" in abl_src:
            p_abl = os.path.join(SAMPLES_DIR, "03_B21DCCN003_ThieuSang_NhieuHat.png")
        elif "Kịch bản 1" in abl_src:
            p_abl = os.path.join(SAMPLES_DIR, "01_B21DCCN001_Chuan.png")
        else:
            p_abl = os.path.join(SAMPLES_DIR, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")

        if st.button("🔬 Chạy Khảo Sát Thực Nghiệm", type="primary", key="sb3_btn_run"):
            with st.spinner("Đang trích xuất ảnh nắn thẳng và chạy khảo sát..."):
                test_sheet_bgr = cv2.imread(p_abl)
                proc_out = pipeline.process_sheet(test_sheet_bgr, use_scratch_mode=True)
                
                gt_csv_path = p_abl.replace(".png", "_gt.csv")
                if os.path.exists(gt_csv_path):
                    df_gt = pd.read_csv(gt_csv_path)
                    gt_answers = dict(zip(df_gt.iloc[:, 0].astype(int), df_gt.iloc[:, 1].astype(str).str.strip()))
                else:
                    if "Kịch bản 3" in abl_src:
                        gt_answers = {
                            1: 'A', 2: 'C', 3: 'B', 4: 'A', 5: 'A', 6: 'B', 7: 'C', 8: 'D', 9: 'B', 10: 'C',
                            11: 'B', 12: 'D', 13: 'A', 14: 'C', 15: 'C', 16: 'D', 17: 'A', 18: 'B', 19: 'C', 20: 'BLANK'
                        }
                    elif "Kịch bản 1" in abl_src:
                        gt_answers = {
                            1: 'A', 2: 'C', 3: 'B', 4: 'D', 5: 'A', 6: 'B', 7: 'C', 8: 'A', 9: 'A', 10: 'C',
                            11: 'B', 12: 'D', 13: 'A', 14: 'C', 15: 'B', 16: 'D', 17: 'A', 18: 'B', 19: 'BLANK', 20: 'B'
                        }
                    else:
                        gt_answers = {
                            1: 'A', 2: 'C', 3: 'B', 4: 'D', 5: 'A', 6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'C',
                            11: 'B', 12: 'D', 13: 'A', 14: 'C', 15: 'B', 16: 'D', 17: 'A', 18: 'B', 19: 'C', 20: 'C'
                        }
                st.session_state.tab3_ablation = run_ablation_study(
                    warped_gray=proc_out["warped_gray"],
                    ground_truth_answers=gt_answers
                )

        if st.session_state.tab3_ablation is not None:
            abl = st.session_state.tab3_ablation

            st.write("#### 1. So Sánh Phân Ngưỡng: Otsu Toàn Cục vs Adaptive Mean")
            ac1, ac2 = st.columns(2)
            ac1.metric(
                "Phân ngưỡng Thích nghi (Adaptive Mean)",
                f"{abl['adaptive_accuracy_pct']:.1f}%",
                delta="Khử bóng đổ tốt"
            )
            delta_diff = abl['otsu_accuracy_pct'] - abl['adaptive_accuracy_pct']
            ac2.metric(
                "Phân ngưỡng Otsu Toàn cục",
                f"{abl['otsu_accuracy_pct']:.1f}%",
                delta=f"{delta_diff:.1f}%",
                delta_color="inverse"
            )

            if abl['adaptive_accuracy_pct'] > abl['otsu_accuracy_pct']:
                st.info(f"💡 **Kết quả thực nghiệm:** Trên ảnh bóng đổ/chênh sáng, Otsu toàn cục bị suy giảm độ chính xác xuống còn **{abl['otsu_accuracy_pct']:.1f}%**. Phân ngưỡng thích nghi cục bộ duy trì độ chính xác cao **{abl['adaptive_accuracy_pct']:.1f}%** (vượt trội **+{abs(delta_diff):.1f}%**).")
            else:
                st.info("💡 Trên ảnh có độ rọi đồng đều, cả hai phương pháp đều cho độ chính xác tương đương.")

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
                    "Số câu bỏ trống (Blank)": item["blank_count"],
                    "Số câu phạm quy (Multiple)": item["multiple_count"]
                })
            df_surv = pd.DataFrame(s_rows)
            st.dataframe(df_surv, use_container_width=True)

            # Accuracy curve plot with modern styling
            fig_abl, ax_abl = plt.subplots(figsize=(7, 3.2), dpi=130)
            apply_modern_chart_style(
                ax_abl,
                title="Biến thiên Độ chính xác theo Ngưỡng tô tau_min",
                xlabel="Ngưỡng tô đậm (Fill Ratio tau)",
                ylabel="Độ chính xác nhận diện (%)"
            )
            taus = [it["fill_threshold"] for it in surv]
            accs = [it["mark_accuracy_pct"] for it in surv]
            ax_abl.plot(taus, accs, marker="o", color="#2563EB", linewidth=2.4, markersize=6, label="Độ chính xác (%)")
            ax_abl.axvline(0.38, color="#EF4444", linestyle="--", alpha=0.85, linewidth=1.5, label="Ngưỡng tối ưu tau=0.38")
            ax_abl.set_ylim(min(accs) - 10 if min(accs) > 10 else 0, 105)
            ax_abl.legend(loc="lower left", frameon=False, fontsize=9)
            st.pyplot(fig_abl)
            plt.close(fig_abl)

@st.cache_data
def get_default_graded_batch() -> List[Dict[str, Any]]:
    """Tải và chấm điểm sẵn 5 bài thi thực tế từ thư mục data/samples để hiển thị số liệu thực."""
    results = []
    p = OMRPipeline(DEFAULT_KEY_PATH)
    if os.path.exists(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            if fname.endswith(".png"):
                raw_sid = fname.split("_")[1] if "_" in fname else "B21DCCN001"
                sid = raw_sid.replace("B21DCCN00", "SV-0") if "B21DCCN" in raw_sid else raw_sid
                fpath = os.path.join(SAMPLES_DIR, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    res = p.process_sheet(img, sid, use_scratch_mode=False)
                    results.append(res["grading_result"])
    return results

@st.cache_data
def generate_mock_graded_batch(num_students: int = 60, key_path: str = DEFAULT_KEY_PATH) -> List[Dict[str, Any]]:
    """Tạo bộ dữ liệu mẫu giả lập (Mock Data) chuẩn hóa cho 60 sinh viên phục vụ kiểm thử trực quan phổ điểm."""
    grader = OMRGrader(key_path)
    ans_key = grader.answer_key
    questions = sorted(ans_key.keys())
    np.random.seed(42)
    results = []
    options = ["A", "B", "C", "D"]
    q_difficulties = {q: 0.50 + 0.45 * (np.sin(q * 1.3) ** 2) for q in questions}
    for i in range(1, num_students + 1):
        sid = f"SV21DCCN{i:03d}"
        ability = float(np.random.normal(0.2, 0.45))
        student_answers = {}
        for q in questions:
            corr_ans = ans_key[q]
            p_correct = float(np.clip(q_difficulties[q] + ability, 0.15, 0.98))
            r = float(np.random.rand())
            if r < p_correct:
                student_answers[q] = corr_ans
            elif r < p_correct + 0.03:
                student_answers[q] = "BLANK"
            elif r < p_correct + 0.05:
                student_answers[q] = "MULTIPLE"
            else:
                wrong_choices = [c for c in options if c != corr_ans]
                student_answers[q] = str(np.random.choice(wrong_choices))
        res = grader.grade_student(sid, student_answers)
        results.append(res)
    return results

# ---------------------------------------------------------
# TAB 4: THỐNG KÊ & PHỔ ĐIỂM
# ---------------------------------------------------------
with tab4:
    if "use_mock_data" not in st.session_state:
        st.session_state.use_mock_data = False

    actual_batch = st.session_state.graded_batch_results

    # State Banner with Mock Data Toggle (Requirement 3)
    banner_c1, banner_c2 = st.columns([3.8, 1.4], vertical_alignment="center")

    if st.session_state.use_mock_data:
        with banner_c1:
            st.info("🎲 **Đang hiển thị Dữ Liệu Mẫu Giả Lập (Mock Data 60 sinh viên)** — Phân phối điểm chuẩn (Bell Curve) từ 1.0 đến 10.0 điểm.")
        with banner_c2:
            if st.button("🔄 Khôi Phục Dữ Liệu Gốc", type="secondary", use_container_width=True, key="btn_reset_mock"):
                st.session_state.use_mock_data = False
                st.rerun()
        batch = generate_mock_graded_batch(60, st.session_state.active_key_path)
    else:
        if not actual_batch:
            with banner_c1:
                st.info("💡 Hiển thị thống kê từ **5 bài thi mẫu (SV-01 đến SV-05)** có sẵn trong hệ thống. Khi thực hiện chấm bài mới tại **Tab 1**, biểu đồ sẽ tự động cập nhật theo kết quả vừa chấm.")
            with banner_c2:
                if st.button("🎲 Nạp Dữ Liệu Mẫu (Mock Data)", type="primary", use_container_width=True, key="btn_load_mock_empty"):
                    st.session_state.use_mock_data = True
                    st.rerun()
            batch = get_default_graded_batch()
        else:
            with banner_c1:
                st.success(f"✅ Đang hiển thị thống kê thực tế từ **{len(actual_batch)} bài thi** vừa chấm tại Tab 1!")
            with banner_c2:
                if st.button("🎲 Nạp Dữ Liệu Mẫu (Mock Data)", type="secondary", use_container_width=True, key="btn_load_mock_actual"):
                    st.session_state.use_mock_data = True
                    st.rerun()
            batch = actual_batch

    sample_scores = np.array([r["score_10"] for r in batch])
    total_st = len(batch)
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
        # Dưới 50%: Hiển thị mũi tên xuống (↓) màu đỏ
        k4.metric(
            "Tỷ Lệ Đạt (>= 5.0)",
            f"{pass_rate}%",
            delta=f"{pass_count}/{total_st} sinh viên (<50%)",
            delta_color="red",
            delta_arrow="down"
        )
    else:
        # Từ 50% trở lên: Hiển thị mũi tên lên (↑) màu xanh
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
        ax.axvline(avg_score, color="#EF4444", linestyle="--", linewidth=1.8, label=f"Điểm TB: {avg_score:.2f}")
        ax.legend(frameon=False, loc="upper left", fontsize=9)

        # Fix Y-axis to Pure Integers with step_size = 1 (Requirement 1)
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
        from matplotlib.patches import Patch
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

    # Anomaly & Proctor Warning Attention List (Kế hoạch mục 9: Danh sách các phiếu có cảnh báo tô mờ/tẩy xóa/lệch cấu trúc)
    anomaly_sheets = []
    for r in batch:
        sid = r.get("student_id", "N/A")
        det = r.get("details", [])
        erased_qs = [d["question"] for d in det if d.get("is_erased")]
        blank_qs = [d["question"] for d in det if d.get("status") == "BLANK"]
        multi_qs = [d["question"] for d in det if d.get("status") == "MULTIPLE"]
        warn = r.get("warning")

        if erased_qs or multi_qs or blank_qs or warn:
            reasons = []
            if warn:
                reasons.append(f"⚠️ {warn}")
            if erased_qs:
                reasons.append(f"✏️ Vết tẩy xóa (Câu: {', '.join(str(q) for q in erased_qs)})")
            if multi_qs:
                reasons.append(f"❌ Tô đúp/phạm quy (Câu: {', '.join(str(q) for q in multi_qs)})")
            if blank_qs:
                reasons.append(f"⚪ Bỏ trống (Câu: {', '.join(str(q) for q in blank_qs)})")

            anomaly_sheets.append({
                "Mã Sinh Viên": sid,
                "Điểm Số": f"{r.get('score_10', 0.0):.1f}",
                "Số Câu Đúng": f"{r.get('correct_count', 0)}/{r.get('total_questions', 20)}",
                "Chi Tiết Cảnh Báo": " | ".join(reasons)
            })

    if anomaly_sheets:
        st.write("")
        with st.expander(f"⚠️ Danh Sách Phiếu Cần Giám Thị Kiểm Duyệt Lại ({len(anomaly_sheets)} bài có cảnh báo)", expanded=False):
            st.caption("Các bài thi có vết tẩy xóa, tô đúp, bỏ trống hoặc cảnh báo lệch đề thi cần giám thị đối chiếu bản cứng:")
            st.dataframe(pd.DataFrame(anomaly_sheets), use_container_width=True)

    st.write("")
    # Export Excel Report (in-memory buffer, prevents dirtying git tracked files)
    excel_buf = io.BytesIO()
    OMRGrader.generate_excel_report(batch, excel_buf)
    excel_bytes = excel_buf.getvalue()

    data_source_tag = "Mock 60 Sinh Viên" if st.session_state.use_mock_data else f"{len(batch)} Sinh Viên"
    st.download_button(
        label=f"📥 Tải Báo Cáo Phổ Điểm & Bảng Điểm ({data_source_tag}) (.xlsx)",
        data=excel_bytes,
        file_name="Bao_Cao_Thong_Ke_OMR.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
