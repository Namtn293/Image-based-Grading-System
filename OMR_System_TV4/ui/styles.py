"""Modern UI Stylesheet and Chart Styling Helpers for OMR Vision.
Standardized Light SaaS & Academic Engineering Theme (Zero emoji text clutter).
"""

import streamlit as st


def inject_custom_styles():
    """Inject polished, modern CSS styling without visual emoji clutter."""
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
        font-size: 14px;
        font-weight: 700;
        color: #2563EB;
        background: #EFF6FF;
        border-radius: 20px;
        padding: 4px 12px;
        margin-bottom: 8px;
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
        font-size: 15px;
        font-weight: 800;
        letter-spacing: -0.5px;
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


def apply_modern_chart_style(ax, title=None, xlabel=None, ylabel=None, title_pad=12):
    """Helper function for modern Matplotlib visual styling."""
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
