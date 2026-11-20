"""
DỰ ÁN XỬ LÝ ẢNH (INT13146) - HỆ THỐNG CHẤM BÀI TRẮC NGHIỆM OMR
Test Suite: test_tv3_benchmark.py (Thành viên 3 - Phân đoạn ảnh & Nhận diện đáp án)
Kiểm thử tích hợp đầu-cuối & Xác thực toàn bộ chỉ tiêu nghiệm thu (Acceptance Criteria):
    1. Delta T_Otsu <= 1 mức xám.
    2. IoU nhị phân >= 98.5% (Thresholding và Morphology) so với OpenCV.
    3. PSNR >= 48.0 dB (hoặc vô hạn khi trùng khớp hoàn hảo).
    4. Độ chính xác nhận diện đáp án 100% trên phiếu kiểm thử chuẩn.
    5. Khớp số lượng thành phần liên thông CCL (Two-Pass Scratch vs OpenCV).
    6. Khảo sát tham số (Ablation Study): Otsu toàn cục vs Adaptive Mean trên tập bóng đổ DS-03.
    7. Xử lý chính xác 4 trạng thái: VALID, BLANK, ERASED_VALID, MULTIPLE_INVALID.
"""

import os
import sys
import json
import csv
import numpy as np
import pytest
import cv2

# Đảm bảo import từ src
_test_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.abspath(os.path.join(_test_dir, ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from src.tv3_thresholding import (
    otsu_threshold_scratch,
    adaptive_mean_threshold_scratch,
    calculate_segmentation_metrics,
    benchmark_otsu_vs_opencv,
    benchmark_adaptive_vs_opencv,
)
from src.tv3_morphology_ccl import (
    create_structuring_element,
    opening_binary_scratch,
    closing_binary_scratch,
    two_pass_ccl_scratch,
    filter_bubble_components,
    calc_binary_metrics,
)
from src.tv3_roi_classifier import (
    STATUS_VALID,
    STATUS_BLANK,
    STATUS_ERASED_VALID,
    STATUS_MULTIPLE_INVALID,
    DEFAULT_TAU_MIN,
    DEFAULT_DELTA_TAU_MIN,
    recognize_all_questions,
)
from src.tv3_segmentation import (
    TV3PipelineConfig,
    OMRSharedConfig,
    TV3PipelineResult,
    TV3SegmentationPipeline,
    run_tv3_pipeline,
    detect_bubble_grid_scratch,
    extract_bubble_answers,
    extract_bubble_answers_with_warnings,
    generate_synthetic_omr_sheet,
    visualize_pipeline_steps,
)


@pytest.fixture(scope="module")
def synthetic_ds01_clean():
    """Fixture sinh phiếu OMR chuẩn không bóng đổ (DS-01 Baseline)."""
    return generate_synthetic_omr_sheet(
        width=800,
        height=1000,
        num_questions=20,
        add_shadow=False,
        add_noise=False,
        seed=101
    )


@pytest.fixture(scope="module")
def synthetic_ds03_shadow():
    """Fixture sinh phiếu OMR có bóng đổ chéo dải đậm (DS-03 Uneven Lighting)."""
    return generate_synthetic_omr_sheet(
        width=800,
        height=1000,
        num_questions=20,
        add_shadow=True,
        add_noise=True,
        seed=202
    )


# ==============================================================================
# 1. KIỂM THỬ KHỞI TẠO VÀ CẤU HÌNH PIPELINE
# ==============================================================================

def test_pipeline_config_initialization():
    """Kiểm tra khởi tạo TV3PipelineConfig với các giá trị mặc định chuẩn."""
    cfg = TV3PipelineConfig()
    assert cfg.threshold_method == "adaptive"
    assert cfg.block_size == 25
    assert cfg.C == 10.0
    assert cfg.binary_method == "BINARY_INV"
    assert cfg.apply_opening is True
    assert cfg.apply_closing is True
    assert cfg.bubble_radius == 12.0
    assert cfg.tau_min == 0.38
    assert cfg.delta_tau_min == 0.15
    assert cfg.num_questions == 20

    pipeline = TV3SegmentationPipeline(cfg)
    assert pipeline.structuring_element.shape == (3, 3)
    assert np.all(pipeline.structuring_element == 1)


# ==============================================================================
# 2. KIỂM THỬ TẬP DS-01 (CHUẨN BASELINE) VÀ NGHIỆM THU ĐỐI CHỨNG OPENCV
# ==============================================================================

def test_ds01_clean_baseline_pipeline(synthetic_ds01_clean):
    """Kiểm thử toàn bộ pipeline trên tập ảnh chuẩn DS-01 (độ chính xác 100%, IoU >= 98.5%)."""
    sheet_gray, gt_answers, gt_statuses, grid_rois = synthetic_ds01_clean
    pipeline = TV3SegmentationPipeline(TV3PipelineConfig(threshold_method="adaptive"))
    result = pipeline.run(sheet_gray, question_grid=grid_rois, run_opencv_benchmark=True)

    # 1. Kiểm tra độ chính xác nhận diện đáp án (100% trên cả 20 câu)
    answers = result.get_answer_vector()
    for q_id in range(1, 21):
        assert answers[q_id] == gt_answers[q_id], f"Câu {q_id}: Nhận diện {answers[q_id]} != GT {gt_answers[q_id]}"

    # 2. Kiểm tra chỉ tiêu nghiệm thu đối chứng OpenCV
    bench = result.benchmark_opencv
    assert bench is not None
    assert bench["acceptance_criteria"]["iou_morph_pass"] is True
    assert bench["morphology_metrics"]["iou_percent"] >= 98.5
    assert bench["morphology_metrics"]["psnr_db"] >= 48.0 or bench["morphology_metrics"]["psnr_db"] == 999.0
    assert bench["ccl_count_match"] is True


# ==============================================================================
# 3. KHẢO SÁT THAM SỐ (ABLATION STUDY): OTSU VS ADAPTIVE TRÊN TẬP BÓNG ĐỔ DS-03
# ==============================================================================

def test_ds03_shadow_ablation_otsu_vs_adaptive():
    """
    Thực nghiệm Ablation Study theo đúng Mục 6.2 và Bảng 7.2 của Kế hoạch đề tài:
    - Mô phỏng vùng bóng đổ sâu (Deep Shadow / Hand Shadow) đè lên nửa phiếu thi (DS-03).
    - Chứng minh phân ngưỡng Otsu toàn cục thất bại do nền giấy trong bóng râm (<= 110)
      bị nhầm thành tiền cảnh (trắng 255), khiến các câu trong bóng đổ bị biến thành MULTIPLE_INVALID.
    - Chứng minh Adaptive Mean (Integral Image O(1)) triệt tiêu gradient chiếu sáng cục bộ,
      đạt độ chính xác nhận diện vượt trội >= 95%.
    """
    # Sinh phiếu cơ sở
    sheet_gray, gt_answers, gt_statuses, grid_rois = generate_synthetic_omr_sheet(
        width=800,
        height=1000,
        num_questions=20,
        add_shadow=False,
        add_noise=False,
        seed=303
    )

    # Thêm bóng đổ thực tế đè lên Cột 2 (Q11-Q20): mức xám nền giảm xuống ~95-105
    # (trong khi vùng sáng là ~240, vết chì là ~35)
    H, W = sheet_gray.shape
    shadow_mask = np.zeros((H, W), dtype=np.float64)
    # Bóng đổ che phủ nửa bên phải tờ phiếu (X >= 400)
    shadow_mask[:, 400:] = 135.0
    # Chuyển tiếp mượt (gradient) ở ranh giới
    for col in range(350, 400):
        shadow_mask[:, col] = (col - 350) / 50.0 * 135.0

    sheet_shadow = np.clip(sheet_gray.astype(np.float64) - shadow_mask, 25, 255).astype(np.uint8)

    # 1. Chạy với Otsu toàn cục
    pipe_otsu = TV3SegmentationPipeline(TV3PipelineConfig(threshold_method="otsu"))
    res_otsu = pipe_otsu.run(sheet_shadow, question_grid=grid_rois, run_opencv_benchmark=False)
    otsu_answers = res_otsu.get_answer_vector()
    otsu_correct = sum(1 for q_id in range(1, 21) if otsu_answers[q_id] == gt_answers[q_id])
    otsu_acc = (otsu_correct / 20.0) * 100.0

    # 2. Chạy với Adaptive Mean Thresholding (b = 25, C = 10.0)
    pipe_adapt = TV3SegmentationPipeline(TV3PipelineConfig(threshold_method="adaptive", block_size=25, C=10.0))
    res_adapt = pipe_adapt.run(sheet_shadow, question_grid=grid_rois, run_opencv_benchmark=False)
    adapt_answers = res_adapt.get_answer_vector()
    adapt_correct = sum(1 for q_id in range(1, 21) if adapt_answers[q_id] == gt_answers[q_id])
    adapt_acc = (adapt_correct / 20.0) * 100.0

    # Khẳng định khoa học:
    # Otsu thất bại trên các câu trong bóng râm (độ chính xác tụt giảm rõ rệt)
    # Adaptive Mean đạt độ chính xác cao áp đảo
    assert otsu_acc <= 65.0, f"Otsu phải bị ảnh hưởng bởi bóng đổ (thực tế: {otsu_acc}%)"
    assert adapt_acc >= 95.0, f"Adaptive Mean phải đạt >= 95% (thực tế: {adapt_acc}%)"
    assert adapt_acc > otsu_acc, f"Adaptive Mean ({adapt_acc}%) phải vượt trội so với Otsu ({otsu_acc}%)"


# ==============================================================================
# 4. KIỂM THỬ XỬ LÝ 4 TRẠNG THÁI (VALID, BLANK, ERASED_VALID, MULTIPLE_INVALID)
# ==============================================================================

def test_ds04_ds05_special_answer_cases(synthetic_ds03_shadow):
    """Kiểm tra logic phân loại chính xác các ca ngoại lệ: Bỏ trống, Tẩy xóa chọn lại, Tô đúp."""
    sheet_gray, gt_answers, gt_statuses, grid_rois = synthetic_ds03_shadow
    pipeline = TV3SegmentationPipeline(TV3PipelineConfig(threshold_method="adaptive"))
    result = pipeline.run(sheet_gray, question_grid=grid_rois, run_opencv_benchmark=False)

    q_dict = result.omr_result.questions

    # 1. Kiểm tra các câu bỏ trống (Q5, Q12)
    assert q_dict[5].status == STATUS_BLANK
    assert q_dict[5].selected is None
    assert q_dict[5].is_valid is False

    assert q_dict[12].status == STATUS_BLANK
    assert q_dict[12].selected is None
    assert q_dict[12].is_valid is False

    # 2. Kiểm tra câu tẩy xóa chọn lại (Q8, Q17)
    assert q_dict[8].status == STATUS_ERASED_VALID
    assert q_dict[8].selected == "C"
    assert q_dict[8].is_valid is True
    assert "tẩy xóa" in q_dict[8].warning.lower()

    assert q_dict[17].status == STATUS_ERASED_VALID
    assert q_dict[17].selected == "C"
    assert q_dict[17].is_valid is True

    # 3. Kiểm tra câu tô đúp phạm quy (Q15)
    assert q_dict[15].status == STATUS_MULTIPLE_INVALID
    assert q_dict[15].selected is None
    assert q_dict[15].is_valid is False
    assert "phạm quy" in q_dict[15].warning.lower()


# ==============================================================================
# 5. XÁC THỰC CHỈ TIÊU NGHIỆM THU OTSU (DELTA T <= 1 MỨC XÁM)
# ==============================================================================

def test_acceptance_criteria_otsu_delta_threshold():
    """Kiểm tra chỉ tiêu nghiệm thu Delta T_Otsu <= 1 mức xám so với cv2.THRESH_OTSU."""
    np.random.seed(42)
    # Tạo ma trận có phân bố hai đỉnh bimodal chuẩn
    c1 = np.random.normal(50, 15, (200, 200))
    c2 = np.random.normal(200, 20, (200, 200))
    img = np.clip(np.vstack([c1, c2]), 0, 255).astype(np.uint8)

    res_otsu = benchmark_otsu_vs_opencv(img, method="BINARY_INV")
    delta_t = res_otsu["delta_threshold"]

    assert delta_t <= 1.0, f"Độ lệch ngưỡng Otsu Delta T={delta_t} vượt quá chỉ tiêu <= 1.0"
    assert res_otsu["metrics"]["iou_percent"] >= 98.5


# ==============================================================================
# 6. XÁC THỰC CHỈ TIÊU NGHIỆM THU IOU VÀ PSNR TRÊN MORPHOLOGY VÀ ADAPTIVE
# ==============================================================================

def test_acceptance_criteria_morphology_and_adaptive_iou():
    """Kiểm tra chỉ tiêu IoU >= 98.5% và PSNR >= 48 dB trên ảnh thực nghiệm."""
    np.random.seed(77)
    test_img = np.random.randint(180, 240, (300, 300), dtype=np.uint8)
    # Thêm các ô tròn tối
    cv2.circle(test_img, (75, 75), 15, 30, -1)
    cv2.circle(test_img, (150, 150), 15, 40, -1)
    cv2.circle(test_img, (225, 225), 15, 50, -1)

    # 1. Adaptive Mean
    res_adapt = benchmark_adaptive_vs_opencv(test_img, block_size=25, C=10.0, method="BINARY_INV")
    assert res_adapt["metrics"]["iou_percent"] >= 98.5
    assert res_adapt["metrics"]["psnr_db"] >= 48.0 or res_adapt["metrics"]["psnr_db"] == 999.0

    # 2. Morphology Opening & Closing
    bin_img = res_adapt["binary_scratch"]
    kernel = create_structuring_element("RECT", (3, 3))
    kernel_cv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    opened_sc = opening_binary_scratch(bin_img, kernel)
    opened_cv = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_cv)
    metrics_open = calc_binary_metrics(opened_sc, opened_cv)
    assert metrics_open["iou_percent"] >= 99.0

    closed_sc = closing_binary_scratch(bin_img, kernel)
    closed_cv = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_cv)
    metrics_close = calc_binary_metrics(closed_sc, closed_cv)
    assert metrics_close["iou_percent"] >= 99.0


# ==============================================================================
# 7. XÁC THỰC CHỈ TIÊU NGHIỆM THU TWO-PASS CCL SO VỚI OPENCV
# ==============================================================================

def test_acceptance_criteria_ccl_match_opencv():
    """Kiểm tra thuật toán Two-Pass CCL Scratch khớp số lượng thành phần với OpenCV."""
    img_bin = np.zeros((200, 200), dtype=np.uint8)
    # Vẽ 5 ô rời rạc rõ ràng
    cv2.circle(img_bin, (40, 40), 12, 255, -1)
    cv2.circle(img_bin, (100, 40), 12, 255, -1)
    cv2.circle(img_bin, (160, 40), 12, 255, -1)
    cv2.circle(img_bin, (70, 120), 12, 255, -1)
    cv2.circle(img_bin, (130, 120), 12, 255, -1)

    k_sc, labels_sc, stats_sc = two_pass_ccl_scratch(img_bin, connectivity=8)
    num_cv, labels_cv, stats_cv, cents_cv = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

    assert k_sc == (num_cv - 1) == 5, f"Khớp số lượng component: Scratch={k_sc}, OpenCV={num_cv-1}"


# ==============================================================================
# 8. KIỂM THỬ HÀM TOP-LEVEL run_tv3_pipeline VÀ XUẤT FILE CSV/JSON
# ==============================================================================

def test_run_tv3_pipeline_and_file_export(tmp_path, synthetic_ds01_clean):
    """Kiểm thử hàm run_tv3_pipeline và các chức năng lưu báo cáo JSON/CSV."""
    sheet_gray, gt_answers, _, grid_rois = synthetic_ds01_clean

    res = run_tv3_pipeline(
        warped_gray=sheet_gray,
        threshold_method="adaptive",
        block_size=25,
        C=10.0,
        question_grid=grid_rois,
        run_opencv_benchmark=True
    )

    assert isinstance(res, TV3PipelineResult)
    assert len(res.omr_result.questions) == 20

    # Kiểm tra lưu CSV
    csv_file = str(tmp_path / "test_results.csv")
    res.save_csv(csv_file)
    assert os.path.exists(csv_file)
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 21  # 1 header + 20 rows
        assert reader[0][0] == "Question_ID"

    # Kiểm tra lưu JSON
    json_file = str(tmp_path / "test_results.json")
    res.save_json(json_file)
    assert os.path.exists(json_file)
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "answers_vector" in data
        assert "questions_detail" in data
        assert len(data["answers_vector"]) == 20

    # Kiểm tra trực quan hóa
    img_dir = str(tmp_path / "viz")
    saved_images = visualize_pipeline_steps(res, img_dir, gt_answers)
    assert len(saved_images) == 6
    for k, v in saved_images.items():
        assert os.path.exists(v)


# ==============================================================================
# 9. KIỂM THỬ CÁC MỤC CODE REVIEW (1.1, 1.3, 1.4, 2.1, 2.6, 3)
# ==============================================================================

def test_extract_bubble_answers_and_scratch_opencv_parity(synthetic_ds03_shadow):
    """
    Xác thực Mục 1.3 & 1.4:
    - extract_bubble_answers chạy không lỗi AttributeError.
    - Chế độ Scratch (use_scratch=True) và OpenCV (use_scratch=False) dùng chung 1 config
      và cho kết quả giống nhau 100% trên cả 20 câu.
    """
    sheet_gray, gt_answers, _, grid_rois = synthetic_ds03_shadow
    shared_cfg = OMRSharedConfig(
        block_size=25,
        C=10.0,
        bubble_radius=12.0,
        inner_radius_offset=0.0,
        tau_min=0.38,
        delta_tau_min=0.15,
    )

    ans_sc, ratios_sc, vis_sc = extract_bubble_answers(
        warped_gray=sheet_gray,
        total_questions=20,
        fill_threshold=0.38,
        use_scratch=True,
        question_grid=grid_rois,
        config=shared_cfg,
    )
    ans_cv, ratios_cv, vis_cv = extract_bubble_answers(
        warped_gray=sheet_gray,
        total_questions=20,
        fill_threshold=0.38,
        use_scratch=False,
        question_grid=grid_rois,
        config=shared_cfg,
    )

    assert len(ans_sc) == 20
    assert ans_sc == ans_cv
    assert vis_sc.shape == (1000, 800, 3)
    assert vis_cv.shape == (1000, 800, 3)

    for q_id in range(1, 21):
        expected = gt_answers[q_id]
        if expected is not None:
            assert ans_sc[q_id] == expected
        elif q_id == 15:
            assert ans_sc[q_id] == "MULTIPLE"
        else:
            assert ans_sc[q_id] == "BLANK"
        for opt in ("A", "B", "C", "D"):
            assert ratios_sc[q_id][opt] == pytest.approx(ratios_cv[q_id][opt], abs=1e-4)


def test_extract_bubble_answers_preserves_erased_valid_warnings(synthetic_ds03_shadow):
    """Xác thực Mục 2.1: Bảo toàn cảnh báo ERASED_VALID cho Tab 4."""
    sheet_gray, _, _, grid_rois = synthetic_ds03_shadow
    cfg = OMRSharedConfig(bubble_radius=12.0, inner_radius_offset=0.0)

    answers, _, _, details = extract_bubble_answers_with_warnings(
        warped_gray=sheet_gray,
        total_questions=20,
        fill_threshold=0.38,
        use_scratch=True,
        question_grid=grid_rois,
        config=cfg,
    )

    assert answers[8] == "C"
    assert answers[17] == "C"
    assert details["statuses"][8] == STATUS_ERASED_VALID
    assert details["statuses"][17] == STATUS_ERASED_VALID
    assert 8 in details["erased_questions"]
    assert 17 in details["erased_questions"]
    assert 8 in details["warnings"]
    assert 17 in details["warnings"]


def test_detect_bubble_grid_scratch_without_cv2_findcontours():
    """Xác thực Mục 1.1: Dò lưới ô tròn hoàn toàn bằng Scratch Adaptive + Two-Pass CCL."""
    img = np.full((1000, 800), 240, dtype=np.uint8)
    grid_gt = {}
    for q in range(1, 21):
        is_col2 = q > 10
        row = (q - 1) % 10
        bx = 470.0 if is_col2 else 110.0
        by = 170.0 + row * 70.0
        grid_gt[q] = {}
        for opt_idx, opt in enumerate(("A", "B", "C", "D")):
            cx = bx + opt_idx * 55.0
            cy = by
            grid_gt[q][opt] = (cx, cy)
            cv2.circle(img, (int(cx), int(cy)), 12, 35, -1)

    detected_grid, stats = detect_bubble_grid_scratch(img, num_questions=20, use_scratch=True)
    assert len(stats) == 80
    assert len(detected_grid) == 20
    for q in range(1, 21):
        for opt in ("A", "B", "C", "D"):
            gx, gy = grid_gt[q][opt]
            dx, dy = detected_grid[q][opt]
            assert abs(gx - dx) < 2.0 and abs(gy - dy) < 2.0


def test_no_cv2_import_in_core_modules_and_tv3_package_import():
    """Xác thực Mục 2.6 & Mục 3: Core modules không import cv2 ở top-level và import được qua package `tv3`."""
    import subprocess

    code = (
        "import sys; "
        "sys.modules['cv2'] = None; "
        "import tv3; "
        "from tv3.tv3_thresholding import otsu_threshold_scratch, adaptive_mean_threshold_scratch; "
        "from tv3.tv3_morphology_ccl import two_pass_ccl_scratch, opening_binary_scratch; "
        "from tv3.tv3_roi_classifier import recognize_all_questions; "
        "from tv3.tv3_segmentation import TV3SegmentationPipeline; "
        "import numpy as np; "
        "img = np.full((60, 60), 220, dtype=np.uint8); "
        "img[20:35, 20:35] = 40; "
        "pipe = TV3SegmentationPipeline(); "
        "res = pipe.run(img, run_opencv_benchmark=False); "
        "assert res.execution_mode == 'scratch'; "
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "OK"


@pytest.mark.parametrize("ksize", [(3, 5), (5, 3)])
@pytest.mark.parametrize("kshape", ["RECT", "CROSS", "ELLIPSE"])
def test_non_square_kernel_pipeline_and_extract_parity(synthetic_ds01_clean, ksize, kshape):
    """Kiểm chứng quy ước morph_kernel_size=(kh, kw) khớp 100% giữa Scratch và OpenCV khi kernel không vuông."""
    sheet_gray, _, _, grid_rois = synthetic_ds01_clean
    cfg = TV3PipelineConfig(
        threshold_method="adaptive",
        block_size=25,
        C=10.0,
        morph_kernel_shape=kshape,
        morph_kernel_size=ksize,
        bubble_radius=12.0,
        inner_radius_offset=0.0,
        tau_min=0.38,
        delta_tau_min=0.15,
    )

    pipeline = TV3SegmentationPipeline(cfg)
    assert pipeline.structuring_element.shape == ksize

    res_sc = pipeline.run(sheet_gray, question_grid=grid_rois, run_opencv_benchmark=True, execution_mode="scratch")
    res_cv = pipeline.run(sheet_gray, question_grid=grid_rois, run_opencv_benchmark=False, execution_mode="opencv")

    bench = res_sc.benchmark_opencv
    assert bench is not None
    assert bench["morphology_metrics"]["iou_percent"] == 100.0
    assert bench["morphology_metrics"]["mse"] == 0.0

    direct_metrics = calc_binary_metrics(res_sc.binary_morph, res_cv.binary_morph)
    assert direct_metrics["diff_pixels"] == 0
    assert direct_metrics["iou_percent"] == 100.0

    ans_sc, ratios_sc, _ = extract_bubble_answers(
        warped_gray=sheet_gray,
        total_questions=20,
        fill_threshold=0.38,
        use_scratch=True,
        question_grid=grid_rois,
        config=cfg,
    )
    ans_cv, ratios_cv, _ = extract_bubble_answers(
        warped_gray=sheet_gray,
        total_questions=20,
        fill_threshold=0.38,
        use_scratch=False,
        question_grid=grid_rois,
        config=cfg,
    )
    assert ans_sc == ans_cv
    for q_id in range(1, 21):
        for opt in ("A", "B", "C", "D"):
            assert ratios_sc[q_id][opt] == pytest.approx(ratios_cv[q_id][opt], abs=1e-6)


def test_pipeline_skips_redundant_ccl_when_question_grid_provided(monkeypatch, synthetic_ds01_clean):
    """
    Xác thực Mục 2.2 (Code Review lần 3):
    - Khi truyền sẵn `question_grid` và `run_opencv_benchmark=False`, `TV3SegmentationPipeline.run`
      tự động bỏ qua Two-Pass CCL (không gọi `two_pass_ccl_scratch`, `ccl_ms == 0.0`) mà vẫn
      nhận diện đúng 100% đáp án.
    - Khi truyền tường minh `run_ccl=False` (kể cả không có `question_grid` hoặc bật benchmark),
      CCL cũng được bỏ qua.
    - Khi truyền tường minh `run_ccl=True`, CCL vẫn chạy bình thường.
    """
    import src.tv3_segmentation as tv3_seg

    sheet_gray, gt_answers, _, grid_rois = synthetic_ds01_clean
    ccl_calls = {"count": 0}
    orig_ccl = tv3_seg.two_pass_ccl_scratch

    def spy_ccl(*args, **kwargs):
        ccl_calls["count"] += 1
        return orig_ccl(*args, **kwargs)

    monkeypatch.setattr(tv3_seg, "two_pass_ccl_scratch", spy_ccl)
    pipeline = TV3SegmentationPipeline(TV3PipelineConfig(threshold_method="adaptive"))

    # 1. Mặc định (run_ccl=None) + có sẵn question_grid + run_opencv_benchmark=False -> bỏ qua CCL
    res_skip = pipeline.run(sheet_gray, question_grid=grid_rois, run_opencv_benchmark=False)
    assert ccl_calls["count"] == 0
    assert res_skip.latency_ms["ccl_ms"] == 0.0
    assert res_skip.num_components == 0
    assert res_skip.raw_stats == []
    assert res_skip.filtered_bubble_stats == []
    assert res_skip.labels.shape == sheet_gray.shape
    assert np.all(res_skip.labels == 0)
    assert res_skip.get_answer_vector() == gt_answers

    # 2. Chế độ OpenCV với question_grid và run_opencv_benchmark=False -> cũng bỏ qua CCL
    res_skip_cv = pipeline.run(
        sheet_gray,
        question_grid=grid_rois,
        run_opencv_benchmark=False,
        execution_mode="opencv",
    )
    assert res_skip_cv.latency_ms["ccl_ms"] == 0.0
    assert res_skip_cv.num_components == 0
    assert res_skip_cv.get_answer_vector() == gt_answers

    # 3. Truyền tường minh run_ccl=False khi bật benchmark hoặc không truyền grid -> vẫn bỏ qua CCL
    res_force_off = pipeline.run(sheet_gray, question_grid=grid_rois, run_opencv_benchmark=True, run_ccl=False)
    assert ccl_calls["count"] == 0
    assert res_force_off.latency_ms["ccl_ms"] == 0.0
    assert res_force_off.benchmark_opencv is not None
    assert res_force_off.benchmark_opencv["ccl_count_match"] is True

    # 4. Truyền tường minh run_ccl=True dù có sẵn question_grid -> chạy CCL bình thường
    res_force_on = pipeline.run(sheet_gray, question_grid=grid_rois, run_opencv_benchmark=False, run_ccl=True)
    assert ccl_calls["count"] == 1
    assert res_force_on.num_components > 0
    assert len(res_force_on.raw_stats) == res_force_on.num_components
    assert res_force_on.get_answer_vector() == gt_answers


