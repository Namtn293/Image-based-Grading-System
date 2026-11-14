"""Unit tests cho Module Thành viên 3: Trích xuất ROI và phân loại ô đáp án OMR."""

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

# Thêm thư mục src vào sys.path để import
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from tv3_roi_classifier import (
    DEFAULT_DELTA_TAU_MIN,
    DEFAULT_TAU_MIN,
    STATUS_BLANK,
    STATUS_ERASED_VALID,
    STATUS_MULTIPLE_INVALID,
    STATUS_VALID,
    auto_align_grid_from_centroids,
    classify_question_bubbles,
    compute_bubble_fill_ratio_scratch,
    create_circular_mask_scratch,
    generate_grid_bubble_rois,
    recognize_all_questions,
)


def test_create_circular_mask_scratch():
    """Kiểm tra tạo mặt nạ hình tròn thuần NumPy."""
    shape = (50, 50)
    center = (25.0, 25.0)
    radius = 10.0
    mask = create_circular_mask_scratch(shape, center, radius)

    assert mask.shape == shape
    assert mask.dtype == bool
    # Tâm chắc chắn phải thuộc mặt nạ
    assert mask[25, 25] is np.True_
    # Điểm ngoài bán kính phải là False
    assert mask[0, 0] is np.False_
    assert mask[25, 45] is np.False_

    # Diện tích lý thuyết ~ pi * R^2 = 3.14159 * 100 ~ 314
    area = np.sum(mask)
    theoretical_area = math.pi * (radius ** 2)
    assert abs(area - theoretical_area) / theoretical_area < 0.05


def test_compute_bubble_fill_ratio_scratch():
    """Kiểm tra tính tỷ lệ lấp đầy (Fill Ratio) trên ảnh nhị phân mô phỏng."""
    img = np.zeros((100, 100), dtype=np.uint8)
    center = (50.0, 50.0)
    radius = 12.0

    # 1. Ảnh trắng hoàn toàn -> Fill ratio = 0.0
    ratio_empty = compute_bubble_fill_ratio_scratch(img, center, radius)
    assert ratio_empty == 0.0

    # 2. Vẽ hình tròn kín (tô đậm 100% bằng giá trị 255)
    y, x = np.ogrid[:100, :100]
    circle_mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= (radius ** 2)
    img[circle_mask] = 255

    ratio_full_patch = compute_bubble_fill_ratio_scratch(img, center, radius, use_local_patch=True)
    ratio_full_global = compute_bubble_fill_ratio_scratch(img, center, radius, use_local_patch=False)

    assert abs(ratio_full_patch - 1.0) < 1e-4
    assert abs(ratio_full_global - 1.0) < 1e-4

    # 3. Tô một nửa hình tròn (50%)
    img_half = np.zeros((100, 100), dtype=np.uint8)
    half_mask = circle_mask & (x >= center[0])
    img_half[half_mask] = 255

    ratio_half = compute_bubble_fill_ratio_scratch(img_half, center, radius)
    assert abs(ratio_half - 0.5) < 0.05


def test_classify_question_bubbles_four_scenarios():
    """Kiểm thử 4 kịch bản quy chuẩn:
    1. VALID: Tô chuẩn từng đáp án A/B/C/D
    2. BLANK: Bỏ trống hoàn toàn
    3. ERASED_VALID: Tẩy xóa chọn lại (tau_1 = 0.65, tau_2 = 0.42 => Delta_tau = 0.23 >= 0.15)
    4. MULTIPLE_INVALID: Tô đúp phạm quy (tau_1 = 0.62, tau_2 = 0.55 => Delta_tau = 0.07 < 0.15)
    """
    # Kịch bản 1: Tô chuẩn đáp án 'B'
    ratios_valid = {"A": 0.05, "B": 0.78, "C": 0.08, "D": 0.02}
    res_valid = classify_question_bubbles(ratios_valid, question_id=1)
    assert res_valid.status == STATUS_VALID
    assert res_valid.selected == "B"
    assert res_valid.is_valid is True
    assert res_valid.warning is None

    # Kiểm tra tô chuẩn lần lượt cho A, C, D
    for target in ["A", "C", "D"]:
        ratios = {opt: (0.75 if opt == target else 0.04) for opt in ["A", "B", "C", "D"]}
        res = classify_question_bubbles(ratios, question_id=1)
        assert res.status == STATUS_VALID
        assert res.selected == target

    # Kịch bản 2: Bỏ trống (tất cả đều < tau_min = 0.38)
    ratios_blank = {"A": 0.05, "B": 0.12, "C": 0.10, "D": 0.02}
    res_blank = classify_question_bubbles(ratios_blank, question_id=2)
    assert res_blank.status == STATUS_BLANK
    assert res_blank.selected is None
    assert res_blank.is_valid is False

    # Kịch bản 3: Tẩy xóa chọn lại (tau_1 = 0.65, tau_2 = 0.42 => Delta_tau = 0.23 >= 0.15)
    # Thí sinh ban đầu tô C (0.42), sau đó tẩy đi và tô đậm lại A (0.65)
    ratios_erased = {"A": 0.65, "B": 0.05, "C": 0.42, "D": 0.08}
    res_erased = classify_question_bubbles(ratios_erased, question_id=3)
    assert res_erased.status == STATUS_ERASED_VALID
    assert res_erased.selected == "A"
    assert res_erased.is_valid is True
    assert res_erased.delta_tau == pytest.approx(0.23, abs=1e-3)
    assert res_erased.warning is not None
    assert "tẩy xóa" in res_erased.warning

    # Kịch bản 4: Tô đúp phạm quy (tau_1 = 0.62, tau_2 = 0.55 => Delta_tau = 0.07 < 0.15)
    # Thí sinh tô cả A và D mà không tẩy sạch
    ratios_mult = {"A": 0.62, "B": 0.05, "C": 0.06, "D": 0.55}
    res_mult = classify_question_bubbles(ratios_mult, question_id=4)
    assert res_mult.status == STATUS_MULTIPLE_INVALID
    assert res_mult.selected is None
    assert res_mult.is_valid is False
    assert res_mult.delta_tau == pytest.approx(0.07, abs=1e-3)
    assert res_mult.warning is not None
    assert "Phạm quy" in res_mult.warning


def test_generate_grid_bubble_rois():
    """Kiểm tra tạo lưới tọa độ ROI 1 cột và 2 cột."""
    # 1 Cột: 20 câu hỏi
    grid_1col = generate_grid_bubble_rois(
        start_x=100.0,
        start_y=150.0,
        step_x=30.0,
        step_y=25.0,
        num_questions=20,
        num_columns=1,
    )
    assert len(grid_1col) == 20
    assert grid_1col[1]["A"] == (100.0, 150.0)
    assert grid_1col[1]["B"] == (130.0, 150.0)
    assert grid_1col[1]["D"] == (190.0, 150.0)
    assert grid_1col[2]["A"] == (100.0, 175.0)
    assert grid_1col[20]["A"] == (100.0, 150.0 + 19 * 25.0)

    # 2 Cột: 20 câu (1-10 cột 1, 11-20 cột 2)
    grid_2col = generate_grid_bubble_rois(
        start_x=100.0,
        start_y=150.0,
        step_x=30.0,
        step_y=25.0,
        num_questions=20,
        num_columns=2,
        col_spacing_x=250.0,
    )
    assert len(grid_2col) == 20
    assert grid_2col[10]["A"] == (100.0, 150.0 + 9 * 25.0)
    # Câu 11 sang cột 2
    assert grid_2col[11]["A"] == (100.0 + 250.0, 150.0)
    assert grid_2col[20]["A"] == (100.0 + 250.0, 150.0 + 9 * 25.0)


def test_auto_align_grid_from_centroids():
    """Kiểm thử gom nhóm lưới tọa độ tự động từ danh sách centroids (có nhiễu tọa độ)."""
    # Tạo danh sách centroids chuẩn 5 câu hỏi x 4 lựa chọn = 20 điểm
    np.random.seed(42)
    centroids = []
    base_x, base_y = 50.0, 80.0
    step_x, step_y = 35.0, 40.0

    for q in range(5):
        for opt in range(4):
            # Thêm nhiễu ngẫu nhiên nhẹ +/- 1 pixel
            noise_x = np.random.uniform(-1.0, 1.0)
            noise_y = np.random.uniform(-1.0, 1.0)
            cx = base_x + opt * step_x + noise_x
            cy = base_y + q * step_y + noise_y
            centroids.append((cx, cy))

    # Xáo trộn ngẫu nhiên thứ tự các tâm như đầu ra thực tế của CCL
    np.random.shuffle(centroids)

    grid = auto_align_grid_from_centroids(centroids, num_questions=5, y_cluster_tol=15.0)
    assert len(grid) == 5

    for q_idx in range(1, 6):
        assert set(grid[q_idx].keys()) == {"A", "B", "C", "D"}
        # Tọa độ x phải tăng dần từ A -> B -> C -> D
        assert grid[q_idx]["A"][0] < grid[q_idx]["B"][0] < grid[q_idx]["C"][0] < grid[q_idx]["D"][0]
        # Tọa độ y của các câu phải tăng dần
        if q_idx > 1:
            assert grid[q_idx]["A"][1] > grid[q_idx - 1]["A"][1]


def test_recognize_all_questions_end_to_end():
    """Kiểm thử tổng hợp quy trình nhận diện toàn bộ 20 câu hỏi OMR trên ảnh mô phỏng."""
    h, w = 600, 400
    img = np.zeros((h, w), dtype=np.uint8)

    # Sinh lưới 20 câu hỏi
    grid = generate_grid_bubble_rois(
        start_x=80.0,
        start_y=50.0,
        step_x=40.0,
        step_y=25.0,
        num_questions=20,
    )
    radius = 10.0

    def draw_bubble(center: Tuple[float, float], fill_rate: float):
        """Vẽ ô tròn với tỷ lệ lấp đầy fill_rate."""
        cx, cy = center
        y, x = np.ogrid[:h, :w]
        c_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (radius ** 2)
        if fill_rate >= 1.0:
            img[c_mask] = 255
        elif fill_rate > 0.0:
            # Chọn ngẫu nhiên số pixel ứng với fill_rate
            pts_y, pts_x = np.where(c_mask)
            num_fill = int(len(pts_y) * fill_rate)
            selected_idx = np.random.choice(len(pts_y), num_fill, replace=False)
            img[pts_y[selected_idx], pts_x[selected_idx]] = 255

    np.random.seed(123)

    # Thiết kế đáp án thực tế:
    # - Câu 1 -> 16: VALID (tô chuẩn luân phiên A, B, C, D)
    # - Câu 17: BLANK (bỏ trống hoàn toàn)
    # - Câu 18: ERASED_VALID (tô A 0.85, tẩy B 0.40 => delta = 0.45 >= 0.15 => chọn A)
    # - Câu 19: MULTIPLE_INVALID (tô C 0.70 và D 0.65 => delta = 0.05 < 0.15 => phạm quy)
    # - Câu 20: VALID (tô D)

    expected_answers = {}
    expected_status = {}

    options_cycle = ["A", "B", "C", "D"]
    for q in range(1, 17):
        target_opt = options_cycle[(q - 1) % 4]
        draw_bubble(grid[q][target_opt], 0.85)
        # Các ô khác có chút nhiễu nhẹ 5%
        for opt in options_cycle:
            if opt != target_opt:
                draw_bubble(grid[q][opt], 0.05)
        expected_answers[q] = target_opt
        expected_status[q] = STATUS_VALID

    # Câu 17: BLANK
    for opt in options_cycle:
        draw_bubble(grid[17][opt], 0.08)
    expected_answers[17] = None
    expected_status[17] = STATUS_BLANK

    # Câu 18: ERASED_VALID (A đậm 0.85, B vết mờ 0.40)
    draw_bubble(grid[18]["A"], 0.85)
    draw_bubble(grid[18]["B"], 0.40)
    expected_answers[18] = "A"
    expected_status[18] = STATUS_ERASED_VALID

    # Câu 19: MULTIPLE_INVALID (C 0.70, D 0.65)
    draw_bubble(grid[19]["C"], 0.70)
    draw_bubble(grid[19]["D"], 0.65)
    expected_answers[19] = None
    expected_status[19] = STATUS_MULTIPLE_INVALID

    # Câu 20: VALID (D 0.80)
    draw_bubble(grid[20]["D"], 0.80)
    expected_answers[20] = "D"
    expected_status[20] = STATUS_VALID

    # Thực hiện nhận diện toàn bộ
    result = recognize_all_questions(
        binary_image=img,
        question_grid=grid,
        radius=radius,
        tau_min=0.38,
        delta_tau_min=0.15,
    )

    # Xác thực kết quả
    for q in range(1, 21):
        q_res = result.questions[q]
        assert q_res.status == expected_status[q], f"Sai trạng thái câu {q}: {q_res.status} != {expected_status[q]}"
        assert q_res.selected == expected_answers[q], f"Sai đáp án câu {q}: {q_res.selected} != {expected_answers[q]}"

    # Kiểm tra summary
    assert result.summary[STATUS_VALID] == 17  # 16 + câu 20
    assert result.summary[STATUS_BLANK] == 1
    assert result.summary[STATUS_ERASED_VALID] == 1
    assert result.summary[STATUS_MULTIPLE_INVALID] == 1

    # Kiểm tra chuyển đổi to_dict
    res_dict = result.to_dict()
    assert "answers_vector" in res_dict
    assert "summary" in res_dict
    assert "questions" in res_dict
    assert res_dict["answers_vector"][18] == "A"
    assert res_dict["answers_vector"][19] is None


def test_edge_cases_and_boundary_safety():
    """Kiểm tra an toàn biên và các trường hợp đặc biệt."""
    # 1. Tâm nằm sát biên hoặc ra ngoài biên ảnh
    small_img = np.ones((20, 20), dtype=np.uint8) * 255
    ratio_corner = compute_bubble_fill_ratio_scratch(small_img, center=(0.0, 0.0), radius=5.0)
    assert 0.0 <= ratio_corner <= 1.0

    # 2. Bán kính <= 0
    ratio_zero_r = compute_bubble_fill_ratio_scratch(small_img, center=(10.0, 10.0), radius=0.0)
    assert ratio_zero_r == 0.0

    # 3. Phân loại với danh sách rỗng
    res_empty = classify_question_bubbles({})
    assert res_empty.status == STATUS_BLANK

    # 4. Ảnh đầu vào kiểu bool
    bool_img = np.zeros((30, 30), dtype=bool)
    bool_img[10:20, 10:20] = True
    ratio_bool = compute_bubble_fill_ratio_scratch(bool_img, center=(15.0, 15.0), radius=5.0)
    assert ratio_bool > 0.8

