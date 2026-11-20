"""Comprehensive Unit Tests for Module TV4:
Histogram Equalization, Median Filter 2D, Robust CSV I/O, and Grading Logic.
Production Automated Test Suite.
"""

import os
import io
import subprocess
import unittest
import numpy as np
import pandas as pd
import cv2
import openpyxl

import sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from tv4_enhancement_eval import (
        histogram_equalization_scratch,
        median_filter_scratch,
        create_anchor_marker_template,
        _pad_2d_scratch,
        save_matrix_to_csv,
        load_matrix_from_csv,
        calculate_mse,
        calculate_psnr,
        calculate_difference_map,
        calculate_binary_iou_dice,
        calculate_ccl_component_iou,
        calculate_otsu_delta_t,
        calculate_corner_error,
        calculate_mark_accuracy,
        benchmark_hist_equalization,
        benchmark_median_filter,
        benchmark_all_modules,
        run_ablation_study,
        OMRGrader
    )
    from pipeline_coordinator import (
        OMRPipeline,
        _resolve_bubble_grid,
        detect_sheet_questions,
        create_anchor_marker_template as coord_anchor_template,
    )
except ImportError:
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
from tv4_enhancement_eval import (
    histogram_equalization_scratch,
    median_filter_scratch,
    create_anchor_marker_template,
    _pad_2d_scratch,
    save_matrix_to_csv,
    load_matrix_from_csv,
    calculate_mse,
    calculate_psnr,
    calculate_difference_map,
    calculate_binary_iou_dice,
    calculate_ccl_component_iou,
    calculate_otsu_delta_t,
    calculate_corner_error,
    calculate_mark_accuracy,
    benchmark_hist_equalization,
    benchmark_median_filter,
    benchmark_all_modules,
    run_ablation_study,
    OMRGrader
)
from pipeline_coordinator import (
    OMRPipeline,
    _resolve_bubble_grid,
    detect_sheet_questions,
    create_anchor_marker_template as coord_anchor_template,
)


class TestTV4Algorithms(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.test_img = np.random.randint(40, 180, size=(120, 150), dtype=np.uint8)
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            for f in os.listdir(self.tmp_dir):
                try:
                    os.remove(os.path.join(self.tmp_dir, f))
                except Exception:
                    pass

    def test_histogram_equalization_shape_and_type(self):
        """Test output shape and dtype for Histogram Equalization."""
        eq = histogram_equalization_scratch(self.test_img)
        self.assertEqual(eq.shape, self.test_img.shape)
        self.assertEqual(eq.dtype, np.uint8)
        self.assertGreaterEqual(eq.min(), 0)
        self.assertLessEqual(eq.max(), 255)

    def test_histogram_equalization_vs_opencv(self):
        """Verify mathematical equivalence between Scratch and OpenCV across random images."""
        for seed in [42, 100, 2026]:
            np.random.seed(seed)
            img = np.random.randint(20, 240, size=(100, 120), dtype=np.uint8)
            eq_scratch = histogram_equalization_scratch(img)
            eq_cv = cv2.equalizeHist(img)

            mse = calculate_mse(eq_scratch, eq_cv)
            psnr = calculate_psnr(eq_scratch, eq_cv)

            self.assertEqual(mse, 0.0, f"MSE should be 0.0, got {mse} for seed {seed}")
            self.assertEqual(psnr, 100.0)

        # Also test exact half-integer (.5) banker's rounding parity with OpenCV
        half_case = np.array([[0, 10, 20, 30, 40, 50, 60]], dtype=np.uint8)
        np.testing.assert_array_equal(
            histogram_equalization_scratch(half_case),
            cv2.equalizeHist(half_case)
        )

    def test_pad_2d_scratch_reflect_and_edge(self):
        """Verify manual _pad_2d_scratch matches np.pad for reflect and edge modes."""
        mat = np.arange(35, dtype=np.uint8).reshape(5, 7)
        for p in [1, 2, 6]:
            np.testing.assert_array_equal(_pad_2d_scratch(mat, p, "edge"), np.pad(mat, p, "edge"))
            np.testing.assert_array_equal(_pad_2d_scratch(mat, p, "reflect"), np.pad(mat, p, "reflect"))

    def test_histogram_equalization_uniform_images(self):
        """Verify edge case: uniform images (all black, all white, all 128)."""
        for val in [0, 50, 128, 200, 255]:
            uniform = np.full((50, 50), val, dtype=np.uint8)
            eq_scratch = histogram_equalization_scratch(uniform)
            eq_cv = cv2.equalizeHist(uniform)
            np.testing.assert_array_equal(eq_scratch, eq_cv)

    def test_median_filter_noise_removal(self):
        """Verify Median Filter effectively removes salt-and-pepper noise."""
        noisy = self.test_img.copy()
        noisy[10, 10] = 255  # salt
        noisy[11, 11] = 0    # pepper

        filtered = median_filter_scratch(noisy, kernel_size=3)
        self.assertNotEqual(filtered[10, 10], 255)
        self.assertNotEqual(filtered[11, 11], 0)
        self.assertEqual(filtered.shape, noisy.shape)
        self.assertEqual(filtered.dtype, np.uint8)

    def test_median_filter_vs_opencv_various_kernels(self):
        """Verify Median Filter matches OpenCV for multiple odd kernel sizes."""
        for k in [3, 5, 7]:
            med_scratch = median_filter_scratch(self.test_img, kernel_size=k, padding_mode="edge")
            med_cv = cv2.medianBlur(self.test_img, k)

            mse = calculate_mse(med_scratch, med_cv)
            self.assertEqual(mse, 0.0, f"Kernel {k} mismatch: MSE={mse}")

    def test_robust_csv_matrix_io(self):
        """Test matrix save to CSV and load back, including messy CSV with trailing commas."""
        # 1. Clean roundtrip
        csv_path = os.path.join(self.tmp_dir, "test_mat.csv")
        save_matrix_to_csv(self.test_img, csv_path)
        loaded = load_matrix_from_csv(csv_path)
        np.testing.assert_array_equal(self.test_img, loaded)

        # 2. Messy CSV with trailing commas, empty lines, and semicolons
        messy_str = io.StringIO(
            "# Instructor verification matrix dump\n"
            "10, 20, 30,\n"
            "40; 50; 60;\n\n"
            "70, 80, 90,\n"
        )
        loaded_messy = load_matrix_from_csv(messy_str)
        expected = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        np.testing.assert_array_equal(loaded_messy, expected)

    def test_omr_grader_logic(self):
        """Test grading logic on 10-point scale, blank and multiple marks."""
        grader = OMRGrader()
        key = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'A'}
        grader.set_answer_key_dict(key)

        student_answers = {
            1: 'A',          # Correct
            2: 'B',          # Correct
            3: 'D',          # Incorrect (Key is C)
            4: 'BLANK',      # Blank
            5: 'MULTIPLE'    # Multiple
        }

        res = grader.grade_student("SV001", student_answers)
        self.assertEqual(res["total_questions"], 5)
        self.assertEqual(res["correct_count"], 2)
        self.assertEqual(res["incorrect_count"], 1)
        self.assertEqual(res["blank_count"], 1)
        self.assertEqual(res["multiple_count"], 1)
        # Score: 2/5 * 10 = 4.0
        self.assertEqual(res["score_10"], 4.0)

        # Also test list/tuple mock inputs and comma-separated multiple answers
        mock_answers = {
            1: ['A'],        # Single element list -> 'A' (Correct)
            2: ['A', 'B'],   # Multi element list -> 'MULTIPLE'
            3: [],           # Empty list -> 'BLANK'
            4: 'A,D',        # Comma-separated -> 'MULTIPLE'
            5: None          # None -> 'BLANK'
        }
        res_mock = grader.grade_student("SV_MOCK", mock_answers)
        self.assertEqual(res_mock["correct_count"], 1)
        self.assertEqual(res_mock["multiple_count"], 2)
        self.assertEqual(res_mock["blank_count"], 2)
        self.assertEqual(res_mock["score_10"], 2.0)

        # Verify NaN / duplicate lists / string question keys are handled accurately
        edge_answers = {
            "1": ["A", "A"],     # duplicate list -> single 'A' (Correct)
            2: float("nan"),     # NaN -> BLANK (not MULTIPLE)
            3: "N/A",            # N/A -> BLANK
            4: "A,C",            # multiple -> MULTIPLE
            5: "INVALID"         # invalid -> MULTIPLE
        }
        res_edge = grader.grade_student("SV_EDGE", edge_answers)
        self.assertEqual(res_edge["correct_count"], 1)
        self.assertEqual(res_edge["blank_count"], 2)
        self.assertEqual(res_edge["multiple_count"], 2)

        acc = calculate_mark_accuracy({1: ["A"], 2: ["A", "B"]}, {1: "A", 2: "MULTIPLE"})
        self.assertEqual(acc, 100.0)

    def test_benchmark_all_modules_and_ablation(self):
        """Verify cross-module benchmark_all_modules and run_ablation_study execute cleanly."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sheet = cv2.imread(os.path.join(base_dir, "data", "sample_sheet.png"))
        self.assertIsNotNone(sheet)
        rows = benchmark_all_modules(sheet)
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            self.assertIn("mse", row)
            self.assertIn("psnr_db", row)
            if row["iou"] is not None:
                # Binary segmentation / edge masks evaluated with IoU >= 98.5% (Report Section 7.1)
                self.assertGreaterEqual(row["iou"], 0.985, f"IoU too low for {row['algorithm']}: {row['iou']}")
            else:
                # Continuous intensity transforms evaluated with MSE < 1.0 and PSNR > 48 dB
                self.assertLess(row["mse"], 1.0, f"MSE too high for {row['algorithm']}: {row['mse']}")
                self.assertGreater(row["psnr_db"], 48.0, f"PSNR too low for {row['algorithm']}: {row['psnr_db']}")

        # Verify ablation study on DS-03 Shadow shows Adaptive Mean outperforms Global Otsu
        shadow_path = os.path.join(base_dir, "data", "samples", "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
        shadow_gray = cv2.imread(shadow_path, cv2.IMREAD_GRAYSCALE)
        gt5 = {
            1: 'A', 2: 'C', 3: 'B', 4: 'D', 5: 'A', 6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'C',
            11: 'B', 12: 'D', 13: 'A', 14: 'C', 15: 'B', 16: 'D', 17: 'A', 18: 'B', 19: 'C', 20: 'C'
        }
        ab = run_ablation_study(shadow_gray, gt5)
        self.assertGreaterEqual(ab["adaptive_accuracy_pct"], 90.0)
        self.assertLess(ab["otsu_accuracy_pct"], ab["adaptive_accuracy_pct"])
        self.assertEqual(len(ab["threshold_survey"]), 6)

    def test_binary_iou_and_dice_metric(self):
        """Test IoU and Dice score calculation on binary masks."""
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m1[2:6, 2:6] = 255
        m2[2:6, 2:6] = 255
        iou, dice = calculate_binary_iou_dice(m1, m2)
        self.assertEqual(iou, 1.0)
        self.assertEqual(dice, 1.0)

    def test_excel_report_generation(self):
        """Test exporting grading batch results to Excel and verify openpyxl styling."""
        grader = OMRGrader()
        key = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        grader.set_answer_key_dict(key)

        batch = {
            "SV001": {1: 'A', 2: 'B', 3: 'C', 4: 'D'},  # 10.0
            "SV002": {1: 'A', 2: 'A', 3: 'C', 4: 'MULTIPLE'},  # 5.0
        }
        graded_batch = grader.grade_batch(batch)
        excel_path = os.path.join(self.tmp_dir, "test_report.xlsx")

        OMRGrader.generate_excel_report(graded_batch, excel_path)
        self.assertTrue(os.path.exists(excel_path))

        wb = openpyxl.load_workbook(excel_path)
        self.assertIn("Bang_Diem_Tong_Hop", wb.sheetnames)
        self.assertIn("Thong_Ke_Cau_Hoi", wb.sheetnames)

        ws1 = wb["Bang_Diem_Tong_Hop"]
        # Check header font is bold and has fill
        self.assertTrue(ws1["A1"].font.bold)
        self.assertIsNotNone(ws1["A1"].fill.start_color.rgb)
        wb.close()

    def test_corner_error_and_delta_t_metrics(self):
        """Verify corner localization Euclidean error and Otsu delta T calculations."""
        pred_corners = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        gt_corners = np.array([[11, 10], [90, 12], [89, 90], [10, 90]])
        # Displacements: sqrt(1^2+0)=1, sqrt(0+2^2)=2, sqrt((-1)^2+0)=1, 0 -> mean = 4/4 = 1.0
        err = calculate_corner_error(pred_corners, gt_corners)
        self.assertEqual(err, 1.0)

        delta_t = calculate_otsu_delta_t(120.5, 122.0)
        self.assertEqual(delta_t, 1.5)

    def test_hist_equalization_textbook_mode_and_diff_map(self):
        """Verify textbook use_min_cdf=False and calculate_difference_map."""
        mat = np.array([[50, 50, 100, 150]], dtype=np.uint8)
        eq_textbook = histogram_equalization_scratch(mat, use_min_cdf=False)
        self.assertEqual(eq_textbook.shape, mat.shape)
        self.assertEqual(eq_textbook.dtype, np.uint8)

        # Difference map amplified by 10
        m1 = np.full((10, 10), 100, dtype=np.uint8)
        m2 = np.full((10, 10), 105, dtype=np.uint8)
        diff = calculate_difference_map(m1, m2, amplify=10.0)
        self.assertEqual(diff[0, 0], 50)

    def test_benchmarks_with_singleton_channel_dimension(self):
        """Verify benchmark functions do not crash on (H, W, 1) image arrays."""
        img_hw1 = np.random.randint(40, 180, size=(60, 60, 1), dtype=np.uint8)
        bm_he = benchmark_hist_equalization(img_hw1, iterations=1)
        self.assertIn("mse", bm_he)
        self.assertEqual(bm_he["amplify"], 15.0)

        bm_mf = benchmark_median_filter(img_hw1, kernel_size=3, iterations=1)
        self.assertIn("mse", bm_mf)
        self.assertEqual(bm_mf["amplify"], 10.0)

    def test_pipeline_e2e_scenarios(self):
        """Verify end-to-end OMR pipeline correctly grades the 5 realistic test scenarios."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(base_dir, "data", "answer_key.csv")
        samples_dir = os.path.join(base_dir, "data", "samples")
        pipeline = OMRPipeline(key_path)

        expected_scores = {
            "01_B21DCCN001_Chuan.png": 8.5,
            "02_B21DCCN002_Nghieng_BanToi.png": 9.0,
            "03_B21DCCN003_ThieuSang_NhieuHat.png": 8.0,
            "04_B21DCCN004_ToSaiQuyChe_TayXoa.png": 7.5,
            "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png": 9.5
        }

        for filename, expected_score in expected_scores.items():
            fpath = os.path.join(samples_dir, filename)
            if not os.path.exists(fpath):
                continue
            img = cv2.imread(fpath)
            res = pipeline.process_sheet(img, filename[:10], use_scratch_mode=True, apply_enhancement=True)
            gr = res["grading_result"]
            self.assertEqual(
                gr["score_10"], expected_score,
                f"Mismatch for {filename}: got {gr['score_10']}, expected {expected_score}"
            )
            self.assertIn("annotated_image", res)
            self.assertEqual(res["annotated_image"].shape[:2], (1000, 800))
            self.assertIn("total_pipeline_ms", res["latencies"])

    def test_adaptive_bubble_grid_resolver_and_edge_cases(self):
        """Verify _resolve_bubble_grid correctly parses standard grid and handles total_questions 1, 2, 5, 20."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        s1_path = os.path.join(base_dir, "data", "samples", "01_B21DCCN001_Chuan.png")
        if os.path.exists(s1_path):
            img = cv2.imread(s1_path, cv2.IMREAD_GRAYSCALE)
            grid, r = _resolve_bubble_grid(img, total_questions=20)
            self.assertEqual(len(grid), 20)
            self.assertIn("A", grid[1])
            self.assertIn("D", grid[20])

        # Boundary total_questions (1, 2, 5, 40)
        blank = np.zeros((1000, 800), dtype=np.uint8)
        for num_q in [1, 2, 5, 40]:
            grid, r = _resolve_bubble_grid(blank, total_questions=num_q)
            self.assertEqual(len(grid), num_q)

    def test_pipeline_edge_case_inputs(self):
        """Stress test pipeline on single-channel 2D, 4-channel BGRA, all black, all white, and random noise."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(base_dir, "data", "answer_key.csv")
        s1_path = os.path.join(base_dir, "data", "samples", "01_B21DCCN001_Chuan.png")
        pipe = OMRPipeline(key_path)

        if os.path.exists(s1_path):
            # 2D grayscale input
            gray = cv2.imread(s1_path, cv2.IMREAD_GRAYSCALE)
            res_gray = pipe.process_sheet(gray, "GRAY_INPUT")
            self.assertEqual(res_gray["grading_result"]["score_10"], 8.5)

            # 4-channel BGRA input
            bgr = cv2.imread(s1_path)
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            res_bgra = pipe.process_sheet(bgra, "BGRA_INPUT")
            self.assertEqual(res_bgra["grading_result"]["score_10"], 8.5)

        # All black sheet -> all questions blank
        black = np.zeros((1000, 800, 3), dtype=np.uint8)
        res_black = pipe.process_sheet(black, "BLACK_SHEET")
        self.assertEqual(res_black["grading_result"]["score_10"], 0.0)
        self.assertEqual(res_black["grading_result"]["blank_count"], 20)

        # All white sheet -> all questions blank
        white = np.full((1000, 800, 3), 255, dtype=np.uint8)
        res_white = pipe.process_sheet(white, "WHITE_SHEET")
        self.assertEqual(res_white["grading_result"]["score_10"], 0.0)
        self.assertEqual(res_white["grading_result"]["blank_count"], 20)

        # Pure random noise -> does not crash
        np.random.seed(99)
        noise = np.random.randint(0, 256, (1000, 800, 3), dtype=np.uint8)
        res_noise = pipe.process_sheet(noise, "NOISE_SHEET")
        self.assertIn("score_10", res_noise["grading_result"])

    def test_pipeline_opencv_mode_equivalence(self):
        """Verify pipeline OpenCV mode (use_scratch_mode=False) runs cleanly and matches scores."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(base_dir, "data", "answer_key.csv")
        s1_path = os.path.join(base_dir, "data", "samples", "01_B21DCCN001_Chuan.png")
        pipe = OMRPipeline(key_path)

        if os.path.exists(s1_path):
            img = cv2.imread(s1_path)
            res_scratch = pipe.process_sheet(img, "COMPARE", use_scratch_mode=True)
            res_cv = pipe.process_sheet(img, "COMPARE", use_scratch_mode=False)
            self.assertEqual(res_scratch["grading_result"]["score_10"], res_cv["grading_result"]["score_10"])

    def test_excel_export_empty_and_large_batch(self):
        """Verify Excel export handles empty batch and large batch without crash."""
        excel_path = os.path.join(self.tmp_dir, "stress_batch.xlsx")
        # Empty batch
        OMRGrader.generate_excel_report([], excel_path)
        self.assertTrue(os.path.exists(excel_path))

        # Large batch of 30 students
        batch = []
        for i in range(30):
            batch.append({
                "student_id": f"SV{i:03d}",
                "score_10": round(float(np.random.uniform(5.0, 10.0)), 1),
                "correct_count": int(np.random.randint(10, 21)),
                "incorrect_count": int(np.random.randint(0, 5)),
                "blank_count": int(np.random.randint(0, 3)),
                "multiple_count": int(np.random.randint(0, 2)),
                "total_questions": 20,
                "accuracy_pct": 85.0,
                "details": [{"question": q, "student_ans": "A", "correct_ans": "A", "status": "CORRECT", "is_correct": True} for q in range(1, 21)]
            })
        OMRGrader.generate_excel_report(batch, excel_path)
        self.assertTrue(os.path.exists(excel_path))

    def test_sheet_structure_mismatch_warning(self):
        """Verify detect_sheet_questions accurately counts sheet rows and warns when key length mismatches."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        s1_path = os.path.join(base_dir, "data", "samples", "01_B21DCCN001_Chuan.png")
        if os.path.exists(s1_path):
            img = cv2.imread(s1_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_q = detect_sheet_questions(gray)
            self.assertEqual(detected_q, 20)

            # Test mismatch warning when key only has 10 questions
            pipe = OMRPipeline()
            pipe.grader.answer_key = {i: 'A' for i in range(1, 11)}
            pipe.grader.total_questions = 10
            res = pipe.process_sheet(img, "TEST_MISMATCH")
            self.assertIsNotNone(res.get("warning"))
            self.assertIn("Cảnh báo cấu trúc", res["warning"])
            self.assertEqual(res.get("detected_questions"), 20)

            # Test NO warning when key matches 20 questions
            pipe.grader.answer_key = {i: 'A' for i in range(1, 21)}
            pipe.grader.total_questions = 20
            res_ok = pipe.process_sheet(img, "TEST_MATCH")
            self.assertIsNone(res_ok.get("warning"))

    def test_ncc_orientation_correction_0_90_180_270(self):
        """Verify NCC Anchor Marker orientation detection and grading for 0, 90, 180, 270 deg rotations."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(base_dir, "data", "answer_key.csv")
        samples_dir = os.path.join(base_dir, "data", "samples")
        pipe = OMRPipeline(key_path)

        cases = [
            ("01_B21DCCN001_Chuan.png", 8.5),
            ("02_B21DCCN002_Nghieng_BanToi.png", 9.0),
        ]
        rotations = [(0, 0), (90, 3), (180, 2), (270, 1)]
        for fname, exp_score in cases:
            img = cv2.imread(os.path.join(samples_dir, fname))
            self.assertIsNotNone(img)
            for angle, k in rotations:
                rot = np.rot90(img, k).copy()
                for use_sc in (True, False):
                    res = pipe.process_sheet(rot, f"{fname}_{angle}", use_scratch_mode=use_sc)
                    self.assertEqual(
                        res["detected_angle"], angle,
                        f"Wrong detected_angle for {fname} rot={angle} use_scratch={use_sc}"
                    )
                    self.assertEqual(
                        res["grading_result"]["score_10"], exp_score,
                        f"Score mismatch for {fname} rot={angle} use_scratch={use_sc}"
                    )

    def test_legacy_sheet_without_anchor_keeps_zero_angle(self):
        """Verify legacy sheets with 4 identical solid black corner squares do not falsely rotate."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(base_dir, "data", "answer_key.csv")
        s1_path = os.path.join(base_dir, "data", "samples", "01_B21DCCN001_Chuan.png")
        img = cv2.imread(s1_path).copy()
        # Overwrite TR concentric anchor with a solid black square like TL/BR/BL
        img[70:100, 726:756] = 0

        pipe = OMRPipeline(key_path)
        res = pipe.process_sheet(img, "LEGACY_4_SOLID", use_scratch_mode=True)
        self.assertEqual(res["detected_angle"], 0)
        self.assertEqual(res["grading_result"]["score_10"], 8.5)

    def test_pure_scratch_tv4_filters_no_cv2_import(self):
        """Verify pure Scratch modules tv4_filters and tv4_metrics do not import cv2 on import."""
        code = (
            "import sys; "
            f"sys.path.insert(0, {SRC_DIR!r}); "
            "import tv4_filters, tv4_metrics; "
            "print('cv2' in sys.modules)"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(proc.stdout.strip(), "False")

    def test_ablation_study_parity_on_erasure_and_shadow(self):
        """Verify run_ablation_study achieves 100% on DS-05 erasure (delta_tau) and DS-03 shadow."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(base_dir, "data", "answer_key.csv")
        samples_dir = os.path.join(base_dir, "data", "samples")
        pipe = OMRPipeline(key_path)

        # Erasure sheet (04): delta_tau rule gives 100% Adaptive accuracy
        p4 = os.path.join(samples_dir, "04_B21DCCN004_ToSaiQuyChe_TayXoa.png")
        df4 = pd.read_csv(p4.replace(".png", "_gt.csv"))
        gt4 = dict(zip(df4.iloc[:, 0].astype(int), df4.iloc[:, 1].astype(str).str.strip()))
        out4 = pipe.process_sheet(cv2.imread(p4), "ABL_04", use_scratch_mode=True)
        ab4 = run_ablation_study(
            out4["warped_gray"], gt4,
            question_grid=out4["question_grid"], bubble_radius=out4["bubble_radius"]
        )
        self.assertEqual(ab4["adaptive_accuracy_pct"], 100.0)

        # Shadow sheet (05): Adaptive (100%) outperforms Global Otsu
        p5 = os.path.join(samples_dir, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png")
        df5 = pd.read_csv(p5.replace(".png", "_gt.csv"))
        gt5 = dict(zip(df5.iloc[:, 0].astype(int), df5.iloc[:, 1].astype(str).str.strip()))
        out5 = pipe.process_sheet(cv2.imread(p5), "ABL_05", use_scratch_mode=True)
        ab5 = run_ablation_study(
            out5["warped_gray"], gt5,
            question_grid=out5["question_grid"], bubble_radius=out5["bubble_radius"]
        )
        self.assertEqual(ab5["adaptive_accuracy_pct"], 100.0)
        self.assertLess(ab5["otsu_accuracy_pct"], ab5["adaptive_accuracy_pct"])

    def test_anchor_marker_single_source_of_truth(self):
        """Verify create_anchor_marker_template is shared identically without duplication."""
        self.assertIs(create_anchor_marker_template, coord_anchor_template)
        tmpl = create_anchor_marker_template(30)
        self.assertEqual(tmpl.shape, (30, 30))
        self.assertEqual(int(tmpl[0, 0]), 0)
        self.assertEqual(int(tmpl[6, 6]), 245)
        self.assertEqual(int(tmpl[15, 15]), 0)

    def test_benchmark_all_modules_includes_ccl_and_ncc(self):
        """Verify benchmark_all_modules benchmarks both Two-Pass CCL and NCC alongside TV1-TV4."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sheet = cv2.imread(os.path.join(base_dir, "data", "sample_sheet.png"))
        rows = benchmark_all_modules(sheet)
        algos = {r["algorithm"]: r for r in rows}

        self.assertIn("Two-Pass CCL (8-Connectivity)", algos)
        ccl_row = algos["Two-Pass CCL (8-Connectivity)"]
        self.assertGreaterEqual(ccl_row["iou"], 0.985)
        self.assertEqual(ccl_row["delta_t"], 0)

        self.assertIn("Template Matching NCC (30x30)", algos)
        self.assertLess(algos["Template Matching NCC (30x30)"]["mse"], 1.0)

        # Standalone component IoU check
        m1 = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 2]], dtype=np.int32)
        m2 = np.array([[5, 5, 0], [5, 0, 0], [0, 0, 3]], dtype=np.int32)
        self.assertEqual(calculate_ccl_component_iou(m1, m2), (1.0, 1.0))

    def test_orientation_warning_on_uncertain_ncc_and_landscape(self):
        """Verify orientation_warning is None with anchor, warns on missing anchor, and flags landscape mismatch."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(base_dir, "data", "answer_key.csv")
        s1_path = os.path.join(base_dir, "data", "samples", "01_B21DCCN001_Chuan.png")
        img = cv2.imread(s1_path)
        pipe = OMRPipeline(key_path)

        # Standard anchored sheet -> no orientation warning
        res_ok = pipe.process_sheet(img, "ANCHOR_OK", use_scratch_mode=True)
        self.assertIsNone(res_ok.get("orientation_warning"))

        # Portrait un-anchored legacy sheet -> soft orientation_warning present, warning still None
        legacy = img.copy()
        legacy[70:100, 726:756] = 0
        res_leg = pipe.process_sheet(legacy, "LEGACY_PORTRAIT", use_scratch_mode=True)
        self.assertIsNotNone(res_leg.get("orientation_warning"))
        self.assertIsNone(res_leg.get("warning"))

        # Landscape un-anchored sheet -> explicit orientation_warning AND warning populated
        landscape_legacy = np.rot90(legacy, 3).copy()
        res_land = pipe.process_sheet(landscape_legacy, "LEGACY_LANDSCAPE", use_scratch_mode=True)
        self.assertIsNotNone(res_land.get("orientation_warning"))
        self.assertIn("Cảnh báo định hướng NCC", res_land["orientation_warning"])
        self.assertIn("Cảnh báo định hướng NCC", res_land.get("warning") or "")

    def test_corner_error_vs_ground_truth_scenario_02(self):
        """Verify corner localization Euclidean error vs saved ground truth is <= 3.0 px on DS-02."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(base_dir, "data", "answer_key.csv")
        s2_path = os.path.join(base_dir, "data", "samples", "02_B21DCCN002_Nghieng_BanToi.png")
        gt_c_path = os.path.join(base_dir, "data", "samples", "02_B21DCCN002_Nghieng_BanToi_corners_gt.csv")
        df_c = pd.read_csv(gt_c_path)
        gt_corners = df_c[["x", "y"]].values.astype(np.float32)

        pipe = OMRPipeline(key_path)
        img2 = cv2.imread(s2_path)
        for use_sc in (True, False):
            res = pipe.process_sheet(img2, "DS02_CORNER", use_scratch_mode=use_sc)
            err = calculate_corner_error(res["corners"], gt_corners)
            self.assertLessEqual(err, 3.0, f"E_corner={err} px exceeded 3.0 px (use_scratch={use_sc})")

    def test_ablation_fp_fn_tradeoff_and_subpixel_parity(self):
        """Verify ablation study reports FP/FN tradeoff on stress sheet and matches pipeline ratios sub-pixel."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(base_dir, "data", "answer_key.csv")
        abl_img_path = os.path.join(base_dir, "data", "ablation", "ablation_faint_pencil_erasure.png")
        abl_gt_path = os.path.join(base_dir, "data", "ablation", "ablation_faint_pencil_erasure_gt.csv")

        df_gt = pd.read_csv(abl_gt_path)
        gt_abl = dict(zip(df_gt.iloc[:, 0].astype(int), df_gt.iloc[:, 1].astype(str).str.strip()))
        pipe = OMRPipeline(key_path)
        out = pipe.process_sheet(cv2.imread(abl_img_path), "ABL_FP_FN", use_scratch_mode=True)
        ab = run_ablation_study(
            out["warped_gray"], gt_abl,
            question_grid=out["question_grid"], bubble_radius=out["bubble_radius"]
        )
        survey_by_tau = {row["fill_threshold"]: row for row in ab["threshold_survey"]}
        self.assertGreater(survey_by_tau[0.25]["fp_count"], 0)
        self.assertEqual(survey_by_tau[0.38]["fp_count"], 0)
        self.assertEqual(survey_by_tau[0.38]["fn_count"], 0)
        self.assertEqual(survey_by_tau[0.38]["mark_accuracy_pct"], 100.0)
        self.assertGreater(survey_by_tau[0.50]["fn_count"], 0)

        # Sub-pixel fill ratio parity check vs DS-05 pipeline
        from tv4_metrics import _compute_bubble_ratios_from_binary
        from tv3_thresholding import adaptive_mean_threshold_scratch
        from tv3_morphology_ccl import opening_binary_scratch
        p4 = os.path.join(base_dir, "data", "samples", "04_B21DCCN004_ToSaiQuyChe_TayXoa.png")
        out4 = pipe.process_sheet(cv2.imread(p4), "SUBPIX", use_scratch_mode=True)
        bin4 = opening_binary_scratch(
            adaptive_mean_threshold_scratch(out4["warped_gray"], block_size=51, C=42.0),
            np.ones((3, 3), dtype=np.uint8),
        )
        _, ab_ratios = _compute_bubble_ratios_from_binary(
            bin4, total_questions=20, question_grid=out4["question_grid"], bubble_radius=out4["bubble_radius"]
        )
        max_diff = max(
            abs(out4["fill_ratios"][q][opt] - ab_ratios[q][opt])
            for q in range(1, 21) for opt in ("A", "B", "C", "D")
        )
        self.assertLess(max_diff, 1e-6)

    def test_non_one_based_question_ids_and_opencv_tv3_fallback_warning(self):
        """Verify non-1-based question keys grade properly and OpenCV mode surfaces TV3 exception warning."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        s1_path = os.path.join(base_dir, "data", "samples", "01_B21DCCN001_Chuan.png")
        img = cv2.imread(s1_path)

        # Map Q1..Q20 answers to IDs 101..120
        df_gt = pd.read_csv(os.path.join(base_dir, "data", "answer_key.csv"))
        shifted_key = {100 + int(row.iloc[0]): str(row.iloc[1]).strip() for _, row in df_gt.iterrows()}
        pipe = OMRPipeline()
        pipe.grader.set_answer_key_dict(shifted_key)
        res_shift = pipe.process_sheet(img, "SHIFT_101", use_scratch_mode=True)
        self.assertEqual(res_shift["grading_result"]["score_10"], 8.5)
        self.assertEqual(res_shift["grading_result"]["details"][0]["question"], 101)

        # Simulate TV3 raising in OpenCV mode -> fallback_warning must still be populated
        import pipeline_coordinator as pc
        orig_cls = pc.TV3SegmentationPipeline

        class _BrokenTV3:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Simulated TV3 OpenCV failure")

        try:
            pc.TV3SegmentationPipeline = _BrokenTV3
            res_fb = pipe.process_sheet(img, "OPENCV_FB", use_scratch_mode=False)
            self.assertIsNotNone(res_fb.get("fallback_warning"))
            self.assertIn("Simulated TV3 OpenCV failure", res_fb["fallback_warning"])
            self.assertEqual(res_fb["grading_result"]["score_10"], 8.5)
        finally:
            pc.TV3SegmentationPipeline = orig_cls


if __name__ == "__main__":
    unittest.main()
