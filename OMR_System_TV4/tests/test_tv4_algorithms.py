"""Comprehensive Unit Tests for Module TV4:
Histogram Equalization, Median Filter 2D, Robust CSV I/O, and Grading Logic.
Course: Digital Image Processing (INT13146) - PTIT
"""

import os
import io
import unittest
import numpy as np
import cv2
import openpyxl

from src.tv4_enhancement_eval import (
    histogram_equalization_scratch,
    median_filter_scratch,
    _pad_2d_scratch,
    save_matrix_to_csv,
    load_matrix_from_csv,
    calculate_mse,
    calculate_psnr,
    calculate_difference_map,
    calculate_binary_iou_dice,
    calculate_otsu_delta_t,
    calculate_corner_error,
    calculate_mark_accuracy,
    benchmark_hist_equalization,
    benchmark_median_filter,
    benchmark_all_modules,
    run_ablation_study,
    OMRGrader
)
from src.pipeline_coordinator import OMRPipeline


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


if __name__ == "__main__":
    unittest.main()
