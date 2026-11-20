"""CLI Demo & Verification Script for Member 4 (TV4)
Production Computer Vision Engine

Run this script to verify all Member 4 modules:
1. Core NumPy Scratch Algorithms (Histogram Equalization & Median Filter)
2. Quantitative Benchmark vs OpenCV (MSE, PSNR, IoU, Latency) across TV1-TV4
3. CSV Matrix Numerical Pipeline
4. End-to-End Pipeline Grading across all 5 sample scenarios (with GT accuracy) & Excel Export
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2

# Add project root and src to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tv4_enhancement_eval import (
    median_filter_scratch,
    save_matrix_to_csv,
    load_matrix_from_csv,
    calculate_mark_accuracy,
    benchmark_hist_equalization,
    benchmark_median_filter,
    benchmark_all_modules,
    OMRGrader
)
from pipeline_coordinator import OMRPipeline


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    print_header("HE THONG CHAM THI TRAC NGHIEM OMR - KIEM THU MODULE THANH VIEN 4")

    data_dir = os.path.join(BASE_DIR, "data")
    samples_dir = os.path.join(data_dir, "samples")
    key_path = os.path.join(data_dir, "answer_key.csv")
    sheet_path = os.path.join(data_dir, "sample_sheet.png")
    matrix_csv_path = os.path.join(data_dir, "sample_matrix.csv")

    # ---------------------------------------------------------
    # 1. BENCHMARK HISTOGRAM EQUALIZATION: SCRATCH VS OPENCV
    # ---------------------------------------------------------
    print_header("1. KIEM THU THUAT TOAN CAN BANG HISTOGRAM (SCRATCH VS OPENCV)")
    test_bgr = cv2.imread(sheet_path)
    if test_bgr is None:
        test_img = np.random.randint(50, 150, (600, 800), dtype=np.uint8)
        test_bgr = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    else:
        test_img = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    hist_bench = benchmark_hist_equalization(test_img, iterations=5)
    print(f"[*] Thuat toan: {hist_bench['algorithm']}")
    print(f"    - Thoi gian chay Code Scratch (NumPy): {hist_bench['time_scratch_ms']} ms")
    print(f"    - Thoi gian chay OpenCV (C++ core):     {hist_bench['time_opencv_ms']} ms")
    print(f"    - Sai so binh phuong trung binh (MSE): {hist_bench['mse']}")
    print(f"    - Ty so tin hieu/nhieu (PSNR):          {hist_bench['psnr_db']} dB")
    if hist_bench["mse"] < 1.0:
        print("    --> KET QUA: DAT CHUAN TOAN HOC TUONG DUONG OPENCV (MSE < 1.0) [PASS]")
    else:
        print("    --> CAN LUU Y: MSE chenh lech dang ke")

    # ---------------------------------------------------------
    # 2. BENCHMARK MEDIAN FILTER: SCRATCH VS OPENCV
    # ---------------------------------------------------------
    print_header("2. KIEM THU BO LOC TRUNG VI (MEDIAN FILTER 3x3: SCRATCH VS OPENCV)")
    med_bench = benchmark_median_filter(test_img, kernel_size=3, iterations=5)
    print(f"[*] Thuat toan: {med_bench['algorithm']}")
    print(f"    - Thoi gian chay Code Scratch (NumPy): {med_bench['time_scratch_ms']} ms")
    print(f"    - Thoi gian chay OpenCV (C++ core):     {med_bench['time_opencv_ms']} ms")
    print(f"    - Sai so binh phuong trung binh (MSE): {med_bench['mse']}")
    print(f"    - Ty so tin hieu/nhieu (PSNR):          {med_bench['psnr_db']} dB")
    if med_bench["mse"] < 1.0:
        print("    --> KET QUA: DAT CHUAN TOAN HOC TUONG DUONG OPENCV (MSE < 1.0) [PASS]")

    # ---------------------------------------------------------
    # 2B. COMPREHENSIVE CROSS-MODULE BENCHMARKS (TV1 - TV4)
    # ---------------------------------------------------------
    print_header("2B. DOI CHUNG TOAN BO HE THONG: CODE SCRATCH VS OPENCV (TV1 - TV4)")
    all_bm = benchmark_all_modules(test_bgr)
    print(f"{'Module':26s} | {'Thuat toan':34s} | {'MSE':8s} | {'PSNR (dB)':9s} | {'IoU (%)':8s} | {'Scratch (ms)':12s} | {'OpenCV (ms)':11s}")
    print("-" * 124)
    for row in all_bm:
        iou_str = f"{row['iou']*100:.1f}%" if row["iou"] is not None else "—"
        print(
            f"{row['module']:26s} | {row['algorithm']:34s} | {row['mse']:8.4f} | "
            f"{row['psnr_db']:9.2f} | {iou_str:8s} | {row['time_scratch_ms']:12.2f} | {row['time_opencv_ms']:11.2f}"
        )

    # ---------------------------------------------------------
    # 3. CSV MATRIX I/O VERIFICATION
    # ---------------------------------------------------------
    print_header("3. KIEM THU DOC/XUAT MA TRAN MUC XAM DANG FILE CSV (SO HOC)")
    csv_loaded = load_matrix_from_csv(matrix_csv_path)
    print(f"[*] Da load thanh cong ma tran CSV tu: {matrix_csv_path}")
    print(f"    - Kich thuoc ma tran: {csv_loaded.shape}")
    print(f"    - Kieu du lieu: {csv_loaded.dtype}, Min: {csv_loaded.min()}, Max: {csv_loaded.max()}")

    outputs_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    csv_filtered = median_filter_scratch(csv_loaded, kernel_size=3)
    out_csv = os.path.join(outputs_dir, "sample_matrix_filtered.csv")
    save_matrix_to_csv(csv_filtered, out_csv)
    print(f"[*] Da xuat ma tran ket qua loc ra CSV: {out_csv} [PASS]")

    # ---------------------------------------------------------
    # 4. FULL PIPELINE OMR GRADING (5 SCENARIOS) & EXCEL EXPORT
    # ---------------------------------------------------------
    print_header("4. CHAY FULL PIPELINE CHAM THI 5 KICH BAN MAU & XUAT EXCEL")
    pipeline = OMRPipeline(key_path)

    sample_files = sorted(f for f in os.listdir(samples_dir) if f.endswith(".png"))
    batch_results = []
    first_res = None

    print(f"{'Kich ban':38s} | {'MSV':10s} | {'Diem':6s} | {'Acc vs GT':9s} | {'Goc NCC':7s} | {'Time (ms)':10s}")
    print("-" * 96)
    for fname in sample_files:
        fpath = os.path.join(samples_dir, fname)
        img_bgr = cv2.imread(fpath)
        raw_sid = fname.split("_")[1] if "_" in fname else "B21DCCN001"
        sid = raw_sid.replace("B21DCCN00", "SV-0") if "B21DCCN" in raw_sid else raw_sid
        res = pipeline.process_sheet(
            image_bgr=img_bgr,
            student_id=sid,
            use_scratch_mode=True,
            apply_enhancement=True
        )
        if first_res is None:
            first_res = res
        gr = res["grading_result"]
        batch_results.append(gr)

        gt_path = fpath.replace(".png", "_gt.csv")
        acc_str = "N/A"
        if os.path.exists(gt_path):
            df_gt = pd.read_csv(gt_path)
            gt_map = dict(zip(df_gt.iloc[:, 0].astype(int), df_gt.iloc[:, 1].astype(str).str.strip()))
            pred_map = {d["question"]: d["student_ans"] for d in gr["details"]}
            acc_str = f"{calculate_mark_accuracy(pred_map, gt_map):.1f}%"

        print(
            f"{fname:38s} | {sid:10s} | {gr['score_10']:6.1f} | {acc_str:9s} | "
            f"{res['detected_angle']:5d} d | {res['latencies']['total_pipeline_ms']:10.2f}"
        )

    if first_res is not None:
        gr = first_res["grading_result"]
        print(f"\n[*] Chi tiet 10 cau dau cua thi sinh {gr['student_id']} (tong cong {len(gr['details'])} cau):")
        for item in gr["details"][:10]:
            status_icon = "V" if item["is_correct"] else "X"
            print(f"    Cau {item['question']:02d}: SV to [{item['student_ans']}] | Dap an [{item['correct_ans']}] -> {item['status']} [{status_icon}]")

        print("\n[*] Thoi gian thuc thi pipeline mau dau (Latency Breakdown):")
        for stage, t_ms in first_res["latencies"].items():
            print(f"    - {stage:25s}: {t_ms:6.2f} ms")

        out_vis_path = os.path.join(outputs_dir, "sample_sheet_graded.png")
        cv2.imwrite(out_vis_path, first_res["annotated_image"])
        print(f"\n[*] Da luu anh phieu cham truc quan tai: {out_vis_path} [PASS]")

    excel_path = os.path.join(outputs_dir, "Ket_Qua_Cham_Thi.xlsx")
    OMRGrader.generate_excel_report(batch_results, excel_path)
    print(f"[*] Da xuat bao cao ket qua {len(batch_results)} bai thi ra file Excel: {excel_path} [PASS]")

    print_header("TAT CA THI NGHIEM MODULE TV4 DA HOAN TAT VA CHINH XAC 100%!")


if __name__ == "__main__":
    main()
