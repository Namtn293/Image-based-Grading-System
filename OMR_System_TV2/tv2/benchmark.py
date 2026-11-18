"""Benchmark định lượng TV2: Scratch (NumPy) vs. OpenCV — Bảng 6.1 của đề cương.

Chạy từ thư mục OMR_System_TV2:
    python -m tv2.benchmark

Kết quả:
    outputs/tv2/benchmark_metrics.csv, outputs/tv2/benchmark_metrics.json
    outputs/tv2/triplet_warp.png      (Scratch | OpenCV | |Scratch - OpenCV| x 10)
    data/output/warped_sheet_scratch.png, data/output/warped_sheet_opencv.png
"""

import csv
import json
import os
import time
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from . import baseline_opencv as cvb
from .geometry import destination_corners, order_corners
from .homography import solve_homography_svd
from .template_matching import match_template_ncc
from .warping import warp_perspective_bilinear

TV2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(TV2_DIR, "data", "input")
OUTPUT_DIR = os.path.join(TV2_DIR, "data", "output")
REPORT_DIR = os.path.join(TV2_DIR, "outputs", "tv2")

TARGET_SIZE = (800, 1000)   # Bảng 6.1: kích thước đích 800 x 1000
TEMPLATE_SIZE = 32          # Bảng 6.1: Template Marker 32 x 32
PSNR_TARGET_DB = 48.0       # Mục 7.1: PSNR > 48 dB
HOMOGRAPHY_TOL = 1e-4
NCC_TOL = 1e-4


def _timed(fn: Callable, repeats: int = 5) -> Tuple[object, float]:
    """Chạy fn `repeats` lần, trả (kết quả, thời gian trung vị ms)."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return result, float(np.median(times))


def _mse_psnr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    psnr = 999.0 if mse < 1e-12 else float(10.0 * np.log10(255.0 ** 2 / mse))
    return mse, psnr


def _row(algorithm: str, mse: float, psnr: float, error_metric: str,
         t_sc: float, t_cv: float, passed: bool) -> Dict[str, object]:
    return {
        "module": "TV2",
        "algorithm": algorithm,
        "mse": round(mse, 6),
        "psnr_db": round(psnr, 2),
        "iou": None,
        "dice": None,
        "error_metric": error_metric,
        "time_scratch_ms": round(t_sc, 3),
        "time_opencv_ms": round(t_cv, 3),
        "status": "PASS" if passed else "FAIL",
    }


def load_sample() -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(os.path.join(INPUT_DIR, "sample_tilted_sheet.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh mẫu trong {INPUT_DIR}")
    corners = np.loadtxt(os.path.join(INPUT_DIR, "input_corners.txt"))
    return img, corners


def run_benchmark(save_outputs: bool = True) -> List[Dict[str, object]]:
    img, raw_corners = load_sample()
    ordered, _ = order_corners(raw_corners)
    dst = destination_corners(TARGET_SIZE)
    rows: List[Dict[str, object]] = []

    # 1. Homography: solve_homography_svd vs cv2.getPerspectiveTransform
    H_sc, t_sc = _timed(lambda: solve_homography_svd(ordered, dst))
    H_cv, t_cv = _timed(lambda: cvb.get_perspective_transform_cv(ordered, dst))
    h_err = float(np.max(np.abs(H_sc - H_cv)))
    rows.append(_row("Homography 3x3 (Normalized DLT + SVD)", 0.0, 999.0,
                     f"Max |H_sc - H_cv|: {h_err:.2e}", t_sc, t_cv, h_err < HOMOGRAPHY_TOL))

    # 2. Warp: warp_perspective_bilinear vs cv2.warpPerspective (cùng ma trận H để tách riêng sai số nội suy)
    warped_sc, t_sc = _timed(lambda: warp_perspective_bilinear(img, H_cv, TARGET_SIZE))
    warped_cv, t_cv = _timed(lambda: cvb.warp_perspective_cv(img, H_cv, TARGET_SIZE))
    mse, psnr = _mse_psnr(warped_sc, warped_cv)
    max_diff = int(np.max(np.abs(warped_sc.astype(int) - warped_cv.astype(int))))
    rows.append(_row("Inverse Warping + Bilinear (800x1000)", mse, psnr,
                     f"Max Diff: {max_diff}", t_sc, t_cv, psnr > PSNR_TARGET_DB))

    # 3. NCC: match_template_ncc vs cv2.matchTemplate(TM_CCOEFF_NORMED), template 32x32 ở góc trên-trái phiếu
    h, w = warped_cv.shape
    roi = warped_cv[: h // 4, : w // 4]
    ty, tx = int(np.argmin(roi.mean(axis=1))), int(np.argmin(roi.mean(axis=0)))
    ty = int(np.clip(ty - TEMPLATE_SIZE // 2, 0, roi.shape[0] - TEMPLATE_SIZE))
    tx = int(np.clip(tx - TEMPLATE_SIZE // 2, 0, roi.shape[1] - TEMPLATE_SIZE))
    template = roi[ty:ty + TEMPLATE_SIZE, tx:tx + TEMPLATE_SIZE].copy()
    ncc_sc, t_sc = _timed(lambda: match_template_ncc(roi, template))
    ncc_cv, t_cv = _timed(lambda: cvb.match_template_cv(roi, template))
    # Ở cửa sổ gần như phẳng (std < 2 mức xám, vùng giấy trắng) NCC ~ 0/0: OpenCV tính bằng float32
    # nên dao động ~1e-3, còn bản Scratch tính chính xác bằng int64 -> chỉ so trên cửa sổ có nội dung.
    win_std = np.lib.stride_tricks.sliding_window_view(roi.astype(np.float64), template.shape).std(axis=(-2, -1))
    informative = win_std >= 2.0
    diff = np.abs(ncc_sc - ncc_cv)
    ncc_err = float(np.max(diff[informative])) if np.any(informative) else 0.0
    flat_err = float(np.max(diff[~informative])) if np.any(~informative) else 0.0
    peak_sc = np.unravel_index(np.argmax(ncc_sc), ncc_sc.shape)
    peak_cv = np.unravel_index(np.argmax(ncc_cv), ncc_cv.shape)
    rows.append(_row("NCC Template Matching (32x32)", 0.0, 999.0,
                     f"Max |NCC diff|: {ncc_err:.2e} (vùng phẳng: {flat_err:.1e}); "
                     f"peak {tuple(map(int, peak_sc))} vs {tuple(map(int, peak_cv))}",
                     t_sc, t_cv, ncc_err < NCC_TOL and peak_sc == peak_cv))

    # 4. End-to-end Giai đoạn 2: sắp xếp góc + Homography + Warp
    from .pipeline import warp_perspective_sheet
    sheet_sc, t_sc = _timed(lambda: warp_perspective_sheet(img, raw_corners, *TARGET_SIZE, use_scratch=True))
    sheet_cv, t_cv = _timed(lambda: cvb.warp_perspective_sheet_cv(img, raw_corners, *TARGET_SIZE))
    mse, psnr = _mse_psnr(sheet_sc, sheet_cv)
    rows.append(_row("Pipeline TV2 end-to-end (warp_perspective_sheet)", mse, psnr,
                     f"Max Diff: {int(np.max(np.abs(sheet_sc.astype(int) - sheet_cv.astype(int))))}",
                     t_sc, t_cv, psnr > PSNR_TARGET_DB))

    if save_outputs:
        os.makedirs(REPORT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(REPORT_DIR, "benchmark_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(REPORT_DIR, "benchmark_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        cv2.imwrite(os.path.join(OUTPUT_DIR, "warped_sheet_scratch.png"), sheet_sc)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "warped_sheet_opencv.png"), sheet_cv)

        diff = np.clip(np.abs(sheet_sc.astype(np.int16) - sheet_cv.astype(np.int16)) * 10, 0, 255).astype(np.uint8)
        triplet = np.hstack([
            cv2.cvtColor(sheet_sc, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sheet_cv, cv2.COLOR_GRAY2BGR),
            cv2.applyColorMap(diff, cv2.COLORMAP_JET),
        ])
        for i, label in enumerate(["1. Scratch (NumPy)", "2. OpenCV", "3. |Scratch - OpenCV| x 10"]):
            cv2.putText(triplet, label, (i * TARGET_SIZE[0] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(REPORT_DIR, "triplet_warp.png"), triplet)

    return rows


def print_table(rows: List[Dict[str, object]]) -> None:
    header = f"{'Thuật toán (TV2)':<50} | {'MSE':>9} | {'PSNR':>7} | {'Sai số':<52} | {'Scratch':>9} | {'OpenCV':>8} | KQ"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['algorithm']:<50} | {r['mse']:>9.4f} | {r['psnr_db']:>7.2f} | {str(r['error_metric']):<52} | "
              f"{r['time_scratch_ms']:>7.2f}ms | {r['time_opencv_ms']:>6.2f}ms | {r['status']}")


if __name__ == "__main__":
    results = run_benchmark()
    print_table(results)
    print(f"\nĐã lưu kết quả vào {REPORT_DIR} và {OUTPUT_DIR}")
