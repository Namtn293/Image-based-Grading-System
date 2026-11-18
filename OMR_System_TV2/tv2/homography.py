"""Ước lượng ma trận Homography 3x3 bằng Normalized DLT + SVD."""

import numpy as np

from .config import DEGENERATE_TOL


def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    """Ma trận chuẩn hóa Hartley: dời trọng tâm về gốc, co giãn để khoảng cách TB = sqrt(2)."""
    center = pts.mean(axis=0)
    mean_dist = np.mean(np.sqrt(np.sum((pts - center) ** 2, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    return np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Áp dụng phép biến đổi thuần nhất 3x3 lên tập điểm (N, 2)."""
    homog = np.column_stack([pts, np.ones(len(pts))]) @ T.T
    return homog[:, :2] / homog[:, 2:3]


def build_dlt_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Lập ma trận A (2N x 9) của hệ thuần nhất A h = 0.

    Mỗi cặp điểm (x_i, y_i) <-> (x'_i, y'_i) sinh ra 2 phương trình:
        [-x_i, -y_i, -1,    0,    0,  0,  x'_i x_i, x'_i y_i, x'_i] h = 0
        [   0,    0,  0, -x_i, -y_i, -1,  y'_i x_i, y'_i y_i, y'_i] h = 0
    """
    n = len(src)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = -x, -y, -1.0
    A[0::2, 6], A[0::2, 7], A[0::2, 8] = u * x, u * y, u
    A[1::2, 3], A[1::2, 4], A[1::2, 5] = -x, -y, -1.0
    A[1::2, 6], A[1::2, 7], A[1::2, 8] = v * x, v * y, v
    return A


def solve_homography_svd(src: np.ndarray, dst: np.ndarray, tol: float = DEGENERATE_TOL) -> np.ndarray:
    """
    Ước lượng ma trận biến đổi phối cảnh Homography H (3x3) từ 4 cặp điểm
    tương ứng bằng thuật toán Direct Linear Transformation (DLT) và phân tích SVD.

    Giải nghiệm h của A h = 0 bằng SVD (A = U * Sigma * V^T): h là hàng cuối của V^T,
    ứng với giá trị kỳ dị nhỏ nhất.

    Ổn định số học (Normalized DLT - Hartley):
    Tọa độ pixel cỡ hàng nghìn khiến các cột của A chênh nhau tới ~10^6 lần.
    Trước khi lập A, điểm nguồn và đích được chuẩn hóa bằng T_src, T_dst;
    sau khi giải được H_n, khôi phục H = T_dst^(-1) * H_n * T_src.

    Args:
        src: 4 điểm trên ảnh nguồn, shape (4, 2).
        dst: 4 điểm trên ảnh đích, shape (4, 2).
        tol: Ngưỡng tương đối phát hiện hệ suy biến.

    Returns:
        H: Ma trận Homography shape (3, 3), chuẩn hóa H[2, 2] = 1.0.

    Raises:
        ValueError: Nếu 4 điểm suy biến (3 điểm thẳng hàng / trùng nhau) khiến H không xác định.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src và dst đều phải có shape (4, 2).")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError("src/dst chứa giá trị NaN/Inf.")

    T_src = _normalization_matrix(src)
    T_dst = _normalization_matrix(dst)
    A = build_dlt_matrix(_apply(T_src, src), _apply(T_dst, dst))

    _, sigma, vt = np.linalg.svd(A)

    # Hạng của A phải bằng 8; nếu giá trị kỳ dị thứ 8 ~ 0 thì nghiệm không duy nhất
    if sigma[-1] < tol * sigma[0]:
        raise ValueError("4 điểm tương ứng bị suy biến (thẳng hàng hoặc trùng nhau), không giải được Homography.")

    H = np.linalg.inv(T_dst) @ vt[-1, :].reshape((3, 3)) @ T_src
    if np.abs(H[2, 2]) < 1e-12:
        raise ValueError("Homography suy biến (h33 ~ 0).")
    return H / H[2, 2]
