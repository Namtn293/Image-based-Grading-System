"""Inverse Warping kết hợp Nội suy song tuyến tính (Bilinear Interpolation)."""

from typing import Tuple

import numpy as np


def warp_perspective_bilinear(
    img: np.ndarray,
    H: np.ndarray,
    out_size: Tuple[int, int]
) -> np.ndarray:
    """
    Nắn thẳng ảnh bằng phương pháp Ánh xạ ngược (Inverse Warping)
    kết hợp Nội suy song tuyến tính (Bilinear Interpolation). Vector hóa 100% bằng NumPy.

    Bản chất toán học:
    1. Quét từng điểm (x', y') trên ảnh phẳng đích.
    2. Chiếu ngược về ảnh nguồn: [x, y, 1]^T ~ H^(-1) * [x', y', 1]^T.
    3. Nội suy mức xám từ 4 điểm nguyên lân cận:
       I(x, y) = (1 - a)(1 - b)*Ia + a*(1 - b)*Ib + (1 - a)*b*Ic + a*b*Id
       với a = x - floor(x), b = y - floor(y).
    Điểm lân cận nằm ngoài ảnh nguồn mang giá trị 0 (BORDER_CONSTANT của OpenCV), nên dải
    1 pixel sát mép ảnh nguồn được nội suy mờ dần thay vì bị cắt cụt.

    Args:
        img: Ảnh đầu vào (H_src, W_src) hoặc (H_src, W_src, C).
        H: Ma trận Homography (3, 3) ánh xạ nguồn -> đích.
        out_size: Kích thước ảnh đích (W_dst, H_dst).

    Returns:
        Ảnh đã nắn thẳng, shape (H_dst, W_dst[, C]), dtype uint8.
    """
    w_dst, h_dst = int(out_size[0]), int(out_size[1])
    if w_dst <= 0 or h_dst <= 0:
        raise ValueError(f"Kích thước đích không hợp lệ: {out_size}")
    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ValueError(f"Ảnh đầu vào phải 2D hoặc 3D, nhận được shape {img.shape}")
    h_src, w_src = img.shape[:2]

    try:
        H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Ma trận Homography suy biến, không thể nghịch đảo.") from exc

    # Lưới tọa độ đích ở dạng thuần nhất (3, N)
    y_dst, x_dst = np.meshgrid(
        np.arange(h_dst, dtype=np.float64),
        np.arange(w_dst, dtype=np.float64),
        indexing="ij"
    )
    coords_dst = np.stack([x_dst.ravel(), y_dst.ravel(), np.ones(h_dst * w_dst)], axis=0)

    coords_src = H_inv @ coords_dst
    w_scale = coords_src[2, :]
    w_scale = np.where(np.abs(w_scale) < 1e-10, 1e-10, w_scale)
    x_src = (coords_src[0, :] / w_scale).reshape((h_dst, w_dst))
    y_src = (coords_src[1, :] / w_scale).reshape((h_dst, w_dst))

    x0 = np.floor(x_src).astype(np.int64)
    y0 = np.floor(y_src).astype(np.int64)
    alpha = x_src - x0
    beta = y_src - y0

    x1 = x0 + 1
    y1 = y0 + 1

    if img.ndim == 3:
        alpha = alpha[..., np.newaxis]
        beta = beta[..., np.newaxis]

    def sample(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        """Lấy mức xám tại (yy, xx); điểm lân cận nằm ngoài ảnh nguồn mang giá trị 0
        (BORDER_CONSTANT) nên vùng sát mép được nội suy mờ dần như cv2.warpPerspective."""
        inside = (xx >= 0) & (xx < w_src) & (yy >= 0) & (yy < h_src)
        vals = img[np.clip(yy, 0, h_src - 1), np.clip(xx, 0, w_src - 1)].astype(np.float64)
        if img.ndim == 3:
            inside = inside[..., np.newaxis]
        return np.where(inside, vals, 0.0)

    interp = (
        (1.0 - alpha) * (1.0 - beta) * sample(y0, x0)
        + alpha * (1.0 - beta) * sample(y0, x1)
        + (1.0 - alpha) * beta * sample(y1, x0)
        + alpha * beta * sample(y1, x1)
    )
    return np.clip(np.round(interp), 0, 255).astype(np.uint8)
