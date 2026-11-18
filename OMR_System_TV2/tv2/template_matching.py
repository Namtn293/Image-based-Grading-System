"""So khớp mẫu Tương quan chéo chuẩn hóa (NCC) — tương đương cv2.TM_CCOEFF_NORMED."""

import numpy as np


def _box_sum(integral: np.ndarray, h: int, w: int) -> np.ndarray:
    """Tổng mọi cửa sổ h x w trong O(1) mỗi vị trí từ ảnh tích phân có viền 0."""
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def _integral(img: np.ndarray) -> np.ndarray:
    """Ảnh tích phân (H+1, W+1) với hàng/cột đầu bằng 0."""
    out = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=img.dtype)
    out[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return out


def match_template_ncc(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    So khớp mẫu bằng Tương quan chéo chuẩn hóa (Normalized Cross-Correlation - NCC).

    NCC(x, y) = Sum((I - I_mean)(T - T_mean)) / sqrt(Sum(I - I_mean)^2 * Sum(T - T_mean)^2) ∈ [-1, 1].

    Tối ưu hóa bộ nhớ & tốc độ:
    - Tử số: vì Sum(T - T_mean) = 0 nên Sum((I - I_mean)(T - T_mean)) = Sum(I * (T - T_mean)).
      Tính bằng einsum trực tiếp trên view cửa sổ trượt, KHÔNG tạo mảng 4 chiều trung gian.
    - Mẫu số: Sum(I - I_mean)^2 = Sum(I^2) - (Sum I)^2 / N, với Sum I và Sum I^2 của mọi
      cửa sổ lấy trong O(1) từ 2 ảnh tích phân. Với ảnh số nguyên, phép tính dùng int64 nên
      chính xác tuyệt đối (cửa sổ phẳng cho phương sai đúng bằng 0).

    Args:
        img: Ảnh mức xám 2D (H, W).
        template: Ảnh mẫu 2D (m, n) với m <= H, n <= W.

    Returns:
        Bản đồ tương quan float32 shape (H - m + 1, W - n + 1).
    """
    img = np.asarray(img)
    template = np.asarray(template)
    if img.ndim != 2 or template.ndim != 2:
        raise ValueError("match_template_ncc yêu cầu ảnh và template đều là mảng 2D mức xám.")

    h_img, w_img = img.shape
    h_tmpl, w_tmpl = template.shape
    if h_tmpl > h_img or w_tmpl > w_img:
        raise ValueError("Kích thước template không được lớn hơn kích thước ảnh!")

    n_px = h_tmpl * w_tmpl
    out_shape = (h_img - h_tmpl + 1, w_img - w_tmpl + 1)

    tmpl_f = template.astype(np.float64)
    t_zero = tmpl_f - tmpl_f.mean()
    t_norm = np.sqrt(np.sum(t_zero ** 2))
    if t_norm < 1e-10:
        return np.zeros(out_shape, dtype=np.float32)

    # Tử số trên view cửa sổ trượt (không sao chép)
    windows = np.lib.stride_tricks.sliding_window_view(img.astype(np.float64), (h_tmpl, w_tmpl))
    numerator = np.einsum("ijkl,kl->ij", windows, t_zero)

    # Mẫu số qua 2 ảnh tích phân (int64 chính xác cho ảnh số nguyên)
    acc = img.astype(np.int64) if np.issubdtype(img.dtype, np.integer) else img.astype(np.float64)
    win_sum = _box_sum(_integral(acc), h_tmpl, w_tmpl)
    win_sum_sq = _box_sum(_integral(acc * acc), h_tmpl, w_tmpl)

    # N * Sum(I - I_mean)^2 = N * Sum(I^2) - (Sum I)^2
    win_var_n = np.maximum((n_px * win_sum_sq - win_sum * win_sum).astype(np.float64), 0.0)
    denominator = np.sqrt(win_var_n / n_px) * t_norm

    ncc_map = np.zeros(out_shape, dtype=np.float32)
    valid = denominator > 1e-10
    ncc_map[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return np.clip(ncc_map, -1.0, 1.0)
