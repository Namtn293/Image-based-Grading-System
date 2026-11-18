"""Module Thành viên 3: Trích xuất ROI ô đáp án, tính Fill Ratio và phân loại đáp án OMR.

Cơ sở lý thuyết:
- Trích xuất vùng quan tâm (ROI) ô trắc nghiệm hình tròn bán kính R tại tâm (cx, cy):
    \\Omega_{q, c} = {(u, v) | (u - cx)^2 + (v - cy)^2 <= R^2}
- Tỷ lệ lấp đầy (Fill Ratio) trên ảnh nhị phân:
    \\tau_{q, c} = (1 / |\\Omega_{q, c}|) * \\sum_{(u, v) \\in \\Omega_{q, c}} I_{bin}(u, v)
  trong đó I_{bin}(u, v) \\in {0, 1} (1: điểm pixel được tô màu đen của bút chì - tiền cảnh).
- Phân loại trạng thái:
    - BLANK: tau_{(1)} < tau_min (mặc định 0.38)
    - VALID: tau_{(1)} >= tau_min và tau_{(2)} < tau_min
    - ERASED_VALID: tau_{(1)} >= tau_min, tau_{(2)} >= tau_min và Delta_tau = tau_{(1)} - tau_{(2)} >= 0.15
    - MULTIPLE_INVALID: tau_{(1)} >= tau_min, tau_{(2)} >= tau_min và Delta_tau < 0.15
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Các hằng số trạng thái
STATUS_VALID = "VALID"
STATUS_BLANK = "BLANK"
STATUS_ERASED_VALID = "ERASED_VALID"
STATUS_MULTIPLE_INVALID = "MULTIPLE_INVALID"

DEFAULT_TAU_MIN = 0.38
DEFAULT_DELTA_TAU_MIN = 0.15
DEFAULT_BUBBLE_RADIUS = 12.0
DEFAULT_OPTIONS = ("A", "B", "C", "D")


@dataclass
class BubbleROI:
    """Đặc tả một ô tròn trắc nghiệm."""
    question_id: int
    option: str
    center: Tuple[float, float]  # (cx, cy)
    radius: float = DEFAULT_BUBBLE_RADIUS

    @property
    def cx(self) -> float:
        return self.center[0]

    @property
    def cy(self) -> float:
        return self.center[1]


@dataclass
class QuestionResult:
    """Kết quả nhận diện cho một câu hỏi trắc nghiệm."""
    question_id: int
    selected: Optional[str]
    status: str
    fill_ratios: Dict[str, float]
    delta_tau: float
    sorted_options: List[Tuple[str, float]]
    is_valid: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành từ điển JSON/dict tiện lợi."""
        return {
            "question_id": self.question_id,
            "selected": self.selected,
            "status": self.status,
            "is_valid": self.is_valid,
            "fill_ratios": {k: round(v, 4) for k, v in self.fill_ratios.items()},
            "delta_tau": round(self.delta_tau, 4),
            "sorted_options": [(opt, round(val, 4)) for opt, val in self.sorted_options],
            "warning": self.warning,
        }


@dataclass
class OMRRecognitionResult:
    """Kết quả nhận diện toàn bộ phiếu thi OMR."""
    questions: Dict[int, QuestionResult]
    answers_vector: Dict[int, Optional[str]] = field(default_factory=dict)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answers_vector": self.answers_vector,
            "summary": self.summary,
            "questions": {qid: qres.to_dict() for qid, qres in self.questions.items()},
        }


def create_circular_mask_scratch(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Tạo mặt nạ hình tròn bán kính R tại tâm (cx, cy) hoàn toàn bằng NumPy.

    Phương trình mặt nạ:
        Omega = {(u, v) | (u - cx)^2 + (v - cy)^2 <= R^2}

    Parameters:
        shape: Tuple (H, W) kích thước mặt nạ.
        center: Tuple (cx, cy) tọa độ tâm hình tròn (x: trục ngang, y: trục dọc).
        radius: Bán kính hình tròn R.

    Returns:
        mask: Mảng boolean numpy kích thước (H, W), True bên trong hình tròn.
    """
    h, w = shape[:2]
    cx, cy = center
    # Tạo lưới tọa độ
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    return dist_sq <= (radius ** 2)


def compute_bubble_fill_ratio_scratch(
    binary_image: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    use_local_patch: bool = True,
    binary_threshold: Optional[float] = None,
) -> float:
    """Tính tỷ lệ lấp đầy (Fill Ratio) của một ô tròn trắc nghiệm.

    Công thức toán học:
        tau = (1 / |Omega|) * sum_{(u, v) in Omega} I_bin(u, v)

    Xử lý tự động chuẩn hóa:
        - Nếu ảnh nhị phân có giá trị {0, 255}, tự động ánh xạ điểm tiền cảnh (tô chì) thành 1,
          nền giấy trắng thành 0.
        - Xử lý biên an toàn (boundary clipping) nếu hình tròn nằm sát mép ảnh.
        - Tối ưu hóa: Trích xuất patch cục bộ bounding box [cy-R, cy+R, cx-R, cx+R] thay vì
          tạo mask kích thước toàn ảnh HxW, giúp tăng tốc độ xử lý gấp nhiều lần.

    Parameters:
        binary_image: Ảnh 2D nhị phân numpy array (giá trị 0 hoặc 255, hoặc 0 và 1).
        center: Tọa độ tâm (cx, cy).
        radius: Bán kính ô tròn R.
        use_local_patch: Nếu True, cắt patch cục bộ để tính cực nhanh.

    Returns:
        tau: Tỷ lệ lấp đầy trong đoạn [0.0, 1.0].
    """
    if binary_image.ndim != 2:
        raise ValueError(f"binary_image phải là ảnh 2 chiều (H, W), nhận được ndim={binary_image.ndim}")

    h, w = binary_image.shape
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)

    if r <= 0:
        return 0.0

    if not use_local_patch:
        mask = create_circular_mask_scratch((h, w), (cx, cy), r)
        total_pixels = int(np.sum(mask))
        if total_pixels == 0:
            return 0.0

        # Chuẩn hóa về [0, 1]: Điểm tô là điểm tiền cảnh (> 0 hoặc > 127)
        bubble_pixels = binary_image[mask]
        if binary_image.dtype == bool:
            filled_count = np.sum(bubble_pixels)
        else:
            if binary_threshold is not None:
                thresh = binary_threshold
            else:
                thresh = 127.0 if (bubble_pixels.size > 0 and float(np.max(bubble_pixels)) > 1.0) else 0.5
            filled_count = np.sum(bubble_pixels > thresh)

        return float(filled_count) / float(total_pixels)

    # Chế độ tối ưu: Cắt patch cục bộ quanh bounding box hình tròn
    r_int = int(math.ceil(r))
    x_min = max(0, int(math.floor(cx - r_int)))
    x_max = min(w, int(math.ceil(cx + r_int + 1)))
    y_min = max(0, int(math.floor(cy - r_int)))
    y_max = min(h, int(math.ceil(cy + r_int + 1)))

    if x_min >= x_max or y_min >= y_max:
        return 0.0

    patch = binary_image[y_min:y_max, x_min:x_max]
    patch_h, patch_w = patch.shape

    # Tâm tương đối bên trong patch cục bộ
    rel_cx = cx - x_min
    rel_cy = cy - y_min

    y_grid, x_grid = np.ogrid[:patch_h, :patch_w]
    local_mask = ((x_grid - rel_cx) ** 2 + (y_grid - rel_cy) ** 2) <= (r ** 2)

    total_pixels = int(np.sum(local_mask))
    if total_pixels == 0:
        return 0.0

    patch_bubble = patch[local_mask]
    if binary_image.dtype == bool:
        filled_count = np.sum(patch_bubble)
    else:
        # Tránh gọi np.max(binary_image) trên toàn ảnh cho mỗi ô
        if binary_threshold is not None:
            thresh = binary_threshold
        else:
            thresh = 127.0 if (patch_bubble.size > 0 and float(np.max(patch_bubble)) > 1.0) else 0.5
        filled_count = np.sum(patch_bubble > thresh)

    return float(filled_count) / float(total_pixels)


def classify_question_bubbles(
    fill_ratios: Dict[str, float],
    question_id: int = 1,
    tau_min: float = DEFAULT_TAU_MIN,
    delta_tau_min: float = DEFAULT_DELTA_TAU_MIN,
) -> QuestionResult:
    """Bộ phân loại trạng thái câu hỏi OMR dựa trên Fill Ratio và luật quyết định.

    Quy tắc phân loại:
        1. Sắp xếp các lựa chọn theo tỷ lệ lấp đầy giảm dần:
           tau_(1) >= tau_(2) >= tau_(3) >= tau_(4)
        2. Trường hợp 1 - Bỏ trống (BLANK):
           Nếu tau_(1) < tau_min: Thí sinh không tô ô nào -> selected = None, status = 'BLANK'.
        3. Trường hợp 2 - Tô chuẩn 1 đáp án (VALID):
           Nếu tau_(1) >= tau_min và tau_(2) < tau_min:
           Chỉ có duy nhất 1 ô vượt ngưỡng -> selected = opt_(1), status = 'VALID'.
        4. Trường hợp 3 - Xử lý vết tẩy xóa (ERASED_VALID):
           Nếu tau_(1) >= tau_min và tau_(2) >= tau_min nhưng Delta_tau = tau_(1) - tau_(2) >= delta_tau_min:
           Ô tô chính đậm vượt trội so với vết tẩy mờ -> selected = opt_(1), status = 'ERASED_VALID'.
        5. Trường hợp 4 - Tô nhiều đáp án / Phạm quy (MULTIPLE_INVALID):
           Nếu tau_(1) >= tau_min và tau_(2) >= tau_min và Delta_tau < delta_tau_min:
           Tô đúp hoặc tẩy không sạch gây tranh chấp -> selected = None, status = 'MULTIPLE_INVALID'.

    Parameters:
        fill_ratios: Dict mapping tên phương án -> fill ratio, ví dụ {'A': 0.12, 'B': 0.85, 'C': 0.05, 'D': 0.08}
        question_id: Số thứ tự câu hỏi (mặc định 1).
        tau_min: Ngưỡng tô hợp lệ cơ sở (mặc định 0.38).
        delta_tau_min: Ngưỡng chênh lệch phân định vết tẩy xóa (mặc định 0.15).

    Returns:
        QuestionResult chứa đầy đủ thông tin phân loại, cờ hợp lệ và cảnh báo nếu có.
    """
    if not fill_ratios:
        return QuestionResult(
            question_id=question_id,
            selected=None,
            status=STATUS_BLANK,
            fill_ratios={},
            delta_tau=0.0,
            sorted_options=[],
            is_valid=False,
            warning="Không có dữ liệu phương án.",
        )

    # Sắp xếp các phương án theo fill ratio giảm dần
    sorted_options = sorted(fill_ratios.items(), key=lambda item: item[1], reverse=True)
    best_opt, tau_1 = sorted_options[0]

    if len(sorted_options) > 1:
        second_opt, tau_2 = sorted_options[1]
        delta_tau = tau_1 - tau_2
    else:
        second_opt, tau_2 = None, 0.0
        delta_tau = tau_1

    # Kiểm tra các điều kiện
    if tau_1 < tau_min:
        # Trường hợp 1: Không có ô nào đạt ngưỡng
        return QuestionResult(
            question_id=question_id,
            selected=None,
            status=STATUS_BLANK,
            fill_ratios=dict(fill_ratios),
            delta_tau=delta_tau,
            sorted_options=sorted_options,
            is_valid=False,
            warning=None,
        )

    if tau_2 < tau_min:
        # Trường hợp 2: Duy nhất 1 ô đạt ngưỡng -> Tô chuẩn
        return QuestionResult(
            question_id=question_id,
            selected=best_opt,
            status=STATUS_VALID,
            fill_ratios=dict(fill_ratios),
            delta_tau=delta_tau,
            sorted_options=sorted_options,
            is_valid=True,
            warning=None,
        )

    # Cả 2 ô đều đạt ngưỡng tau_min
    if delta_tau >= delta_tau_min:
        # Trường hợp 3: Có ô thứ hai vượt ngưỡng nhưng ô thứ nhất đậm hơn rõ rệt (vết tẩy xóa)
        warning_msg = (
            f"Phát hiện vết tẩy xóa tại ô {second_opt} (tau={tau_2:.3f}); "
            f"chọn ô {best_opt} (tau={tau_1:.3f}, delta={delta_tau:.3f} >= {delta_tau_min:.2f})"
        )
        return QuestionResult(
            question_id=question_id,
            selected=best_opt,
            status=STATUS_ERASED_VALID,
            fill_ratios=dict(fill_ratios),
            delta_tau=delta_tau,
            sorted_options=sorted_options,
            is_valid=True,
            warning=warning_msg,
        )

    # Trường hợp 4: Cả 2 ô đều đạt ngưỡng và delta_tau < delta_tau_min -> Phạm quy
    warning_msg = (
        f"Phạm quy tô đúp hoặc tẩy không sạch giữa {best_opt} (tau={tau_1:.3f}) "
        f"và {second_opt} (tau={tau_2:.3f}); delta={delta_tau:.3f} < {delta_tau_min:.2f}"
    )
    return QuestionResult(
        question_id=question_id,
        selected=None,
        status=STATUS_MULTIPLE_INVALID,
        fill_ratios=dict(fill_ratios),
        delta_tau=delta_tau,
        sorted_options=sorted_options,
        is_valid=False,
        warning=warning_msg,
    )


def generate_grid_bubble_rois(
    start_x: float,
    start_y: float,
    step_x: float,
    step_y: float,
    num_questions: int = 20,
    options: Sequence[str] = DEFAULT_OPTIONS,
    num_columns: int = 1,
    col_spacing_x: float = 0.0,
    col_start_y: Optional[float] = None,
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Sinh lưới tọa độ tâm các ô trắc nghiệm chuẩn theo cấu trúc phiếu thi.

    Hỗ trợ cả phiếu dạng 1 cột dọc (1..N) hoặc nhiều cột song song
    (ví dụ: cột 1 từ câu 1-10, cột 2 từ câu 11-20).

    Parameters:
        start_x: Tọa độ x của ô (Câu 1, Phương án A).
        start_y: Tọa độ y của ô (Câu 1, Phương án A).
        step_x: Khoảng cách x giữa các phương án A -> B -> C -> D.
        step_y: Khoảng cách y giữa các hàng câu hỏi liên tiếp.
        num_questions: Tổng số câu hỏi (ví dụ: 20).
        options: Danh sách ký hiệu phương án, mặc định ('A', 'B', 'C', 'D').
        num_columns: Số cột lớn chia đề (1, 2, ...).
        col_spacing_x: Khoảng cách giữa các cột lớn theo phương x.
        col_start_y: Tọa độ y bắt đầu của các cột sau (nếu None thì lấy bằng start_y).

    Returns:
        grid_rois: Dict mapping question_id -> {option_str -> (cx, cy)}
    """
    grid: Dict[int, Dict[str, Tuple[float, float]]] = {}
    questions_per_col = math.ceil(num_questions / num_columns)
    col_base_y = start_y if col_start_y is None else col_start_y

    for q_idx in range(1, num_questions + 1):
        # Xác định câu hỏi thuộc cột nào
        col_idx = (q_idx - 1) // questions_per_col
        row_in_col = (q_idx - 1) % questions_per_col

        q_start_x = start_x + col_idx * col_spacing_x
        q_start_y = col_base_y + row_in_col * step_y

        opt_dict: Dict[str, Tuple[float, float]] = {}
        for opt_idx, opt_name in enumerate(options):
            cx = q_start_x + opt_idx * step_x
            cy = q_start_y
            opt_dict[opt_name] = (float(cx), float(cy))

        grid[q_idx] = opt_dict

    return grid


def _cluster_points_by_y(
    pts: np.ndarray,
    y_cluster_tol: float,
) -> List[List[Tuple[float, float]]]:
    """Gom nhóm danh sách điểm (N, 2) thành các hàng theo tọa độ y tăng dần."""
    if len(pts) == 0:
        return []
    sorted_pts = pts[np.argsort(pts[:, 1])]
    rows: List[List[Tuple[float, float]]] = []
    current_row: List[Tuple[float, float]] = [(float(sorted_pts[0, 0]), float(sorted_pts[0, 1]))]
    current_y = float(sorted_pts[0, 1])

    for pt in sorted_pts[1:]:
        x, y = float(pt[0]), float(pt[1])
        if abs(y - current_y) <= y_cluster_tol:
            current_row.append((x, y))
            current_y = sum(p[1] for p in current_row) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [(x, y)]
            current_y = y

    if current_row:
        rows.append(current_row)

    rows.sort(key=lambda row_pts: sum(p[1] for p in row_pts) / len(row_pts))
    return rows


def _estimate_column_canonical_xs(
    col_rows: List[List[Tuple[float, float]]],
    num_options: int,
) -> Tuple[List[float], float]:
    """Ước lượng tọa độ x chuẩn cho từng phương án (A, B, C, D) trong một cột."""
    complete_rows = [sorted(r, key=lambda p: p[0]) for r in col_rows if len(r) == num_options]
    if complete_rows:
        xs_mat = np.array([[p[0] for p in r] for r in complete_rows], dtype=np.float64)
        canonical_xs = [float(v) for v in np.median(xs_mat, axis=0)]
        diffs = np.diff(canonical_xs)
        step_x = float(np.median(diffs)) if len(diffs) > 0 and np.median(diffs) > 0 else 40.0
        return canonical_xs, step_x

    # Fallback khi không có hàng nào đủ cả num_options điểm: gom cụm tọa độ x
    all_xs = sorted(p[0] for r in col_rows for p in r)
    if not all_xs:
        return [float(k * 40.0) for k in range(num_options)], 40.0

    # Tính khoảng cách giữa các điểm liên tiếp trong từng hàng
    intra_dxs = []
    for r in col_rows:
        sr = sorted(p[0] for p in r)
        for i in range(len(sr) - 1):
            dx = sr[i + 1] - sr[i]
            if dx > 5.0:
                intra_dxs.append(dx)
    step_x = float(np.median(intra_dxs)) if intra_dxs else 40.0

    # Gom cụm theo x với ngưỡng 0.45 * step_x
    x_clusters: List[List[float]] = [[all_xs[0]]]
    for x in all_xs[1:]:
        if abs(x - float(np.mean(x_clusters[-1]))) <= 0.45 * step_x:
            x_clusters[-1].append(x)
        else:
            x_clusters.append([x])

    cluster_centers = [float(np.median(c)) for c in x_clusters]
    if len(cluster_centers) >= num_options:
        # Chọn num_options cụm có nhiều điểm nhất và sắp xếp theo x
        ranked = sorted(zip(x_clusters, cluster_centers), key=lambda item: len(item[0]), reverse=True)[:num_options]
        canonical_xs = sorted(c for _, c in ranked)
        diffs = np.diff(canonical_xs)
        if len(diffs) > 0 and np.median(diffs) > 0:
            step_x = float(np.median(diffs))
        return canonical_xs, step_x

    start_x = cluster_centers[0]
    canonical_xs = [start_x + k * step_x for k in range(num_options)]
    return canonical_xs, step_x


def auto_align_grid_from_centroids(
    centroids: Sequence[Tuple[float, float]],
    num_questions: int = 20,
    options: Sequence[str] = DEFAULT_OPTIONS,
    y_cluster_tol: Optional[float] = None,
    num_columns: Optional[int] = None,
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Tự động phân nhóm và sắp xếp danh sách các tâm (cx, cy) phát hiện từ CCL thành lưới OMR.

    Hỗ trợ cả phiếu 1 cột và phiếu nhiều cột (ví dụ 2 cột: câu 1-10 và 11-20),
    đồng thời khớp theo tọa độ x chuẩn của từng phương án A/B/C/D để không bị lệch nhãn
    khi một hàng bị thiếu ô (ví dụ thiếu ô A).

    Parameters:
        centroids: Danh sách các cặp tọa độ (cx, cy).
        num_questions: Số lượng câu hỏi dự kiến.
        options: Danh sách phương án mỗi câu (mặc định A, B, C, D).
        y_cluster_tol: Dung sai khoảng cách y để gộp chung một hàng. Nếu None, tự động ước tính.
        num_columns: Số cột lớn của phiếu (1, 2, ...). Nếu None, tự động phát hiện từ phân bố x/y.

    Returns:
        grid: Dict mapping question_id -> {option -> (cx, cy)}
    """
    if not centroids:
        raise ValueError("Danh sách centroids rỗng.")

    pts = np.array(centroids, dtype=np.float64)
    num_opts = len(options)

    # Ước lượng y_cluster_tol nếu chưa được cung cấp
    if y_cluster_tol is None:
        sorted_y = np.sort(pts[:, 1])
        if len(sorted_y) > 1:
            diffs_y = np.diff(sorted_y)
            pos_diffs = diffs_y[diffs_y > 2.0]
            y_cluster_tol = float(np.median(pos_diffs) * 0.5) if len(pos_diffs) > 0 else 10.0
        else:
            y_cluster_tol = 10.0

    # Phân cụm sơ bộ theo y để ước lượng khoảng cách giữa các ô trong cùng hàng & phát hiện đa cột
    initial_bands = _cluster_points_by_y(pts, y_cluster_tol)

    intra_dxs: List[float] = []
    for band in initial_bands:
        sb = sorted(p[0] for p in band)
        for i in range(len(sb) - 1):
            dx = sb[i + 1] - sb[i]
            if dx > 5.0:
                intra_dxs.append(dx)
    # Phần lớn khoảng cách x liên tiếp là giữa A-B-C-D; lấy phân vị 40% để tránh lẫn khoảng cách giữa 2 cột
    step_x_est = float(np.percentile(intra_dxs, 40)) if intra_dxs else 40.0

    # Tách cột lớn theo trục x nếu là phiếu nhiều cột
    sorted_all_x = np.sort(pts[:, 0])
    x_gaps = np.diff(sorted_all_x)

    if num_columns is None:
        max_band_len = max((len(b) for b in initial_bands), default=0)
        has_wide_gap = bool(len(x_gaps) > 0 and float(np.max(x_gaps)) > 1.8 * step_x_est)
        if has_wide_gap and (max_band_len > num_opts or (num_questions > 10 and len(initial_bands) <= math.ceil(num_questions * 0.75))):
            detected_cols = 2
        else:
            detected_cols = 1
    else:
        detected_cols = max(1, int(num_columns))

    if detected_cols > 1 and len(x_gaps) >= detected_cols - 1:
        # Tìm (detected_cols - 1) khoảng trống lớn nhất trên trục x làm ranh giới chia cột
        split_indices = np.argsort(x_gaps)[-(detected_cols - 1):]
        split_thresholds = sorted(
            float((sorted_all_x[idx] + sorted_all_x[idx + 1]) * 0.5)
            for idx in split_indices
        )
        col_point_groups: List[np.ndarray] = []
        prev_thresh = -float("inf")
        for thresh in split_thresholds:
            mask = (pts[:, 0] > prev_thresh) & (pts[:, 0] <= thresh)
            if np.any(mask):
                col_point_groups.append(pts[mask])
            prev_thresh = thresh
        tail_mask = pts[:, 0] > prev_thresh
        if np.any(tail_mask):
            col_point_groups.append(pts[tail_mask])
    else:
        col_point_groups = [pts]

    # Ghép từng hàng trong mỗi cột thành câu hỏi theo tọa độ x chuẩn
    grid: Dict[int, Dict[str, Tuple[float, float]]] = {}
    q_counter = 1

    for col_pts in col_point_groups:
        if q_counter > num_questions:
            break
        col_rows = _cluster_points_by_y(col_pts, y_cluster_tol)
        canonical_xs, step_x = _estimate_column_canonical_xs(col_rows, num_opts)
        match_tol = max(step_x * 0.55, 10.0)

        for row_pts in col_rows:
            if q_counter > num_questions:
                break
            row_pts_sorted = sorted(row_pts, key=lambda p: p[0])
            row_y = float(sum(p[1] for p in row_pts_sorted) / len(row_pts_sorted))

            # Khớp từng điểm vào slot phương án gần nhất (tránh lệch nhãn khi thiếu ô A)
            assigned: Dict[int, Tuple[float, float]] = {}
            candidates: List[Tuple[float, int, Tuple[float, float]]] = []
            for p_idx, pt in enumerate(row_pts_sorted):
                for opt_idx, cx_canon in enumerate(canonical_xs):
                    dist = abs(pt[0] - cx_canon)
                    if dist <= match_tol:
                        candidates.append((dist, opt_idx, pt))

            candidates.sort(key=lambda item: item[0])
            used_pts = set()
            for dist, opt_idx, pt in candidates:
                pt_key = (pt[0], pt[1])
                if opt_idx not in assigned and pt_key not in used_pts:
                    assigned[opt_idx] = (float(pt[0]), float(pt[1]))
                    used_pts.add(pt_key)

            opt_dict: Dict[str, Tuple[float, float]] = {}
            for opt_idx, opt_name in enumerate(options):
                if opt_idx in assigned:
                    opt_dict[opt_name] = assigned[opt_idx]
                else:
                    # Nội suy tọa độ ô bị thiếu từ canonical_xs và y trung bình hàng
                    opt_dict[opt_name] = (float(canonical_xs[opt_idx]), row_y)

            grid[q_counter] = opt_dict
            q_counter += 1

    return grid


def recognize_all_questions(
    binary_image: np.ndarray,
    question_grid: Dict[int, Dict[str, Tuple[float, float]]],
    radius: float = DEFAULT_BUBBLE_RADIUS,
    tau_min: float = DEFAULT_TAU_MIN,
    delta_tau_min: float = DEFAULT_DELTA_TAU_MIN,
) -> OMRRecognitionResult:
    """Nhận diện toàn bộ các câu hỏi trên phiếu thi OMR.

    Parameters:
        binary_image: Ảnh nhị phân phẳng (H, W).
        question_grid: Lưới tọa độ các ô mapping question_id -> {option -> (cx, cy)}.
        radius: Bán kính ô tròn R (mặc định 12.0).
        tau_min: Ngưỡng tô hợp lệ cơ sở (mặc định 0.38).
        delta_tau_min: Ngưỡng phân định vết tẩy xóa (mặc định 0.15).

    Returns:
        OMRRecognitionResult chứa vector đáp án, chi tiết từng câu và bảng tổng hợp trạng thái.
    """
    questions: Dict[int, QuestionResult] = {}
    answers_vector: Dict[int, Optional[str]] = {}
    summary = {
        STATUS_VALID: 0,
        STATUS_BLANK: 0,
        STATUS_ERASED_VALID: 0,
        STATUS_MULTIPLE_INVALID: 0,
    }

    # Tính ngưỡng nhị phân một lần duy nhất cho toàn bộ ảnh
    if binary_image.dtype == bool or binary_image.size == 0:
        bin_thresh: Optional[float] = None
    else:
        bin_thresh = 127.0 if float(np.max(binary_image)) > 1.0 else 0.5

    # Sắp xếp theo thứ tự câu hỏi tăng dần
    sorted_q_ids = sorted(question_grid.keys())

    for q_id in sorted_q_ids:
        opt_coords = question_grid[q_id]
        fill_ratios: Dict[str, float] = {}

        for opt_name, center in opt_coords.items():
            ratio = compute_bubble_fill_ratio_scratch(
                binary_image=binary_image,
                center=center,
                radius=radius,
                use_local_patch=True,
                binary_threshold=bin_thresh,
            )
            fill_ratios[opt_name] = ratio

        q_res = classify_question_bubbles(
            fill_ratios=fill_ratios,
            question_id=q_id,
            tau_min=tau_min,
            delta_tau_min=delta_tau_min,
        )

        questions[q_id] = q_res
        answers_vector[q_id] = q_res.selected
        summary[q_res.status] = summary.get(q_res.status, 0) + 1

    return OMRRecognitionResult(
        questions=questions,
        answers_vector=answers_vector,
        summary=summary,
    )
