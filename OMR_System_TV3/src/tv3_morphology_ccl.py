"""
Module: tv3_morphology_ccl.py
Thành viên phụ trách: Thành viên 3 (Phân đoạn ảnh & Nhận diện ô tô)
Đề tài: Hệ thống Chấm bài Trắc nghiệm Tự động từ Ảnh chụp (OMR) - Nhóm 4

Nội dung cài đặt:
1. Toán tử Hình thái học nhị phân 100% NumPy from scratch:
   - Phần tử cấu trúc (Structuring Element): RECT, CROSS, ELLIPSE.
   - 4 phép toán cơ bản: Erode (Co), Dilate (Giãn), Opening (Mở), Closing (Đóng).
   - Tối ưu hóa bằng trượt cửa sổ vector hóa (sliding_window_view) và padding chuẩn.
2. Gán nhãn thành phần liên thông 2 lượt (Two-Pass CCL) & Cấu trúc Disjoint-Set (Union-Find):
   - Cấu trúc DisjointSet với nén đường dẫn (path compression) và ưu tiên nhãn nhỏ hơn.
   - Thuật toán Two-Pass Connected Component Labeling hỗ trợ 4-lân cận và 8-lân cận.
   - Trích xuất thông số hình học từng vùng liên thông: area, bbox, centroid, aspect_ratio, extent.
   - Hàm lọc ứng viên ô tô đáp án OMR (filter_bubble_components).
3. Hàm đối chứng Benchmark với OpenCV:
   - So sánh định lượng: IoU, Dice Score, MSE, độ lệch tâm, độ lệch bbox, thời gian thực thi.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import time
import numpy as np
import cv2


# =============================================================================
# PHẦN 1: PHẦN TỬ CẤU TRÚC (STRUCTURING ELEMENT) TỪ ĐẦU BẰNG NUMPY
# =============================================================================

def create_structuring_element(
    shape: str = "RECT",
    ksize: Union[int, Tuple[int, int]] = (3, 3),
    dtype: np.dtype = np.uint8
) -> np.ndarray:
    """
    Tạo phần tử cấu trúc (Structuring Element / Kernel) nhị phân từ đầu bằng NumPy.
    Khớp 100% chuẩn hình học của OpenCV (cv2.getStructuringElement).

    Args:
        shape: Kiểu hình học phần tử cấu trúc:
               - 'RECT' | 'rect': Hình chữ nhật toàn 1.
               - 'CROSS' | 'cross': Hình chữ thập (tâm chữ thập = 1).
               - 'ELLIPSE' | 'ellipse': Hình elip / đường tròn nội tiếp khung ksize.
        ksize: Kích thước kernel dạng int lẻ (k) hoặc tuple (kh, kw) lẻ.
        dtype: Kiểu dữ liệu mảng trả về (mặc định np.uint8).

    Returns:
        np.ndarray: Ma trận nhị phân kích thước (kh, kw) chứa các giá trị 0 và 1.
    """
    if isinstance(ksize, int):
        kh, kw = ksize, ksize
    elif isinstance(ksize, (tuple, list)) and len(ksize) == 2:
        kh, kw = int(ksize[0]), int(ksize[1])
    else:
        raise ValueError("ksize phải là số nguyên dương lẻ hoặc tuple (kh, kw) số nguyên dương lẻ.")

    if kh <= 0 or kw <= 0 or kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"Kích thước kernel phải là các số nguyên dương lẻ, nhận được ({kh}, {kw}).")

    shape_norm = shape.strip().upper()
    element = np.zeros((kh, kw), dtype=dtype)

    if shape_norm in ("RECT", "RECTANGLE"):
        element[:, :] = 1

    elif shape_norm in ("CROSS", "+"):
        r_center = kh // 2
        c_center = kw // 2
        element[r_center, :] = 1
        element[:, c_center] = 1

    elif shape_norm in ("ELLIPSE", "OVAL", "CIRCLE"):
        r = kh // 2
        c = kw // 2
        # Cài đặt theo chuẩn toán học elip của OpenCV:
        # Với mỗi hàng dy = i - r, tính độ rộng dx = c * sqrt(1 - dy^2 / r^2)
        for i in range(kh):
            dy = i - r
            if r > 0:
                val = 1.0 - (dy * dy) / float(r * r)
                dx = int(round(c * np.sqrt(max(0.0, val))))
            else:
                dx = c
            col_start = max(0, c - dx)
            col_end = min(kw, c + dx + 1)
            element[i, col_start:col_end] = 1

    else:
        raise ValueError(f"Kiểu shape '{shape}' không được hỗ trợ. Hãy chọn 'RECT', 'CROSS', hoặc 'ELLIPSE'.")

    return element


# =============================================================================
# PHẦN 2: TOÁN TỬ HÌNH THÁI HỌC NHỊ PHÂN (BINARY MORPHOLOGY FROM SCRATCH)
# =============================================================================

def _normalize_binary_input(img_bin: np.ndarray) -> Tuple[np.ndarray, int, np.dtype]:
    """
    Chuẩn hóa ảnh nhị phân đầu vào, xác định giá trị tiền cảnh (foreground scale 1 hoặc 255).
    """
    if not isinstance(img_bin, np.ndarray):
        raise TypeError("Ảnh đầu vào phải là numpy.ndarray.")
    if img_bin.ndim != 2:
        raise ValueError(f"Ảnh đầu vào phải là ảnh 2D, nhận được mảng có {img_bin.ndim} chiều.")

    orig_dtype = img_bin.dtype
    max_val = int(img_bin.max()) if img_bin.size > 0 else 0
    fg_val = 255 if max_val > 1 else 1

    return img_bin, fg_val, orig_dtype


def erode_binary_scratch(
    img_bin: np.ndarray,
    kernel: Optional[np.ndarray] = None,
    border_value: Optional[int] = None
) -> np.ndarray:
    """
    Toán tử Co nhị phân (Binary Erosion): (A ⊖ B)(x, y) = min_{s in B} A(x+s, y+s)
    Làm mảnh viền nét, xóa các chấm nhiễu cô lập và vết lem nhỏ.

    Áp dụng kỹ thuật trượt cửa sổ vector hóa np.lib.stride_tricks.sliding_window_view
    kết hợp padding chuẩn biên (constant value = max_val / 255) đúng chuẩn toán học và OpenCV.

    Args:
        img_bin: Ảnh nhị phân 2D (tiền cảnh > 0).
        kernel: Phần tử cấu trúc nhị phân (mặc định RECT 3x3 nếu None).
        border_value: Giá trị đệm biên (mặc định = fg_val để không bào mòn biên ảnh).

    Returns:
        np.ndarray: Ảnh sau phép Co (cùng kích thước và kiểu dữ liệu với ảnh gốc).
    """
    img, fg_val, orig_dtype = _normalize_binary_input(img_bin)

    if kernel is None:
        kernel = create_structuring_element("RECT", (3, 3))
    kh, kw = kernel.shape
    ph, pw = (kh - 1) // 2, (kw - 1) // 2

    # Giá trị padding chuẩn cho Erosion: biên vô hạn = fg_val để không làm hẹp biên ảnh giả
    pad_val = fg_val if border_value is None else border_value

    # Đệm biên đối xứng hoặc hằng số
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='constant', constant_values=pad_val)

    # Trượt cửa sổ ma trận dạng vector hóa (không dùng vòng lặp for thủ công)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))

    # Với phép Co nhị phân: mọi điểm trong cửa sổ tại vị trí kernel == 1 đều phải là foreground
    k_mask = (kernel > 0)
    result = np.where(np.all(windows[..., k_mask] > 0, axis=-1), fg_val, 0)

    return result.astype(orig_dtype)


def dilate_binary_scratch(
    img_bin: np.ndarray,
    kernel: Optional[np.ndarray] = None,
    border_value: Optional[int] = None
) -> np.ndarray:
    """
    Toán tử Giãn nhị phân (Binary Dilation): (A ⊕ B)(x, y) = max_{s in B} A(x+s, y+s)
    Làm dày nét bút chì, lấp đầy các khe hở và làm liền các nét đứt.

    Args:
        img_bin: Ảnh nhị phân 2D (tiền cảnh > 0).
        kernel: Phần tử cấu trúc nhị phân (mặc định RECT 3x3 nếu None).
        border_value: Giá trị đệm biên (mặc định = 0).

    Returns:
        np.ndarray: Ảnh sau phép Giãn.
    """
    img, fg_val, orig_dtype = _normalize_binary_input(img_bin)

    if kernel is None:
        kernel = create_structuring_element("RECT", (3, 3))
    kh, kw = kernel.shape
    ph, pw = (kh - 1) // 2, (kw - 1) // 2

    pad_val = 0 if border_value is None else border_value
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='constant', constant_values=pad_val)

    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))

    # Với phép Giãn nhị phân: chỉ cần ít nhất 1 điểm trong cửa sổ tại vị trí kernel == 1 là foreground
    k_mask = (kernel > 0)
    result = np.where(np.any(windows[..., k_mask] > 0, axis=-1), fg_val, 0)

    return result.astype(orig_dtype)


def opening_binary_scratch(
    img_bin: np.ndarray,
    kernel: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Toán tử Mở nhị phân (Binary Opening): A ∘ B = (A ⊖ B) ⊕ B
    Co trước rồi Giãn sau: Khử nhiễu muối tiêu, gọt bỏ gai nhọn và xóa nét vẽ mảnh thừa.

    Args:
        img_bin: Ảnh nhị phân 2D.
        kernel: Phần tử cấu trúc (mặc định RECT 3x3).

    Returns:
        np.ndarray: Ảnh sau phép Mở.
    """
    if kernel is None:
        kernel = create_structuring_element("RECT", (3, 3))
    eroded = erode_binary_scratch(img_bin, kernel)
    return dilate_binary_scratch(eroded, kernel)


def closing_binary_scratch(
    img_bin: np.ndarray,
    kernel: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Toán tử Đóng nhị phân (Binary Closing): A • B = (A ⊕ B) ⊖ B
    Giãn trước rồi Co sau: Lấp đầy các lỗ thủng nhỏ bên trong ô tô chì và nối liền đoạn hở.

    Args:
        img_bin: Ảnh nhị phân 2D.
        kernel: Phần tử cấu trúc (mặc định RECT 3x3).

    Returns:
        np.ndarray: Ảnh sau phép Đóng.
    """
    if kernel is None:
        kernel = create_structuring_element("RECT", (3, 3))
    dilated = dilate_binary_scratch(img_bin, kernel)
    return erode_binary_scratch(dilated, kernel)


def morphology_binary_scratch(
    img_bin: np.ndarray,
    op: str = "open",
    kernel: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Giao diện hợp nhất thực hiện 4 phép toán hình thái học nhị phân.

    Args:
        img_bin: Ảnh nhị phân 2D.
        op: Tên phép toán: 'erode', 'dilate', 'open', 'close'.
        kernel: Phần tử cấu trúc (mặc định RECT 3x3).

    Returns:
        np.ndarray: Ảnh kết quả.
    """
    op_norm = op.strip().lower()
    if op_norm in ("erode", "erosion"):
        return erode_binary_scratch(img_bin, kernel)
    elif op_norm in ("dilate", "dilation"):
        return dilate_binary_scratch(img_bin, kernel)
    elif op_norm in ("open", "opening"):
        return opening_binary_scratch(img_bin, kernel)
    elif op_norm in ("close", "closing"):
        return closing_binary_scratch(img_bin, kernel)
    else:
        raise ValueError(f"Toán tử '{op}' không hợp lệ. Chọn 'erode', 'dilate', 'open', hoặc 'close'.")


# =============================================================================
# PHẦN 3: GÁN NHÃN THÀNH PHẦN LIÊN THÔNG (TWO-PASS CCL & DISJOINT-SET)
# =============================================================================

class DisjointSet:
    """
    Cấu trúc dữ liệu Disjoint-Set (Union-Find) hỗ trợ gán nhãn thành phần liên thông.
    Tối ưu hóa bằng:
    1. Nén đường dẫn triệt để (Path Compression) lặp không đệ quy (tránh tràn ngăn xếp stack).
    2. Hợp nhất ưu tiên nhãn nhỏ hơn (Union by Smallest Label) giữ trật tự nhãn tăng dần.
    """
    def __init__(self, initial_size: int = 0):
        self.parent: List[int] = list(range(initial_size))

    def add(self) -> int:
        """Thêm một phần tử mới với nhãn tự trỏ vào chính nó."""
        new_id = len(self.parent)
        self.parent.append(new_id)
        return new_id

    def find(self, i: int) -> int:
        """
        Tìm nhãn đại diện gốc (root) kèm nén đường dẫn hai lượt lặp.
        Mọi node trên đường đi đều được trỏ trực tiếp về gốc.
        """
        root = i
        while self.parent[root] != root:
            root = self.parent[root]

        curr = i
        while curr != root:
            parent_node = self.parent[curr]
            self.parent[curr] = root
            curr = parent_node

        return root

    def union(self, i: int, j: int) -> int:
        """
        Hợp nhất 2 tập hợp chứa i và j. Luôn gán node cha là nhãn gốc có giá trị nhỏ hơn.
        """
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if root_i < root_j:
                self.parent[root_j] = root_i
                return root_i
            else:
                self.parent[root_i] = root_j
                return root_j
        return root_i

    def __len__(self) -> int:
        return len(self.parent)


def two_pass_ccl_scratch(
    img_bin: np.ndarray,
    connectivity: int = 8
) -> Tuple[int, np.ndarray, List[Dict[str, Any]]]:
    """
    Thuật toán Gán nhãn thành phần liên thông hai lượt (Two-Pass Connected Component Labeling).
    Cài đặt 100% bằng NumPy from scratch kết hợp cấu trúc Disjoint-Set.

    Hỗ trợ cả 8-lân cận (mặc định) và 4-lân cận.
    Tối ưu hóa duyệt qua tọa độ non-zero theo thứ tự quét raster (np.nonzero)
    và bảng tra cứu (LUT) vector hóa trong Pass 2.

    Args:
        img_bin: Ảnh nhị phân 2D (mức 0 và >0).
        connectivity: Kiểu lân cận (4 hoặc 8, mặc định 8).

    Returns:
        num_components: Số lượng vùng liên thông tiền cảnh K (không tính nền 0).
        labels: Ma trận nhãn int32 cùng kích thước, đánh số 0 (nền), 1, 2, ..., K.
        stats: Danh sách K từ điển chứa thông số hình học từng vùng liên thông:
               - 'label': Nhãn int (1 .. K)
               - 'area': Diện tích (số lượng pixel)
               - 'bbox': Tuple (x, y, w, h) - tọa độ góc trên-trái và kích thước
               - 'centroid': Tuple (cx, cy) - trọng tâm mức pixel thực (float)
               - 'aspect_ratio': Tỷ lệ chiều rộng / chiều cao (w / h)
               - 'extent': Độ đặc = area / (w * h)
    """
    if not isinstance(img_bin, np.ndarray) or img_bin.ndim != 2:
        raise ValueError("Ảnh đầu vào phải là mảng NumPy 2 chiều.")
    if connectivity not in (4, 8):
        raise ValueError("connectivity chỉ chấp nhận giá trị 4 hoặc 8.")

    H, W = img_bin.shape
    labels = np.zeros((H, W), dtype=np.int32)

    # Lấy tọa độ các pixel tiền cảnh theo thứ tự quét raster chuẩn C-order
    nonzero_rows, nonzero_cols = np.nonzero(img_bin > 0)
    num_fg_pixels = len(nonzero_rows)

    if num_fg_pixels == 0:
        return 0, labels, []

    ds = DisjointSet(initial_size=1)  # Label 0 dành riêng cho nền (Background)

    # ---------------------------------------------------------
    # PASS 1: Quét tiến & Gán nhãn tạm thời kèm liên kết DisjointSet
    # ---------------------------------------------------------
    for i in range(num_fg_pixels):
        r = int(nonzero_rows[i])
        c = int(nonzero_cols[i])

        neighbors = []

        # Kiểm tra lân cận 4 hướng trước đó
        if r > 0 and labels[r - 1, c] > 0:
            neighbors.append(labels[r - 1, c])
        if c > 0 and labels[r, c - 1] > 0:
            neighbors.append(labels[r, c - 1])

        # Bổ sung 2 đường chéo trước đó cho 8-lân cận
        if connectivity == 8:
            if r > 0 and c > 0 and labels[r - 1, c - 1] > 0:
                neighbors.append(labels[r - 1, c - 1])
            if r > 0 and c + 1 < W and labels[r - 1, c + 1] > 0:
                neighbors.append(labels[r - 1, c + 1])

        if not neighbors:
            # Tạo nhãn tạm mới
            new_label = ds.add()
            labels[r, c] = new_label
        else:
            # Gán nhãn nhỏ nhất trong số các lân cận đã duyệt
            min_label = min(neighbors)
            labels[r, c] = min_label
            for n_label in neighbors:
                if n_label != min_label:
                    ds.union(min_label, n_label)

    # ---------------------------------------------------------
    # PASS 2: Tra cứu gốc đại diện & Chuẩn hóa nhãn liên tục 1 .. K
    # ---------------------------------------------------------
    num_raw_labels = len(ds)
    lut = np.zeros(num_raw_labels, dtype=np.int32)
    for lbl in range(1, num_raw_labels):
        lut[lbl] = ds.find(lbl)

    # Ánh xạ nhãn gốc về trật tự xuất hiện đầu tiên trong ảnh
    raw_roots = lut[labels]
    unique_roots, first_indices = np.unique(raw_roots, return_index=True)

    valid_mask = unique_roots > 0
    unique_roots = unique_roots[valid_mask]
    first_indices = first_indices[valid_mask]

    # Sắp xếp theo thứ tự quét raster xuất hiện đầu tiên
    ordered_roots = unique_roots[np.argsort(first_indices)]
    num_components = len(ordered_roots)

    # Tạo bảng tra cứu ánh xạ nhãn cuối cùng: raw_label -> continuous label
    final_lut = np.zeros(num_raw_labels, dtype=np.int32)
    for new_id, root_val in enumerate(ordered_roots, start=1):
        final_lut[lut == root_val] = new_id

    # Cập nhật toàn bộ ma trận nhãn bằng phép chỉ mục mảng NumPy tốc độ C
    final_labels = final_lut[labels]

    # ---------------------------------------------------------
    # TRÍCH XUẤT THÔNG SỐ HÌNH HỌC (STATS) TỪNG VÙNG LIÊN THÔNG
    # ---------------------------------------------------------
    fg_r, fg_c = np.nonzero(final_labels > 0)
    fg_lbl = final_labels[fg_r, fg_c]

    areas = np.bincount(fg_lbl, minlength=num_components + 1)
    sum_r = np.bincount(fg_lbl, weights=fg_r, minlength=num_components + 1)
    sum_c = np.bincount(fg_lbl, weights=fg_c, minlength=num_components + 1)

    min_r = np.full(num_components + 1, H, dtype=np.int32)
    max_r = np.full(num_components + 1, -1, dtype=np.int32)
    min_c = np.full(num_components + 1, W, dtype=np.int32)
    max_c = np.full(num_components + 1, -1, dtype=np.int32)

    np.minimum.at(min_r, fg_lbl, fg_r)
    np.maximum.at(max_r, fg_lbl, fg_r)
    np.minimum.at(min_c, fg_lbl, fg_c)
    np.maximum.at(max_c, fg_lbl, fg_c)

    stats: List[Dict[str, Any]] = []
    for k in range(1, num_components + 1):
        area_k = int(areas[k])
        if area_k == 0:
            continue
        c_min = int(min_c[k])
        r_min = int(min_r[k])
        c_max = int(max_c[k])
        r_max = int(max_r[k])

        w = int(c_max - c_min + 1)
        h = int(r_max - r_min + 1)
        cx = float(sum_c[k] / area_k)
        cy = float(sum_r[k] / area_k)
        aspect_ratio = float(w / max(1, h))
        extent = float(area_k / (w * h))

        stats.append({
            "label": k,
            "area": area_k,
            "bbox": (c_min, r_min, w, h),
            "centroid": (cx, cy),
            "aspect_ratio": aspect_ratio,
            "extent": extent
        })

    return num_components, final_labels, stats


def filter_bubble_components(
    labels: np.ndarray,
    stats: List[Dict[str, Any]],
    min_area: int = 40,
    max_area: int = 2000,
    aspect_ratio_range: Tuple[float, float] = (0.6, 1.6),
    min_extent: float = 0.40,
    max_extent: float = 1.0
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Lọc danh sách các thành phần liên thông để chỉ giữ lại các ứng viên ô tròn trắc nghiệm (OMR Bubbles).
    Tự động loại bỏ:
    - Nhiễu hạt muối tiêu hoặc chấm bụi chì nhỏ (area < min_area).
    - Đường viền bảng, khung bao, chữ in tiêu đề (area > max_area hoặc aspect_ratio quá méo).
    - Các vệt xước, nét gạch dài không đặc (extent < min_extent).

    Args:
        labels: Ma trận nhãn từ two_pass_ccl_scratch.
        stats: Danh sách thông số các thành phần từ two_pass_ccl_scratch.
        min_area: Diện tích tối thiểu (pixel).
        max_area: Diện tích tối đa (pixel).
        aspect_ratio_range: Khoảng tỷ lệ w/h hợp lệ (gần 1.0 với hình tròn/vuông).
        min_extent: Độ đặc tối thiểu (diện tích / diện tích bbox).
        max_extent: Độ đặc tối đa.

    Returns:
        filtered_labels: Ma trận nhãn mới đã lọc bỏ các thành phần không hợp lệ.
        filtered_stats: Danh sách thông số các ô đáp án hợp lệ.
    """
    min_ar, max_ar = aspect_ratio_range
    filtered_stats: List[Dict[str, Any]] = []
    valid_labels_set = set()

    for item in stats:
        area = item["area"]
        ar = item["aspect_ratio"]
        ext = item["extent"]

        if (min_area <= area <= max_area and
            min_ar <= ar <= max_ar and
            min_extent <= ext <= max_extent):
            filtered_stats.append(item)
            valid_labels_set.add(item["label"])

    # Tạo ma trận nhãn đã lọc
    filtered_labels = np.zeros_like(labels)
    # Ánh xạ nhãn mới liên tục 1 .. M
    new_stats: List[Dict[str, Any]] = []
    for new_idx, item in enumerate(filtered_stats, start=1):
        old_lbl = item["label"]
        filtered_labels[labels == old_lbl] = new_idx
        item_copy = dict(item)
        item_copy["label"] = new_idx
        new_stats.append(item_copy)

    return filtered_labels, new_stats


# =============================================================================
# PHẦN 4: HÀM ĐỐI CHỨNG VÀ BENCHMARK VỚI OPENCV
# =============================================================================

def calc_binary_metrics(
    mask_scratch: np.ndarray,
    mask_opencv: np.ndarray
) -> Dict[str, float]:
    """
    Tính toán các chỉ số định lượng so sánh giữa 2 mặt nạ nhị phân:
    - IoU (Jaccard Index) (%)
    - Dice Score (%)
    - Mean Squared Error (MSE)
    - PSNR (dB)
    - Số pixel sai khác (diff_pixels)
    """
    m1 = (mask_scratch > 0).astype(np.uint8)
    m2 = (mask_opencv > 0).astype(np.uint8)

    intersection = np.logical_and(m1 == 1, m2 == 1).sum()
    union = np.logical_or(m1 == 1, m2 == 1).sum()

    if union == 0:
        iou = 100.0
        dice = 100.0
    else:
        iou = float(intersection / union) * 100.0
        dice = float(2.0 * intersection / (m1.sum() + m2.sum())) * 100.0

    diff_pixels = int(np.count_nonzero(m1 != m2))
    mse = float(np.mean((m1.astype(float) - m2.astype(float)) ** 2))

    if mse == 0.0:
        psnr = float('inf')
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))

    return {
        "iou_percent": iou,
        "dice_percent": dice,
        "mse": mse,
        "psnr_db": psnr,
        "diff_pixels": diff_pixels
    }


def benchmark_morphology_vs_opencv(
    img_bin: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = (3, 3),
    shape: str = "RECT",
    op: str = "all",
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Thực nghiệm đối chứng toàn diện 4 phép toán Hình thái học nhị phân giữa
    Code from Scratch (NumPy) và OpenCV (cv2.erode, cv2.dilate, cv2.morphologyEx).

    Args:
        img_bin: Ảnh nhị phân 2D (uint8 0/255).
        kernel_size: Kích thước kernel.
        shape: Kiểu kernel ('RECT', 'CROSS', 'ELLIPSE').
        op: Phép toán cần đo ('erode', 'dilate', 'open', 'close', hoặc 'all').
        num_runs: Số lần lặp lại để lấy thời gian trung bình.

    Returns:
        Dict chứa các chỉ số đối chứng và thời gian chạy ms của từng phép toán.
    """
    k_scratch = create_structuring_element(shape, kernel_size)

    # Tạo kernel tương ứng của OpenCV để đối chứng
    cv_shape_map = {
        "RECT": cv2.MORPH_RECT,
        "CROSS": cv2.MORPH_CROSS,
        "ELLIPSE": cv2.MORPH_ELLIPSE
    }
    cv_kshape = cv_shape_map.get(shape.upper(), cv2.MORPH_RECT)
    kw, kh = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[1], kernel_size[0])
    k_cv = cv2.getStructuringElement(cv_kshape, (kw, kh))

    operations = ["erode", "dilate", "open", "close"] if op == "all" else [op.lower()]
    results: Dict[str, Any] = {}

    for operation in operations:
        # 1. Chạy Scratch
        t0 = time.perf_counter()
        res_scratch = None
        for _ in range(num_runs):
            if operation == "erode":
                res_scratch = erode_binary_scratch(img_bin, k_scratch)
            elif operation == "dilate":
                res_scratch = dilate_binary_scratch(img_bin, k_scratch)
            elif operation == "open":
                res_scratch = opening_binary_scratch(img_bin, k_scratch)
            elif operation == "close":
                res_scratch = closing_binary_scratch(img_bin, k_scratch)
        t_scratch_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

        # 2. Chạy OpenCV
        t0 = time.perf_counter()
        res_cv = None
        for _ in range(num_runs):
            if operation == "erode":
                res_cv = cv2.erode(img_bin, k_cv)
            elif operation == "dilate":
                res_cv = cv2.dilate(img_bin, k_cv)
            elif operation == "open":
                res_cv = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, k_cv)
            elif operation == "close":
                res_cv = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, k_cv)
        t_cv_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

        metrics = calc_binary_metrics(res_scratch, res_cv)
        metrics["time_scratch_ms"] = round(t_scratch_ms, 3)
        metrics["time_opencv_ms"] = round(t_cv_ms, 3)
        metrics["speed_ratio"] = round(t_scratch_ms / max(0.001, t_cv_ms), 2)

        results[operation] = metrics

    return results


def benchmark_ccl_vs_opencv(
    img_bin: np.ndarray,
    connectivity: int = 8,
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Thực nghiệm đối chứng thuật toán Gán nhãn thành phần liên thông Two-Pass Scratch
    với hàm chuẩn của OpenCV (cv2.connectedComponentsWithStats).

    Đánh giá:
    - Độ trùng khớp số lượng thành phần K (count_match: True/False).
    - IoU trung bình giữa các thành phần tương ứng.
    - Sai lệch vị trí trọng tâm cực đại (max_centroid_delta).
    - Sai lệch khung bao cực đại (max_bbox_delta).
    - Thời gian thực thi trung bình (ms).

    Args:
        img_bin: Ảnh nhị phân 2D (uint8).
        connectivity: 4 hoặc 8.
        num_runs: Số lần lặp đo thời gian.

    Returns:
        Dict tổng hợp các chỉ số nghiệm thu.
    """
    # 1. Chạy Scratch
    t0 = time.perf_counter()
    k_sc, labels_sc, stats_sc = 0, None, []
    for _ in range(num_runs):
        k_sc, labels_sc, stats_sc = two_pass_ccl_scratch(img_bin, connectivity=connectivity)
    t_scratch_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # 2. Chạy OpenCV
    t0 = time.perf_counter()
    num_cv, labels_cv, stats_cv, cents_cv = 0, None, None, None
    for _ in range(num_runs):
        num_cv, labels_cv, stats_cv, cents_cv = cv2.connectedComponentsWithStats(
            img_bin, connectivity=connectivity
        )
    t_cv_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    k_cv = num_cv - 1  # Không tính nhãn nền 0

    count_match = (k_sc == k_cv)

    # Đối chứng chi tiết từng vùng liên thông
    ious = []
    centroid_deltas = []
    bbox_deltas = []

    # Tạo từ điển tra cứu stats theo label ID của scratch
    stats_map = {item["label"]: item for item in stats_sc}

    for cv_lbl in range(1, num_cv):
        pts = np.argwhere(labels_cv == cv_lbl)
        if len(pts) == 0:
            continue
        # Lấy nhãn tương ứng ở Scratch tại vị trí đầu tiên
        sc_lbl = int(labels_sc[pts[0, 0], pts[0, 1]])
        if sc_lbl == 0:
            ious.append(0.0)
            continue

        mask_cv = (labels_cv == cv_lbl)
        mask_sc = (labels_sc == sc_lbl)

        inter = np.logical_and(mask_cv, mask_sc).sum()
        union = np.logical_or(mask_cv, mask_sc).sum()
        iou = float(inter / union) if union > 0 else 1.0
        ious.append(iou)

        # So sánh Centroid
        if sc_lbl in stats_map:
            sc_stat = stats_map[sc_lbl]
            cv_cx, cv_cy = float(cents_cv[cv_lbl, 0]), float(cents_cv[cv_lbl, 1])
            sc_cx, sc_cy = sc_stat["centroid"]
            dist = np.sqrt((cv_cx - sc_cx) ** 2 + (cv_cy - sc_cy) ** 2)
            centroid_deltas.append(dist)

            # So sánh Bounding Box (x, y, w, h)
            cv_bx, cv_by, cv_bw, cv_bh = stats_cv[cv_lbl, 0:4]
            sc_bx, sc_by, sc_bw, sc_bh = sc_stat["bbox"]
            b_diff = abs(cv_bx - sc_bx) + abs(cv_by - sc_by) + abs(cv_bw - sc_bw) + abs(cv_bh - sc_bh)
            bbox_deltas.append(b_diff)

    mean_iou = float(np.mean(ious)) * 100.0 if ious else 100.0
    max_centroid_delta = float(np.max(centroid_deltas)) if centroid_deltas else 0.0
    max_bbox_delta = int(np.max(bbox_deltas)) if bbox_deltas else 0

    return {
        "scratch_component_count": k_sc,
        "opencv_component_count": k_cv,
        "count_match": count_match,
        "mean_iou_percent": round(mean_iou, 3),
        "max_centroid_delta_px": round(max_centroid_delta, 4),
        "max_bbox_delta_px": max_bbox_delta,
        "time_scratch_ms": round(t_scratch_ms, 3),
        "time_opencv_ms": round(t_cv_ms, 3),
        "speed_ratio": round(t_scratch_ms / max(0.001, t_cv_ms), 2)
    }
