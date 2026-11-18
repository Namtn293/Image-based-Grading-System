"""Cấu hình và hằng số dùng chung của Module TV2."""

from dataclasses import dataclass
from typing import Tuple

# Ngưỡng tương đối để coi hệ DLT là suy biến (sigma_min < tol * sigma_max)
DEGENERATE_TOL = 1e-10

# Thứ tự kênh màu mặc định của ảnh màu đầu vào (cv2.imread trả về BGR)
DEFAULT_COLOR_ORDER = "bgr"


@dataclass(frozen=True)
class PerspectiveConfig:
    """Tham số của pipeline nắn phẳng & định hướng phiếu.

    Attributes:
        target_size: Kích thước ảnh phẳng đích (W, H). None = tự tính từ 4 góc.
        search_margin_ratio: Tỷ lệ cạnh ảnh dùng làm vùng tìm Marker ở mỗi góc.
        color_order: 'bgr' hoặc 'rgb' — thứ tự kênh khi ảnh đầu vào là ảnh màu.
        degenerate_tol: Ngưỡng phát hiện 4 điểm suy biến khi giải Homography.
    """
    target_size: Tuple[int, int] = (800, 1000)
    search_margin_ratio: float = 0.25
    color_order: str = DEFAULT_COLOR_ORDER
    degenerate_tol: float = DEGENERATE_TOL

    def __post_init__(self) -> None:
        if self.target_size is not None:
            w, h = self.target_size
            if int(w) <= 0 or int(h) <= 0:
                raise ValueError(f"target_size phải dương, nhận được {self.target_size}")
        if not 0.0 < self.search_margin_ratio <= 0.5:
            raise ValueError("search_margin_ratio phải thuộc (0, 0.5].")
        if self.color_order.lower() not in ("bgr", "rgb"):
            raise ValueError("color_order chỉ nhận 'bgr' hoặc 'rgb'.")
