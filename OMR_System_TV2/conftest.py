"""Cấu hình pytest cho TV2: đảm bảo `import tv2` / `import tv2_perspective` hoạt động
khi chạy pytest từ thư mục OMR_System_TV2 hoặc từ thư mục gốc repo."""

import os
import sys

import numpy as np
import pytest

TV2_DIR = os.path.dirname(os.path.abspath(__file__))
if TV2_DIR not in sys.path:
    sys.path.insert(0, TV2_DIR)

DATA_INPUT_DIR = os.path.join(TV2_DIR, "data", "input")


@pytest.fixture
def anchor_template() -> np.ndarray:
    """Anchor Marker 24x24: ô vuông trắng có lõi đen (khác các marker đen đặc ở 3 góc còn lại)."""
    anchor = np.zeros((24, 24), np.uint8)
    anchor[4:20, 4:20] = 255
    anchor[9:15, 9:15] = 0
    return anchor


@pytest.fixture
def reference_sheet(anchor_template) -> np.ndarray:
    """Phiếu chuẩn 300x400 ở hướng 0°: Anchor ở TR, marker đen ở 3 góc còn lại, nội dung bất đối xứng."""
    sheet = np.full((400, 300), 235, np.uint8)
    sheet[10:34, 266:290] = anchor_template
    for (y, x) in [(10, 10), (366, 10), (366, 266)]:
        sheet[y:y + 24, x:x + 24] = 0
    sheet[150:170, 60:120] = 40
    sheet[250:290, 200:215] = 90
    return sheet


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """Ảnh tổng hợp có lưới ca-rô + hình tròn để kiểm tra độ mượt của nội suy."""
    import cv2
    img = np.zeros((600, 500), dtype=np.uint8)
    for r in range(0, 600, 30):
        img[r:r + 15, :] = 200
    for c in range(0, 500, 30):
        img[:, c:c + 15] = 150
    cv2.circle(img, (250, 300), 80, 255, -1)
    cv2.circle(img, (250, 300), 50, 50, -1)
    return img
