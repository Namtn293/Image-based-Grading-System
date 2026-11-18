import numpy as np
import pytest

from tv2 import corners_for_angle, detect_markers_and_correct_rotation

# np.rot90(k) xoay ngược chiều kim đồng hồ -> xoay thuận `angle` độ tương ứng k = (4 - angle/90) % 4
ROTATIONS = [(0, 0), (90, 3), (180, 2), (270, 1)]


@pytest.mark.parametrize("angle, k", ROTATIONS)
def test_detects_angle_and_restores_sheet(reference_sheet, anchor_template, angle, k):
    rotated = np.rot90(reference_sheet, k)
    corrected, detected, scores = detect_markers_and_correct_rotation(rotated, anchor_template)
    assert detected == angle
    np.testing.assert_array_equal(corrected, reference_sheet)
    assert [name for name, _ in scores] == ["TL", "TR", "BR", "BL"]


def test_color_input_with_rgb_order(reference_sheet, anchor_template):
    rgb = np.dstack([reference_sheet] * 3)
    rotated = np.rot90(rgb, 2)
    corrected, detected, _ = detect_markers_and_correct_rotation(rotated, anchor_template, color_order="rgb")
    assert detected == 180 and corrected.shape == rgb.shape


def test_corners_for_angle_rolls_clockwise():
    ordered = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)  # TL, TR, BR, BL
    np.testing.assert_array_equal(corners_for_angle(ordered, 90), ordered[[1, 2, 3, 0]])
    with pytest.raises(ValueError):
        corners_for_angle(ordered, 45)
