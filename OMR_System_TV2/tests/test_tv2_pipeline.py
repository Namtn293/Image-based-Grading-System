import subprocess
import sys

import numpy as np
import pytest

import tv2_perspective
from conftest import TV2_DIR
from tv2 import PerspectiveConfig, process_perspective_pipeline, warp_perspective_sheet


def _place_on_desk(sheet: np.ndarray, margin: int = 50):
    canvas = np.full((sheet.shape[0] + 2 * margin, sheet.shape[1] + 2 * margin), 20, np.uint8)
    canvas[margin:margin + sheet.shape[0], margin:margin + sheet.shape[1]] = sheet
    h, w = sheet.shape
    corners = np.array([[margin, margin], [margin + w - 1, margin],
                        [margin + w - 1, margin + h - 1], [margin, margin + h - 1]], np.float32)
    return canvas, corners


def test_pipeline_without_anchor_keeps_target_size():
    img = np.full((500, 400), 180, np.uint8)
    corners = np.array([[350, 420], [50, 60], [40, 410], [360, 50]], np.float32)
    warped, info = process_perspective_pipeline(img, corners, target_size=(300, 400))
    assert warped.shape == (400, 300)
    assert info["homography_matrix"].shape == (3, 3)
    assert info["detected_angle"] == 0


@pytest.mark.parametrize("angle, k", [(0, 0), (90, 3), (180, 2), (270, 1)])
def test_pipeline_corrects_rotation_without_distortion(reference_sheet, anchor_template, angle, k):
    canvas, corners = _place_on_desk(np.rot90(reference_sheet, k))
    warped, info = process_perspective_pipeline(canvas, corners, anchor_template=anchor_template, target_size=(300, 400))
    assert info["detected_angle"] == angle
    assert warped.shape == (400, 300)
    np.testing.assert_array_equal(warped, reference_sheet)


def test_warp_sheet_with_anchor(reference_sheet, anchor_template):
    canvas, corners = _place_on_desk(np.rot90(reference_sheet, 2))
    out = warp_perspective_sheet(canvas, corners, 300, 400, anchor_template=anchor_template)
    np.testing.assert_array_equal(out, reference_sheet)


def test_config_overrides_arguments(reference_sheet):
    canvas, corners = _place_on_desk(reference_sheet)
    warped, info = process_perspective_pipeline(canvas, corners, target_size=(10, 10), config=PerspectiveConfig(target_size=(150, 200)))
    assert warped.shape == (200, 150) and info["output_size"] == (150, 200)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PerspectiveConfig(color_order="hsv")
    with pytest.raises(ValueError):
        PerspectiveConfig(search_margin_ratio=0.9)


def test_legacy_facade_exposes_public_api():
    for name in ["order_corners", "solve_homography_svd", "warp_perspective_bilinear",
                 "warp_perspective_sheet", "match_template_ncc",
                 "detect_markers_and_correct_rotation", "process_perspective_pipeline"]:
        assert callable(getattr(tv2_perspective, name))


def test_core_package_does_not_import_opencv():
    code = "import sys, tv2; tv2.warp_perspective_sheet; print('cv2' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=TV2_DIR, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
