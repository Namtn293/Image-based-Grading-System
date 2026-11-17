"""Generate synthetic sample sheets (5 realistic scenarios: DS-01 to DS-05) and CSV matrices for OMR testing."""
import os
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)


def create_base_omr_sheet(student_id: str, student_fill: dict, erased_marks: dict = None) -> np.ndarray:
    """Create an 800x1000 OMR sheet with given student answers and optional faint erasures."""
    sheet = np.full((1000, 800, 3), 245, dtype=np.uint8)

    # Outer fiducial border (placed below top banner area and above bottom legend area)
    cv2.rectangle(sheet, (36, 62), (764, 942), (20, 20, 20), 3)

    # 4 corner fiducial square markers
    cv2.rectangle(sheet, (44, 70), (74, 100), (0, 0, 0), -1)
    cv2.rectangle(sheet, (726, 70), (756, 100), (0, 0, 0), -1)
    cv2.rectangle(sheet, (726, 904), (756, 934), (0, 0, 0), -1)
    cv2.rectangle(sheet, (44, 904), (74, 934), (0, 0, 0), -1)

    # Header with crisp thickness=2 and LINE_AA
    cv2.putText(
        sheet, "PHIEU TRA LOI TRAC NGHIEM TIEU CHUAN",
        (115, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 10, 10), 2, cv2.LINE_AA
    )
    cv2.putText(
        sheet, f"Thi sinh: {student_id}   |   Ma de: 101",
        (115, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (30, 30, 30), 2, cv2.LINE_AA
    )

    options = ["A", "B", "C", "D"]
    start_y = int(1000 * 0.16)
    end_y = int(1000 * 0.89)
    col1_x = int(800 * 0.13)
    col2_x = int(800 * 0.57)
    row_height = (end_y - start_y) / 10.0
    radius = int(min(800, 1000) * 0.018)

    for q in range(1, 21):
        if q <= 10:
            row_idx = q - 1
            base_x = col1_x
        else:
            row_idx = q - 11
            base_x = col2_x

        center_y = int(start_y + (row_idx + 0.5) * row_height)
        cv2.putText(
            sheet, f"Q{q:02d}:",
            (base_x - 52, center_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 2, cv2.LINE_AA
        )

        ans_val = student_fill.get(q)
        if isinstance(ans_val, str):
            marked_opts = [ans_val]
        elif isinstance(ans_val, (list, tuple, set)):
            marked_opts = list(ans_val)
        else:
            marked_opts = []

        for opt_idx, opt in enumerate(options):
            center_x = int(base_x + (opt_idx + 1) * (radius * 2.8))

            # 1. Optional pencil erasure mark (drawn BEFORE printed letter/circle so printed ink stays visible!)
            if erased_marks and erased_marks.get(q) == opt:
                # Light gray eraser smudge across the bubble interior
                cv2.circle(sheet, (center_x, center_y), radius - 2, (205, 205, 205), -1)
                # Realistic un-erased dark pencil streak residue (~40-42% of inner bubble area, triggering Delta_tau rule)
                cv2.ellipse(
                    sheet,
                    (center_x - 2, center_y - 1),
                    (radius - 5, max(radius - 7, 5)),
                    25, 0, 360,
                    (50, 50, 50),
                    -1
                )

            # 2. Printed bubble outline and option letter (light gray ink like real OMR forms)
            cv2.circle(sheet, (center_x, center_y), radius, (55, 55, 55), 2, cv2.LINE_AA)
            cv2.putText(
                sheet, opt,
                (center_x - 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA
            )

            # 3. Student's dark pencil filled mark
            if opt in marked_opts:
                cv2.circle(sheet, (center_x, center_y), radius - 2, (25, 25, 25), -1)

    return sheet


def save_groundtruth_csv(filename: str, fill_dict: dict):
    """Save ground-truth answer key CSV for a sample sheet scenario."""
    lines = ["question_id,ground_truth_answer"]
    for q in sorted(fill_dict.keys()):
        val = fill_dict[q]
        if val is None:
            ans_str = "BLANK"
        elif isinstance(val, list):
            ans_str = "MULTIPLE" if len(val) > 1 else val[0]
        else:
            ans_str = str(val)
        lines.append(f"{q},{ans_str}")
    path = os.path.join(SAMPLES_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated {path}")


def main():
    # 1. Generate 60x60 sample_matrix.csv (Low contrast + Salt & Pepper noise)
    np.random.seed(42)
    mat60 = np.random.randint(50, 100, size=(60, 60), dtype=np.uint8)
    mat60[10, 10] = 255
    mat60[20, 20] = 0
    mat60[30, 30] = 255
    mat60[40, 40] = 0
    csv60_path = os.path.join(DATA_DIR, "sample_matrix.csv")
    np.savetxt(csv60_path, mat60, fmt="%d", delimiter=",")
    print(f"Generated {csv60_path} ({mat60.shape})")

    # 2. Generate 15x15 compact matrix for easy table inspection (Noisy + Low-Contrast)
    np.random.seed(7)
    mat15_low = np.random.randint(55, 95, size=(15, 15), dtype=np.uint8)
    csv15_low_path = os.path.join(DATA_DIR, "sample_matrix_low_contrast.csv")
    np.savetxt(csv15_low_path, mat15_low, fmt="%d", delimiter=",")
    print(f"Generated {csv15_low_path} ({mat15_low.shape})")

    mat15 = mat15_low.copy()
    mat15[2, 3] = 255   # Salt noise
    mat15[5, 7] = 0     # Pepper noise
    mat15[9, 11] = 255  # Salt noise
    mat15[12, 4] = 0    # Pepper noise
    csv15_path = os.path.join(DATA_DIR, "sample_matrix_15x15.csv")
    np.savetxt(csv15_path, mat15, fmt="%d", delimiter=",")
    print(f"Generated {csv15_path} ({mat15.shape})")

    # 3. Scenario 1 (DS-01): Standard Baseline Sheet (B21DCCN001 - Score 8.5/10: 17 correct, 2 wrong, 1 blank)
    fill_sv1 = {
        1: 'A', 2: 'C', 3: 'B', 4: 'D', 5: 'A',
        6: 'B', 7: 'C', 8: 'A',   # Q8 wrong (key is D)
        9: 'A', 10: 'C',
        11: 'B', 12: 'D', 13: 'A', 14: 'C', 15: 'B',
        16: 'D', 17: 'A', 18: 'B', 19: None,  # Q19 blank
        20: 'B'  # Q20 wrong (key is D)
    }
    sheet1 = create_base_omr_sheet("B21DCCN001", fill_sv1)
    cv2.imwrite(os.path.join(DATA_DIR, "sample_sheet.png"), sheet1)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "01_B21DCCN001_Chuan.png"), sheet1)
    save_groundtruth_csv("01_B21DCCN001_Chuan_gt.csv", fill_sv1)

    # 4. Scenario 2 (DS-02): Sheet on Dark Desk with Perspective Tilt (B21DCCN002 - Score 9.0/10: 18 correct, 2 wrong)
    fill_sv2 = {
        1: 'A', 2: 'C', 3: 'B', 4: 'D', 5: 'A',
        6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'C',
        11: 'B', 12: 'A',  # Q12 wrong (key D)
        13: 'A', 14: 'C', 15: 'B', 16: 'D', 17: 'A', 18: 'B',
        19: 'C', 20: 'A'   # Q20 wrong (key D)
    }
    sheet2_flat = create_base_omr_sheet("B21DCCN002", fill_sv2)
    desk = np.full((1120, 920, 3), 45, dtype=np.uint8)
    src_quad = np.array([[0, 0], [799, 0], [799, 999], [0, 999]], dtype=np.float32)
    dst_quad = np.array([[75, 65], [840, 50], [855, 1055], [55, 1045]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    sheet2_tilted = cv2.warpPerspective(
        sheet2_flat, M, (920, 1120), dst=desk, borderMode=cv2.BORDER_TRANSPARENT
    )
    cv2.imwrite(os.path.join(SAMPLES_DIR, "02_B21DCCN002_Nghieng_BanToi.png"), sheet2_tilted)
    save_groundtruth_csv("02_B21DCCN002_Nghieng_BanToi_gt.csv", fill_sv2)

    # 5. Scenario 3 (DS-04): Low-light + Salt-and-Pepper Noise (B21DCCN003 - Score 8.0/10: 16 correct, 3 wrong, 1 blank)
    fill_sv3 = {
        1: 'A', 2: 'C', 3: 'B', 4: 'A',  # Q4 wrong
        5: 'A', 6: 'B', 7: 'C', 8: 'D', 9: 'B',  # Q9 wrong
        10: 'C', 11: 'B', 12: 'D', 13: 'A', 14: 'C',
        15: 'C',  # Q15 wrong
        16: 'D', 17: 'A', 18: 'B', 19: 'C', 20: None  # Q20 blank
    }
    sheet3_clean = create_base_omr_sheet("B21DCCN003", fill_sv3)
    # Simulate dim uniform room lighting [20..142]
    np.random.seed(99)
    sheet3_dark = np.clip(sheet3_clean.astype(np.float64) * 0.52 + 15.0, 0, 255).astype(np.uint8)
    # Add salt and pepper impulse noise
    rnd = np.random.rand(1000, 800)
    sheet3_dark[rnd < 0.012] = 255
    sheet3_dark[(rnd >= 0.012) & (rnd < 0.024)] = 0
    cv2.imwrite(os.path.join(SAMPLES_DIR, "03_B21DCCN003_ThieuSang_NhieuHat.png"), sheet3_dark)
    save_groundtruth_csv("03_B21DCCN003_ThieuSang_NhieuHat_gt.csv", fill_sv3)

    # 6. Scenario 4 (DS-05): Multiple Marked & Erasure Sheet (B21DCCN004 - Score 7.5/10: 15 correct, 1 wrong, 2 blank, 2 multiple)
    fill_sv4 = {
        1: 'A', 2: 'C', 3: ['B', 'D'],  # Q3 MULTIPLE!
        4: 'D', 5: 'A', 6: 'B', 7: 'C', 8: 'D',
        9: 'A', 10: None,  # Q10 BLANK!
        11: 'B', 12: ['A', 'C'],  # Q12 MULTIPLE!
        13: 'A', 14: 'C', 15: 'B', 16: 'D',
        17: 'A', 18: 'D',  # Q18 wrong (key B)
        19: 'C', 20: None  # Q20 BLANK!
    }
    erased_sv4 = {5: 'C', 14: 'A'}  # Erased C on Q5 and marked A; erased A on Q14 and marked C
    sheet4 = create_base_omr_sheet("B21DCCN004", fill_sv4, erased_marks=erased_sv4)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "04_B21DCCN004_ToSaiQuyChe_TayXoa.png"), sheet4)
    save_groundtruth_csv("04_B21DCCN004_ToSaiQuyChe_TayXoa_gt.csv", fill_sv4)

    # 7. Scenario 5 (DS-03): Uneven Lighting & Diagonal Shadow (B21DCCN005 - Score 9.5/10: 19 correct, 1 wrong)
    fill_sv5 = {
        1: 'A', 2: 'C', 3: 'B', 4: 'D', 5: 'A',
        6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'C',
        11: 'B', 12: 'D', 13: 'A', 14: 'C', 15: 'B',
        16: 'D', 17: 'A', 18: 'B', 19: 'C', 20: 'C'  # Q20 wrong (key D)
    }
    sheet5_clean = create_base_omr_sheet("B21DCCN005", fill_sv5)
    yy, xx = np.mgrid[0:1000, 0:800]
    # Cast hand/phone shadow across bottom/right side (bright paper=1.0 [245], shadow=0.33 [~80])
    shadow_score = 0.55 * (yy / 1000.0) + 0.45 * (xx / 800.0)
    illumination = np.where(shadow_score > 0.48, 0.33, 1.0)
    # Soften shadow penumbra edge slightly using Gaussian blur for realism
    illumination = cv2.GaussianBlur(illumination.astype(np.float32), (35, 35), 10.0)
    sheet5_shadow = np.clip(sheet5_clean.astype(np.float64) * illumination[:, :, np.newaxis], 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "05_B21DCCN005_BongDo_ChieuSangKhongDeu.png"), sheet5_shadow)
    save_groundtruth_csv("05_B21DCCN005_BongDo_ChieuSangKhongDeu_gt.csv", fill_sv5)

    print("Successfully generated all 5 sample scenarios (DS-01 to DS-05) in data/samples/!")


if __name__ == "__main__":
    main()
