"""Module TV4 - Grader: OMR Grading Logic, Grayscale Matrix CSV I/O & Excel Reporting
Thành viên 4: Nghiệp vụ Chấm điểm trắc nghiệm, I/O Ma trận CSV & Xuất báo cáo Excel

Các thành phần chính:
1. I/O Ma trận mức xám CSV (Numerical Grayscale Matrix Data Pipeline):
   - save_matrix_to_csv: Lưu ma trận 2D số nguyên mức xám [0, 255] sang file CSV.
   - load_matrix_from_csv: Đọc ma trận CSV chịu lỗi cao (chấp nhận dấu phẩy, chấm phẩy, khoảng trắng).
2. OMRGrader (Lớp Quản lý Chấm thi):
   - Quản lý và nạp đáp án chuẩn (Answer Key) từ CSV hoặc Dictionary.
   - Chấm điểm chuẩn hóa theo thang điểm 10.
   - Phân loại trạng thái: CORRECT (Đúng), INCORRECT (Sai), BLANK (Bỏ trống), MULTIPLE (Tô nhiều ô).
   - Tích hợp luật phát hiện vết tẩy xóa Delta-tau.
   - Chấm thi hàng loạt (grade_batch).
   - Xuất báo cáo Excel tự động (.xlsx) chuẩn OpenPyXL gồm 2 Sheet chuyên nghiệp:
     + Bang_Diem_Tong_Hop: Bảng điểm chi tiết từng sinh viên, phối màu trạng thái câu hỏi.
     + Thong_Ke_Cau_Hoi: Thống kê định lượng độ khó và tỷ lệ làm đúng từng câu hỏi.
"""

from __future__ import annotations
import os
import io
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

try:
    from .tv4_metrics import _normalize_mark_value, VALID_OMR_OPTIONS
except ImportError:
    from tv4_metrics import _normalize_mark_value, VALID_OMR_OPTIONS


# ==============================================================================
# 1. MATRIX CSV I/O (LƯU TRỮ VÀ TRUY XUẤT MA TRẬN MỨC XÁM DẠNG CSV)
# ==============================================================================

def save_matrix_to_csv(matrix: np.ndarray, file_path: str) -> None:
    """Save 2D uint8 image matrix to CSV file.

    Format: comma-separated integer values (0-255).
    """
    if matrix.ndim != 2:
        raise ValueError(f"Only 2D matrices can be saved to CSV, got shape {matrix.shape}")
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    np.savetxt(file_path, matrix.astype(int), fmt="%d", delimiter=",")


def load_matrix_from_csv(file_path_or_content: Union[str, io.StringIO]) -> np.ndarray:
    """Robustly load 2D uint8 image matrix from CSV file or StringIO buffer.

    Tolerates:
      - Trailing commas at line ends
      - Semicolons or commas as delimiters
      - Extra whitespaces and empty lines
      - Comment or header lines with non-numeric text
    """
    lines = []
    if isinstance(file_path_or_content, str):
        if not os.path.exists(file_path_or_content):
            raise FileNotFoundError(f"CSV matrix file not found: {file_path_or_content}")
        with open(file_path_or_content, "r", encoding="utf-8-sig") as f:
            raw_lines = f.readlines()
    else:
        raw_lines = file_path_or_content.getvalue().splitlines()

    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Replace semicolons with commas and split
        parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
        if not parts:
            continue
        try:
            row_vals = [float(p) for p in parts]
            lines.append(row_vals)
        except ValueError:
            # Skip non-numeric title/header row if present
            continue

    if not lines:
        raise ValueError("No valid numeric matrix rows found in CSV")

    # Ensure all rows have equal column count
    num_cols = len(lines[0])
    for idx, row in enumerate(lines):
        if len(row) != num_cols:
            raise ValueError(f"Inconsistent column count at row {idx+1}: expected {num_cols}, got {len(row)}")

    data = np.array(lines, dtype=np.float64)
    clipped = np.clip(np.round(data), 0, 255).astype(np.uint8)
    return clipped


# ==============================================================================
# 2. OMR GRADING ENGINE (LỚP ĐIỀU HÀNH CHẤM THI VÀ XUẤT EXCEL)
# ==============================================================================

class OMRGrader:
    """Manages Answer Key, grading logic, and report generation."""

    VALID_OPTIONS = {"A", "B", "C", "D"}

    def __init__(self, answer_key_path: Optional[str] = None):
        self.answer_key: Dict[int, str] = {}
        if answer_key_path and os.path.exists(answer_key_path):
            self.load_answer_key(answer_key_path)

    def load_answer_key(self, csv_path: Union[str, os.PathLike, io.StringIO]) -> Dict[int, str]:
        """Robustly load answer key from CSV file.

        Handles:
          - Semicolon or Comma delimiters
          - UTF-8 with BOM from Windows Excel
          - Headers with extra spaces
        """
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # Strip whitespace from column names
        df.columns = [str(c).strip().lower() for c in df.columns]
        if len(df.columns) < 2:
            raise ValueError(f"Answer key CSV must contain at least 2 columns: question_id and answer, found {df.columns}")

        q_col = df.columns[0]
        ans_col = df.columns[1]

        key_dict = {}
        for _, row in df.iterrows():
            try:
                q_num = int(row[q_col])
                raw_opt = row[ans_col]
                if raw_opt is None or (isinstance(raw_opt, float) and np.isnan(raw_opt)):
                    continue
                ans = str(raw_opt).strip().upper()
                if ans in self.VALID_OPTIONS:
                    key_dict[q_num] = ans
            except (ValueError, TypeError):
                continue

        if not key_dict:
            raise ValueError("No valid question-answer pairs found in answer key CSV")

        self.answer_key = key_dict
        return self.answer_key

    def set_answer_key_dict(self, key_dict: Dict[int, str]) -> None:
        """Directly set answer key from dictionary."""
        self.answer_key = {int(k): str(v).strip().upper() for k, v in key_dict.items()}

    def grade_student(
        self,
        student_id: str,
        student_answers: Dict[Any, Any],
        erased_questions: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """Grade single student submission against answer key."""
        if not self.answer_key:
            raise ValueError("Answer key has not been loaded yet!")

        total_questions = len(self.answer_key)
        correct_count = 0
        incorrect_count = 0
        blank_count = 0
        multiple_count = 0

        question_details = []

        for q_num in sorted(self.answer_key.keys()):
            correct_ans = self.answer_key[q_num]
            if q_num in student_answers:
                raw_ans = student_answers[q_num]
            else:
                raw_ans = student_answers.get(str(q_num), "BLANK")

            student_ans = _normalize_mark_value(raw_ans)

            if student_ans == "BLANK":
                status = "BLANK"
                blank_count += 1
                is_correct = False
            elif student_ans == "MULTIPLE":
                status = "MULTIPLE"
                multiple_count += 1
                is_correct = False
            elif student_ans == correct_ans:
                status = "CORRECT"
                correct_count += 1
                is_correct = True
            else:
                status = "INCORRECT"
                incorrect_count += 1
                is_correct = False

            is_erased_q = bool(erased_questions and q_num in erased_questions)
            item_detail = {
                "question": q_num,
                "student_ans": student_ans,
                "correct_ans": correct_ans,
                "status": status,
                "is_correct": is_correct,
                "is_erased": is_erased_q
            }
            if is_erased_q:
                item_detail["warning"] = "Vết tẩy xóa nghi vấn (đã giải mã theo luật Δτ)"

            question_details.append(item_detail)

        # Score on 10-point scale
        score_10 = round((correct_count / float(total_questions)) * 10.0, 2)

        return {
            "student_id": student_id,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "blank_count": blank_count,
            "multiple_count": multiple_count,
            "score_10": score_10,
            "details": question_details,
            "erased_questions": list(erased_questions.keys()) if erased_questions else []
        }

    def grade_batch(
        self,
        batch_submissions: Dict[str, Dict[int, Any]]
    ) -> List[Dict[str, Any]]:
        """Grade a batch of submissions."""
        results = []
        for sid, answers in batch_submissions.items():
            res = self.grade_student(sid, answers)
            results.append(res)
        return results

    @staticmethod
    def generate_excel_report(
        batch_results: List[Dict[str, Any]],
        output_excel_path: Union[str, io.BytesIO, os.PathLike]
    ) -> Union[str, io.BytesIO, os.PathLike]:
        """Export grading report to Excel (.xlsx) with professional OpenPyXL styling.

        Supports both file path strings and in-memory io.BytesIO targets.
        Includes 2 styled sheets:
          1. Bang_Diem_Tong_Hop (Summary score table with colors).
          2. Thong_Ke_Cau_Hoi (Per-question statistics and difficulty analysis).
        """
        if isinstance(output_excel_path, (str, bytes, os.PathLike)):
            out_str = str(output_excel_path)
            parent_dir = os.path.dirname(os.path.abspath(out_str))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        wb = openpyxl.Workbook()

        # Styles
        font_header = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        fill_header = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        fill_sub_header = PatternFill(start_color="2E86C1", end_color="2E86C1", fill_type="solid")
        align_center = Alignment(horizontal="center", vertical="center")
        thin_border = Border(
            left=Side(style="thin", color="D3D3D3"),
            right=Side(style="thin", color="D3D3D3"),
            top=Side(style="thin", color="D3D3D3"),
            bottom=Side(style="thin", color="D3D3D3")
        )

        fill_correct = PatternFill(start_color="E8F8F5", end_color="E8F8F5", fill_type="solid")
        fill_wrong = PatternFill(start_color="FDEDEC", end_color="FDEDEC", fill_type="solid")
        fill_warn = PatternFill(start_color="FEF9E7", end_color="FEF9E7", fill_type="solid")

        # -------------------------------------------------------------
        # SHEET 1: BẢNG ĐIỂM TỔNG HỢP
        # -------------------------------------------------------------
        ws1 = wb.active
        ws1.title = "Bang_Diem_Tong_Hop"

        headers1 = [
            "STT", "Mã Sinh Viên", "Điểm Số (Hệ 10)", "Số Câu Đúng",
            "Số Câu Sai", "Số Câu Bỏ Trống", "Số Câu Phạm Quy", "Tổng Số Câu"
        ]
        q_keys = []
        if batch_results:
            q_keys = sorted({d["question"] for r in batch_results for d in r["details"]})
            for qk in q_keys:
                headers1.append(f"Q{qk:02d}")

        ws1.append(headers1)
        for cell in ws1[1]:
            cell.font = font_header
            cell.fill = fill_header
            cell.alignment = align_center

        for stt, r in enumerate(batch_results, start=1):
            row_data = [
                stt,
                r.get("student_id", f"SV{stt:03d}"),
                r.get("score_10", 0.0),
                r.get("correct_count", 0),
                r.get("incorrect_count", 0),
                r.get("blank_count", 0),
                r.get("multiple_count", 0),
                r.get("total_questions", len(r.get("details", [])))
            ]
            det_dict = {d["question"]: d for d in r["details"]}
            for qk in q_keys:
                item = det_dict.get(qk, {"student_ans": "N/A"})
                row_data.append(item["student_ans"])

            ws1.append(row_data)
            row_idx = ws1.max_row
            # Format row cells
            for col_idx, val in enumerate(row_data, start=1):
                c = ws1.cell(row=row_idx, column=col_idx)
                c.border = thin_border
                c.alignment = align_center
                if col_idx > 8:
                    qk = q_keys[col_idx - 9]
                    status = det_dict.get(qk, {}).get("status", "")
                    if status == "CORRECT":
                        c.fill = fill_correct
                    elif status == "INCORRECT":
                        c.fill = fill_wrong
                    elif status in ("BLANK", "MULTIPLE"):
                        c.fill = fill_warn

        if batch_results:
            scores = [r.get("score_10", 0.0) for r in batch_results]
            avg_score = round(float(np.mean(scores)), 2)
            avg_correct = round(float(np.mean([r.get("correct_count", 0) for r in batch_results])), 1)
            avg_incorrect = round(float(np.mean([r.get("incorrect_count", 0) for r in batch_results])), 1)
            avg_blank = round(float(np.mean([r.get("blank_count", 0) for r in batch_results])), 1)
            avg_multiple = round(float(np.mean([r.get("multiple_count", 0) for r in batch_results])), 1)
            tot_q = batch_results[0].get("total_questions", len(q_keys))

            summary_row1 = [
                "TB", f"Điểm Trung Bình ({len(batch_results)} SV)", avg_score,
                avg_correct, avg_incorrect, avg_blank, avg_multiple, tot_q
            ]
            for _ in q_keys:
                summary_row1.append("—")

            ws1.append(summary_row1)
            sum_row_idx = ws1.max_row
            font_summary = Font(name="Segoe UI", size=10, bold=True, color="1E293B")
            fill_summary = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
            for col_idx in range(1, len(summary_row1) + 1):
                c = ws1.cell(row=sum_row_idx, column=col_idx)
                c.font = font_summary
                c.fill = fill_summary
                c.border = thin_border
                c.alignment = align_center

        # Auto-adjust column widths for Sheet 1
        for col in ws1.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws1.column_dimensions[col_letter].width = max(max_len + 3, 10)

        # -------------------------------------------------------------
        # SHEET 2: THỐNG KÊ CÂU HỎI
        # -------------------------------------------------------------
        ws2 = wb.create_sheet(title="Thong_Ke_Cau_Hoi")
        headers2 = [
            "Câu Hỏi", "Đáp Án Đúng", "Số SV Đúng", "Số SV Sai",
            "Số SV Bỏ Trống", "Số SV Phạm Quy", "Tỷ Lệ Đúng (%)", "Tỷ Lệ Sai (%)"
        ]
        ws2.append(headers2)
        for cell in ws2[1]:
            cell.font = font_header
            cell.fill = fill_sub_header
            cell.alignment = align_center

        if batch_results:
            total_students = len(batch_results)
            first_det_map = {
                d["question"]: d["correct_ans"]
                for r in batch_results
                for d in r["details"]
            }
            for qk in q_keys:
                correct = sum(1 for r in batch_results if any(d["question"] == qk and d["is_correct"] for d in r["details"]))
                wrong = sum(1 for r in batch_results if any(d["question"] == qk and d["status"] == "INCORRECT" for d in r["details"]))
                blank = sum(1 for r in batch_results if any(d["question"] == qk and d["status"] == "BLANK" for d in r["details"]))
                mult = sum(1 for r in batch_results if any(d["question"] == qk and d["status"] == "MULTIPLE" for d in r["details"]))

                acc_rate = round((correct / max(total_students, 1)) * 100.0, 1)
                err_rate = round(100.0 - acc_rate, 1)

                key_ans = first_det_map.get(qk, "")

                ws2.append([
                    f"Câu {qk:02d}", key_ans, correct, wrong, blank, mult, acc_rate, err_rate
                ])
                row_idx = ws2.max_row
                for col_idx in range(1, len(headers2) + 1):
                    c = ws2.cell(row=row_idx, column=col_idx)
                    c.border = thin_border
                    c.alignment = align_center

            if q_keys:
                all_cor = [sum(1 for r in batch_results if any(d["question"] == qk and d["is_correct"] for d in r["details"])) for qk in q_keys]
                all_acc = [round((c / max(total_students, 1)) * 100.0, 1) for c in all_cor]
                avg_acc = round(float(np.mean(all_acc)), 1)
                sum_row2 = [
                    "Toàn Đề", "—", round(float(np.mean(all_cor)), 1),
                    round(float(total_students - np.mean(all_cor)), 1),
                    "—", "—", avg_acc, round(100.0 - avg_acc, 1)
                ]
                ws2.append(sum_row2)
                sum_row_idx2 = ws2.max_row
                font_summary = Font(name="Segoe UI", size=10, bold=True, color="1E293B")
                fill_summary = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
                for col_idx in range(1, len(headers2) + 1):
                    c = ws2.cell(row=sum_row_idx2, column=col_idx)
                    c.font = font_summary
                    c.fill = fill_summary
                    c.border = thin_border
                    c.alignment = align_center

        for col in ws2.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws2.column_dimensions[col_letter].width = max(max_len + 3, 12)

        wb.save(output_excel_path)
        return output_excel_path
