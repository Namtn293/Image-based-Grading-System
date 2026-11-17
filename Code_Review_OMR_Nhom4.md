# Code Review toàn bộ dự án OMR – Nhóm 4

**Ngày review:** 17/11/2026
**Đối chiếu với:** `Ke_Hoach_De_Tai_OMR_Nhom4.md`
**Phạm vi:** `OMR_System_TV1`, `OMR_System_TV2`, `OMR_System_TV3`, `OMR_System_TV4` (~12.8k dòng Python)

---

## 0. Tóm tắt

| Hạng mục | Kết quả |
| :--- | :--- |
| Test TV1 | 17/18 pass. Test bị fail là `test_pipeline_execution_integrity`, do vượt giới hạn thời gian 1.5 s |
| Test TV2 (`test_tv2.py`) | 5/5 pass. Trên Windows cần đặt `PYTHONIOENCODING=utf-8`, nếu không sẽ lỗi `UnicodeEncodeError` |
| Test TV3 | 69/69 pass |
| Test TV4 | 20/20 pass |
| Độ khớp với OpenCV | Gaussian, Canny, Otsu, Adaptive (Integral Image), Morphology, CCL, Histogram Equalization và Median đều cài đặt đúng công thức và khớp OpenCV |
| Luật phân loại τ/Δτ (mục 5.3) | Cài đặt đúng |

**Vấn đề chính:** pipeline end-to-end chưa đúng với kế hoạch ở ba điểm:
- Chế độ Scratch vẫn gọi OpenCV ở các bước xử lý ảnh.
- Hough và NCC không nằm trong luồng chạy thật.
- Chế độ Scratch và chế độ OpenCV chạy hai thuật toán khác nhau, nên các thí nghiệm 6.2 và 6.3 chưa có giá trị đối chứng.

---

## 1. 🔴 Nghiêm trọng: lệch kế hoạch hoặc cho kết quả sai

### 1.1. Chế độ "Scratch" vẫn gọi OpenCV để xử lý ảnh

Kế hoạch mục 1.2 ghi rõ: *"tuyệt đối không gọi các hàm xử lý có sẵn của OpenCV trong luồng chính"*. Hiện tại code vi phạm ở các chỗ sau:

- `OMR_System_TV4/src/pipeline_coordinator.py:163-166`: hàm dò lưới ô tròn `_resolve_bubble_grid` dùng `cv2.adaptiveThreshold` và `cv2.findContours`. Hàm này được gọi ở cả hai chế độ (dòng 482).
- `pipeline_coordinator.py:293-296`: `detect_sheet_questions` cũng dùng `cv2.adaptiveThreshold` và `cv2.findContours` (gọi ở dòng 472).
- `pipeline_coordinator.py:430`: dùng `cv2.resize` để đưa ảnh về chiều cao 1000 px.

**Đề xuất:** viết lại bước dò lưới bằng `adaptive_mean_threshold_scratch` và `two_pass_ccl_scratch` + `filter_bubble_components` của TV3. Phép resize thì dùng hàm nội suy bilinear của TV2.

### 1.2. Hough và NCC không được dùng trong pipeline thật

Theo kế hoạch mục 4, giai đoạn 1 bước 4 là Hough, giai đoạn 2 bước 4 là NCC + xoay chuẩn hướng. Thực tế:

- `OMR_System_TV1/src/tv1_preprocessing.py:144-149`: hàm tìm 4 góc phiếu `detect_sheet_corners` chỉ phân ngưỡng cố định `gray >= 100`, sau đó lấy bao lồi và xấp xỉ RDP. Ảnh biên Canny tính ở `process_preprocessing_pipeline` không được dùng tới. Hàm `hough_lines_standard` chỉ được gọi trong benchmark.
- Hàm `detect_markers_and_correct_rotation` (TV2) chỉ được gọi trong `test_tv2.py`.

**Đã kiểm chứng** (ảnh `01_B21DCCN001_Chuan.png`, chế độ Scratch):

| Ảnh đầu vào | Điểm | Số câu đúng | Số câu bỏ trống |
| :--- | :---: | :---: | :---: |
| Gốc 0° | 4.5 | 9 | 1 |
| Xoay 180° | 2.5 | 5 | 5 |
| Xoay 90° | 0.0 | 0 | 16 |

Như vậy kịch bản DS-02 *"phiếu bị xoay ngang 90° hoặc ngược 180°"* chưa được xử lý.

**Đề xuất:** đưa Canny + Hough vào `detect_sheet_corners` (tìm 2 cạnh ngang, 2 cạnh dọc rồi tính giao điểm bằng `intersect_lines_polar`). Sau bước warp thì gọi NCC để sửa hướng phiếu. Lưu ý: muốn dò được góc xoay thì phiếu phải có 1 marker khác biệt với 3 marker còn lại. Thiết kế hiện tại trong kế hoạch là 4 marker giống nhau.

### 1.3. Hàm `extract_bubble_answers` lần nào chạy cũng lỗi

`OMR_System_TV3/src/tv3_segmentation.py:632-634` gọi các thuộc tính không tồn tại:

| Code đang viết | Tên đúng |
| :--- | :--- |
| `res.omr_result.question_results` | `res.omr_result.questions` |
| `q_res.bubble_fill_ratios` | `q_res.fill_ratios` |
| `q_res.selected_answer` | `q_res.selected` |

Đã chạy thử và xác nhận ra lỗi `AttributeError: 'OMRRecognitionResult' object has no attribute 'question_results'`.

**Hệ quả:**
- `OMR_System_TV4/src/tv4_enhancement_eval.py:770-777`: hàm khảo sát tham số `run_ablation_study` bắt `AttributeError` rồi lặng lẽ chuyển sang `cv2.adaptiveThreshold`. Vì vậy con số "Adaptive" hiển thị ở Tab 3 thực chất là kết quả của OpenCV.
- Nhánh Otsu dùng lưới mẫu cố định, còn nhánh Adaptive (nếu chạy được) dùng lưới của TV3. Hai nhánh không cùng lưới nên phép so sánh bị nhiễu.

### 1.4. Chế độ Scratch và chế độ OpenCV không chạy cùng một thuật toán

| Tham số | Scratch (TV3, `pipeline_coordinator.py:488-496`) | OpenCV (`_extract_bubble_answers_cv`) |
| :--- | :--- | :--- |
| Adaptive block size, C | 25, 10 | 51, 42 |
| Morphology | Mở + Đóng | Chỉ Mở |
| Bán kính vùng đo fill | R đầy đủ, tính cả viền in sẵn của ô | R − 3 |
| Luật Δτ cho vết tẩy xóa | Có | Không (cứ ≥ 2 ô vượt ngưỡng là MULTIPLE) |

Đã thấy khác biệt thật: ảnh `03_B21DCCN003_ThieuSang_NhieuHat.png` câu 20 cho kết quả Scratch là `MULTIPLE`, OpenCV là `BLANK`.

Vì vậy thí nghiệm 6.3 (End-to-End Scratch vs OpenCV) chưa so sánh được: chênh lệch đo được là do khác tham số, không phải do khác cách cài đặt. Ngoài ra, đo trên toàn bán kính R thì viền in sẵn của ô trống cũng làm τ tăng lên.

**Đề xuất:** gom toàn bộ tham số vào một config dùng chung (block, C, kernel, bán kính trong, τ_min, Δτ) cho cả hai chế độ.

### 1.5. Kết quả benchmark luôn ra PASS

`OMR_System_TV1/src/tv1_preprocessing.py:435` và `:458`:

```python
"status": "PASS" if abs(len(lines_sc) - lines_cv_count) <= 5 else "PASS"
"status": "PASS" if corner_err <= 3.0 else "PASS"
```

Nhánh `else` phải là `"FAIL"`.

### 1.6. Lỗi bị nuốt im lặng nhưng nhãn chế độ vẫn ghi "Scratch"

`pipeline_coordinator.py:510`: khối `except Exception:` chuyển sang `_extract_bubble_answers_cv` (OpenCV), trong khi kết quả trả về vẫn ghi `"mode": "Scratch (NumPy)"`. Trên 5 ảnh mẫu thì nhánh TV3 chạy được thật, nhưng hễ TV3 có bug là sẽ bị che mất. Hai cách xử lý:
- Ghi log lỗi và đổi nhãn chế độ khi phải chuyển sang OpenCV, hoặc
- Không bắt lỗi ở chế độ Scratch, để lỗi hiện ra.

---

## 2. 🟠 Mức trung bình

### 2.1. Cảnh báo tẩy xóa bị mất
`pipeline_coordinator.py:502-503` đổi `ERASED_VALID` thành một đáp án bình thường, làm mất `warning`. Kế hoạch mục 9 (Tab 4) cần *"danh sách các phiếu có cảnh báo (tô mờ/tẩy xóa) để giám thị xác nhận lại"*, nhưng phần giám thị xác nhận lại ở Tab 1 (`app.py:1010`) hiện chỉ hiện các câu BLANK/MULTIPLE.

### 2.2. Nhánh RDP không bao giờ chạy được
`OMR_System_TV1/src/tv1_hough.py:315` đưa vào `ramer_douglas_peucker` một đa giác khép kín (điểm đầu trùng điểm cuối). Khi đó khoảng cách tới "đoạn thẳng" suy biến luôn bằng 0 (dòng 252), nên hàm luôn trả về 2 điểm. Đã thử với hình vuông khép kín: kết quả là 2 điểm.

Vì vậy code luôn rơi xuống nhánh dự phòng lấy 4 điểm cực trị theo tổng/hiệu tọa độ. Nhánh này vẫn đúng ở góc nghiêng 20°, nhưng thuật toán thực tế đang chạy khác với tài liệu mô tả.

**Đề xuất:** chạy RDP trên đa giác hở, tách thành hai nửa tại điểm xa nhất. Ngoài ra, phần lấy mẫu 3000 điểm theo thứ tự raster (`np.linspace`, dòng 301-303) có thể bỏ sót đúng điểm góc. Nên chỉ lấy x-min/x-max của mỗi hàng (2H điểm).

### 2.3. Kiểm tra "nền bàn tối" quá mong manh
`tv1_preprocessing.py:126-132` quyết định có phải nền bàn tối hay không chỉ dựa vào **4 pixel** ở 4 góc ảnh. Một vệt bóng hay một hạt nhiễu ở góc là chọn sai nhánh. Ngưỡng `100` cũng đang cố định. Nên lấy trung bình hoặc trung vị của các dải viền ảnh, và dùng Otsu thay vì ngưỡng cố định.

### 2.4. NCC tốn rất nhiều bộ nhớ
`OMR_System_TV2/tv2_perspective.py:273-284` tạo mảng `win_zero` kích thước `(H−m+1)×(W−n+1)×m×n` kiểu float64. Với vùng góc 250×200 và template 32×32, mảng này chiếm khoảng **300 MB mỗi góc**.

**Đề xuất:** vì Σ(T − T̄) = 0 nên tử số = Σ I·(T − T̄), có thể tính bằng `einsum` mà không cần trừ trung bình từng cửa sổ. Tổng và tổng bình phương của từng cửa sổ thì tính bằng ảnh tích phân.

### 2.5. Code bị nhân bản
`OMR_System_TV4/src/tv2_perspective.py` là bản sao y hệt `OMR_System_TV2/tv2_perspective.py`. Sửa một bên sẽ lệch bên kia.

### 2.6. Trùng tên package `src` và chèn `sys.path` thủ công
TV1, TV3 và TV4 đều có package tên `src` và đều tự chèn `sys.path` (ví dụ `tv3_segmentation.py:35-40`). Khi nạp cả ba trong cùng một tiến trình, `src` trỏ nhầm sang thư mục của TV3 và báo `ModuleNotFoundError: No module named 'src.pipeline_coordinator'` (đã gặp khi chạy thử). Code hiện chạy được là nhờ thứ tự import tình cờ.

**Đề xuất:** đổi tên package thành `tv1`, `tv2`, `tv3`, `tv4` (hoặc gộp chung thành `omr/`) và cài bằng `pip install -e .`.

### 2.7. Chạy chậm hơn kế hoạch
Đo được 1.1–1.7 s mỗi phiếu ở chế độ Scratch, trong khi bảng 7.2 ghi 0.46 s. Test TV1 cũng fail vì vượt giới hạn 1.5 s.

### 2.8. Ghi đè file đang được git theo dõi
`app.py:1046` và `app.py:2486` ghi đè `data/Ket_Qua_Cham_Thi.xlsx` (file đang nằm trong git) mỗi lần Streamlit chạy lại, kể cả khi đang hiển thị dữ liệu giả lập. Nên ghi workbook vào `io.BytesIO` rồi truyền thẳng cho `st.download_button`.

### 2.9. Dữ liệu và số liệu thực nghiệm chưa đúng kế hoạch
- Repo chỉ có **5 ảnh**, do `generate_samples.py` sinh ra. Kế hoạch mục 2 ghi **60 ảnh chụp thật**, chia 5 tập.
- Đáp án chuẩn (GT) dùng cho khảo sát tham số đang viết cứng trong `app.py:2208-2221`. Nên đưa ra file `.csv` đi kèm từng ảnh.
- Các con số ở bảng 7.2 hiện chưa có script đo tương ứng.

---

## 3. 🟡 Nhỏ

- **Lệch 1 pixel:** `tv2_perspective.py:406-411` (`process_perspective_pipeline`) map góc về `[dst_w, dst_h]`, trong khi `warp_perspective_sheet` dùng `w−1, h−1`.
- **Docstring sai:** `tv2_perspective.py:311` ghi BR → `rot90 k=3`, nhưng code dùng `k=1`. Code đúng, docstring sai.
- **Tài liệu và code lệch nhau:** kế hoạch mục 5.1 ghi `K_y = [[1,2,1],[0,0,0],[-1,-2,-1]]`, còn code dùng dấu ngược lại. Code đúng với quy ước trục y hướng xuống, nên sửa tài liệu. Miền θ của Hough cũng khác: tài liệu ghi [−π/2, π/2), code dùng [0, π).
- **Hiệu năng CCL:** `OMR_System_TV3/src/tv3_morphology_ccl.py:419-421` đánh lại nhãn với độ phức tạp O(K·L). Trên ảnh nhiễu 15% mất 3.5 s. Thay bằng:
  ```python
  root_to_new = np.zeros(num_raw_labels, np.int32)
  root_to_new[ordered_roots] = np.arange(1, num_components + 1)
  final_lut = root_to_new[lut]
  ```
- **Hiệu năng fill ratio:** `tv3_roi_classifier.py:201` gọi `np.max(binary_image)` trên toàn ảnh cho mỗi ô. Chỉ cần tính một lần.
- **Dò lưới tự động:** `auto_align_grid_from_centroids` giả định phiếu chỉ có 1 cột. Phiếu 2 cột sẽ bị gộp 8 ô vào cùng một hàng, và nếu thiếu ô A thì toàn bộ nhãn A/B/C/D bị lệch. Hàm này hiện không dùng mặc định.
- **Phụ thuộc OpenCV không cần thiết:** `tv3_thresholding.py` và `tv3_morphology_ccl.py` import `cv2` ngay đầu file, nên module thuật toán chính cũng phải có OpenCV mới chạy. Nên chuyển import vào trong các hàm benchmark.
- **Benchmark TV2 có thể bị bỏ qua:** `tv4_enhancement_eval.py:446` chỉ thử `from .tv2_perspective`. Khi module được nạp ở dạng top-level, `has_tv2 = False` và benchmark TV2 bị bỏ qua mà không báo gì.
- **Giá trị PSNR khi hai ảnh trùng khớp không thống nhất:** TV1 trả 999, TV4 trả 100.
- **Median filter chậm với kernel lớn:** sắp xếp chèn cần O(k⁴) lượt xử lý trên toàn ảnh. Với k = 7 là khoảng 1176 lượt, rất chậm.
- **Dọn repo:**
  - Các file `.pyc` trong `__pycache__` của TV1 và TV2 vẫn nằm trong git, dù đã có trong `.gitignore`. Dùng `git rm --cached` để gỡ.
  - File kế hoạch bị copy trùng ở `OMR_System_TV3/Ke_Hoach_De_Tai_OMR_Nhom4.md`.
  - Thư mục `.idea/` đang được git theo dõi.

---

## 4. Thứ tự ưu tiên sửa

1. **Mục 1.3:** đổi tên thuộc tính trong `extract_bubble_answers`. Chỉ sửa vài dòng.
2. **Mục 1.5:** sửa nhánh `else` thành `"FAIL"`.
3. **Mục 1.4:** dùng chung một bộ tham số cho chế độ Scratch và OpenCV.
4. **Mục 1.1:** viết lại `_resolve_bubble_grid` và `detect_sheet_questions` bằng CCL của TV3.
5. **Mục 1.6 và 2.1:** báo lỗi rõ khi chuyển sang OpenCV; giữ lại cảnh báo `ERASED_VALID` để Tab 4 dùng.
6. **Mục 1.2:** đưa Hough vào bước tìm góc và NCC vào bước sửa hướng phiếu.
7. **Mục 2.6 và 2.5:** cấu trúc lại package và bỏ file trùng lặp.
8. **Mục 2.9:** bổ sung dataset thật và script đo số liệu cho bảng 7.2.
