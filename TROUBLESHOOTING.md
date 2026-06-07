# 🔧 Troubleshooting - Model Không Sinh Output

## 🎯 Tình huống: Model sinh output trống

```json
{
  "raw_output": "(Output trống từ model)",
  "num_generated": 1,
  "qa_pairs": [{
    "question": "⚠️ Model sinh output trống",
    "answer": "(Xem Raw Output để debug)"
  }]
}
```

---

## 🔍 **Cách Debug Bước Từng Bước**

### **Bước 1: Chạy Test Script**

```bash
cd ViQAG
python test_model.py
```

**Xem output:**
```
[GENERATED] Output tokens count: 0
[WARNING] 0 tokens generated!
```

→ **Model không sinh tokens** → Skip đến "Giải pháp 2"

---

hoặc

```
[DECODED] Output (skip_special=True): 'question: Đại học là gì?, answer: Trường công lập'
✅ Output contains 'question:' - good format!
```

→ **Model sinh được nhưng Gradio không capture** → Skip đến "Giải pháp 1"

---

### **Bước 2: Check Terminal Log khi Chạy Gradio**

```bash
python app_gradio.py
```

Nhập text → **Check terminal output** xem có những dòng nào:

```
[GENERATED] Output tokens count: 256
[GENERATED] All output tokens: [101, 5783, 6245, ...]
[DECODED] Full output: 'question: ...
```

---

## 💡 **Các Giải Pháp**

### **Giải pháp 1: Output sinh được nhưng capture có vấn đề**

Các dấu hiệu:
- Terminal log hiển thị output
- Nhưng Gradio UI thấy "(Output trống)"

**Fix:**
- Chắc chắn `output_storage.set()` được gọi
- Kiểm tra không có exception trong `generate()`
- Thêm print sau mỗi bước

```python
# Trong generate():
print(f"[BEFORE_SET] result_text = {repr(result_text)}")
output_storage.set(result_text if result_text else "(Model sinh trống)")
print(f"[AFTER_SET] storage value = {repr(output_storage.get())}")
```

---

### **Giải pháp 2: Model sinh output trống (0 tokens)**

**Nguyên nhân:**
- Model không được train tốt
- Dataset quá nhỏ
- Learning rate quá cao/thấp
- Model bị overfitting

**Fix:**

#### **A. Kiểm tra dữ liệu training**
```bash
# Check sample
head -1 data/processed_data/train.jsonl | python -m json.tool | head -20

# Count samples
wc -l data/processed_data/*.jsonl
```

**Expected:**
- ✅ 500+ samples mỗi split
- ✅ Format: `instruction`, `paragraph`, `questions_answers`

**Nếu:**
- ❌ < 100 samples → quá nhỏ
- ❌ Format khác → cần retrain

---

#### **B. Thử sử dụng dataset gốc (không rebuild)**

Trong `untitled62.py`:
```python
# Bước 2: Thay từ
input_dir='data/examples_ai_flashcard_fixed'
# Thành
input_dir='data/examples_ai_flashcard'  # ← Dataset gốc, có nhiều samples hơn
```

Chạy lại training + test

---

#### **C. Thử generation parameters khác**

Trong `app_gradio.py`, sửa `generate()`:
```python
outputs = self.model.generate(
    input_ids=inputs['input_ids'],
    max_length=256,
    num_beams=1,          # ← Thay từ 3 xuống 1
    do_sample=False,      # ← Thay từ True xuống False
    temperature=0.5,      # ← Thay từ 0.8 xuống 0.5
    top_p=0.9
)
```

---

#### **D. Retrain model với settings tốt hơn**

```bash
cd ViQAG
!python train.py fine-tuning \
  --model VietAI/vit5-base \
  --dataset_path data/processed_data \
  --epoch 30 \              # ← Tăng từ 10
  --batch 8 \               # ← Tăng từ 4
  --lr 5e-5 \               # ← Giảm từ 1e-4
  --checkpoint_dir ./models/vit5-flashcard-v2
```

---

### **Giải pháp 3: Model sinh output nhưng format sai**

**Dấu hiệu:**
```
[DECODED] Full output: 'question question question'  ❌
```

**Giải pháp:**
- Model không học được format `question: ... answer: ...`
- Check instructions.txt vs actual training data
- Cần retrain hoặc sử dụng model khác

---

## 🧪 **Quick Diagnostic Checklist**

```
Chạy test_model.py:
☐ Model load thành công?
☐ Tokenize thành công?
☐ Generate output? (tokens count > 0?)
☐ Output format đúng?

Chạy Gradio:
☐ Nhập text?
☐ Terminal log hiển thị [GENERATED]?
☐ [DECODED] output hiển thị gì?
☐ Raw Output field hiển thị gì?

Check Data:
☐ Training data format đúng?
☐ Sample count đủ lớn?
☐ Instructions match?
```

---

## 📊 **Expected vs Actual**

### ✅ **Good Output**
```
[GENERATED] Output tokens count: 45
[DECODED] Full output: 'question: Đại học là gì?, answer: Trường công lập'
```
↓
```json
{
  "raw_output": "question: Đại học là gì?, answer: Trường công lập",
  "qa_pairs": [{
    "question": "Đại học là gì?",
    "answer": "Trường công lập"
  }]
}
```

### ❌ **Bad Output #1: Empty**
```
[GENERATED] Output tokens count: 0
[DECODED] Full output: ''
```
→ Model không sinh được → Retrain

### ❌ **Bad Output #2: Wrong Format**
```
[GENERATED] Output tokens count: 50
[DECODED] Full output: 'question question question'
```
→ Model bị overfitting → Retrain với learning rate khác

### ❌ **Bad Output #3: Gibberish**
```
[DECODED] Full output: 'hdfksjdhfsdk askldfjsadf'
```
→ Tokenizer/model mismatch → Check model path

---

## 🚀 **Next Steps**

1. **Chạy `test_model.py`** → Xác định model sinh gì thực tế
2. **Kiểm tra log** → Xem exact output từ model
3. **Chọn giải pháp tương ứng** → Apply fix
4. **Test lại** → Xem kết quả

---

## 📌 **Resources**

- [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) - Analysis scores model
- [DEBUG_GUIDE.md](DEBUG_GUIDE.md) - Chi tiết debug terminal
- [GRADIO_GUIDE.md](GRADIO_GUIDE.md) - Cách sử dụng Gradio

---

**Cần giúp? Chạy test_model.py và chia sẻ output!** 🎯
