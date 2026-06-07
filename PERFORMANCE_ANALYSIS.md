# 📊 Model Performance Analysis & Improvement Guide

## 🔴 **Current Scores (VERY LOW)**

```
BLEU1: 4.1%  | BLEU2: 3.0%  | BLEU3: 2.5%  | BLEU4: 1.7%
ROUGE1: 10%  | ROUGE2: 6%   | ROUGEL: 9%
METEOR: 3%   | BERTScore: 0.44
```

**Assessment:** Model không học được generate Q&A tốt ❌

---

## 🔍 **Nguyên nhân chính**

### 1️⃣ **Dataset quá nhỏ**
- Training set: ~1000 samples (quá nhỏ)
- Model cần ít nhất 5000-10000 samples để học tốt

### 2️⃣ **Data quality issues**
- Dữ liệu gốc có lỗi semantic (câu trả lời không phải substring)
- Sau rebuild_dataset, còn lại quá ít samples

### 3️⃣ **Model architecture**
- ViT5-base có thể không đủ mạnh
- Cần ViT5-large hoặc model lớn hơn

### 4️⃣ **Training issues**
- Epoch = 10 (quá ít, thường cần 30+)
- Learning rate = 0.0001 (có thể cần điều chỉnh)
- Batch size = 4 (quá nhỏ)

---

## 💡 **Cách cải thiện (Priority Order)**

### 🥇 **Priority 1: Tăng dataset size** (CRITICAL)

**Option A:** Sử dụng dữ liệu gốc không rebuild
```bash
# Trong untitled62.py, thay đổi:
processor.process_data(
    input_dir='data/examples_ai_flashcard',  # ← Dùng GỐC thay vì _fixed
    output_dir='data/processed_data',
)
```

**Option B:** Thu thập thêm dữ liệu từ nguồn khác
- Tìm Vietnamese QA datasets từ Hugging Face
- Cộng thêm dữ liệu mới train

---

### 🥈 **Priority 2: Tăng training epochs & resources**

```python
# Trong untitled62.py Step 3:
!python train.py fine-tuning \
  --model VietAI/vit5-base \
  --dataset_path data/processed_data \
  --epoch 50 \              # ← Tăng từ 10 lên 50
  --batch 8 \               # ← Tăng từ 4 lên 8
  --lr 5e-5 \               # ← Giảm từ 1e-4 xuống 5e-5
  --gradient_accumulation_steps 2 \
  --checkpoint_dir ./models/vit5-flashcard-v2
```

---

### 🥉 **Priority 3: Sử dụng model lớn hơn**

```python
# Thay vì VietAI/vit5-base:
--model VietAI/vit5-large  # Model lớn hơn, tốt hơn
# hoặc
--model google/mt5-base     # Multilingual T5
```

---

### 4️⃣ **Priority 4: Data preprocessing tốt hơn**

Kiểm tra/sửa:
- Loại bỏ samples với answer quá ngắn (<3 tokens)
- Loại bỏ context quá ngắn (<50 tokens)
- Chuẩn hóa text (xóa diacritics conflicts, ...)

---

## 🧪 **Test quick fix**

### Bước 1: Sử dụng data gốc
```bash
# Trong untitled62.py, cập nhật processor.process_data():
input_dir='data/examples_ai_flashcard'  # NOT _fixed
```

### Bước 2: Train lại với settings tốt hơn
```bash
# Tăng epoch
--epoch 30
--batch 8
```

### Bước 3: Chạy Gradio app
```bash
python app_gradio.py
```

---

## 📈 **Expected improvements**

| Scenario | BLEU1 | ROUGE1 | BERTScore |
|----------|-------|--------|-----------|
| **Hiện tại** | 4.1% | 10% | 0.44 |
| Sử dụng data gốc | **8-12%** | **15-20%** | **0.55-0.65** |
| Tăng epoch + batch | **12-18%** | **20-30%** | **0.65-0.75** |
| Large model + full data | **25-35%** | **35-45%** | **0.75-0.85** |

---

## ⚠️ **Note**

- Scores này **vẫn thấp** so với SOTA (50%+)
- Nhưng đủ để model generate có ý nghĩa
- Cần dữ liệu chất lượng cao để reach 50%+

---

## 🔧 **Quick Debug Checklist**

- [ ] Check processed_data có bao nhiêu samples?
  ```bash
  wc -l data/processed_data/*.jsonl
  ```

- [ ] Check training data format match model?
  ```bash
  head -1 data/processed_data/train.jsonl | python -m json.tool
  ```

- [ ] Check model weights đã save đúng?
  ```bash
  ls -lh ./models/vit5-flashcard/epoch_*
  ```

- [ ] Check validation/test error rate?
  ```bash
  # Xem evaluation.py output
  ```

---

## 📌 **Recommendation**

1. **Ngay bây giờ:** Dùng data gốc (không rebuild) → sẽ tốt hơn
2. **Tuần này:** Train lại với epoch=30, batch=8
3. **Tháng này:** Collect thêm 5000+ samples từ sources khác
4. **Sau:** Thử mt5-base hoặc models lớn khác

---

**Liên hệ nếu cần giúp implement các improvement này!** 🚀
