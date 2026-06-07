# 🔧 Debug Guide - Model Không Sinh Output

## 🔴 **Triệu chứng**
```
❓ Câu hỏi 1: Không thể parse
✅ Câu trả lời 1: Không có output từ model
```

**Raw Output:** Trống hoặc không có ý nghĩa

---

## 🔍 **Cách Debug**

### 1️⃣ **Kiểm tra Log Terminal**

Khi chạy `python app_gradio.py`, xem terminal output:

```
[INPUT] qag: Trong Machine Learning...
[TOKENIZED] Input IDs shape: torch.Size([1, 512])
[GENERATED] Output shape: torch.Size([1, 256])
[RAW OUTPUT] Length: 45 chars
[RAW OUTPUT] Content: question: ..., answer: ...
```

**Điều gì cần check:**
- ✅ `[INPUT]` - Input được format đúng không?
- ✅ `[TOKENIZED]` - Tokenize thành công không?
- ✅ `[GENERATED]` - Model generate output không?
- ✅ `[RAW OUTPUT]` - Output là gì?

---

### 2️⃣ **Các tình huống có thể**

#### **A. Output quá ngắn/trống**
```
[RAW OUTPUT] Length: 0 chars
[RAW OUTPUT] Content: 
```

**Nguyên nhân:** Model không được train tốt
**Giải pháp:** Xem [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)

---

#### **B. Output là chuỗi ký tự lạ**
```
[RAW OUTPUT] Content: question question question...
```

**Nguyên nhân:** Model bị overfitting hoặc token bị lặp
**Giải pháp:** Sửa generation parameters:

```python
# Trong app_gradio.py, hàm generate():
outputs = self.model.generate(
    ...,
    num_beams=1,              # ← Giảm beam search
    do_sample=False,          # ← Greedy decoding
    length_penalty=2.0,       # ← Phạt output ngắn
)
```

---

#### **C. Output không match format**
```
[RAW OUTPUT] Content: "không phải question: format"
```

**Nguyên nhân:** Model train trên format khác
**Giải pháp:** Cần xem training data format thực tế

```bash
# Kiểm tra format:
head -1 data/processed_data/train.jsonl | python -m json.tool
```

---

### 3️⃣ **Quick Test Script**

Tạo file `test_model.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "./models/vit5-flashcard/epoch_10"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Loading model from {model_path}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    print("✅ Model loaded!")
    
    # Test
    context = "Đại học Bách khoa Hà Nội là trường công lập hàng đầu."
    input_text = f"qag: {context}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    print(f"\nInput shape: {inputs['input_ids'].shape}")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            max_length=256,
            num_beams=3,
            num_return_sequences=1
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n✅ Output ({len(result)} chars):")
    print(result)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
```

Chạy:
```bash
cd ViQAG
python test_model.py
```

---

### 4️⃣ **Check Model Config**

```bash
# Xem config model
cat ./models/vit5-flashcard/epoch_10/config.json | grep -A5 "vocab_size\|max_position"

# Xem model size
ls -lh ./models/vit5-flashcard/epoch_10/
```

---

### 5️⃣ **Check Training Data Format**

```bash
# Xem format dữ liệu training
head -1 data/processed_data/train.jsonl | python -m json.tool | head -20

# Xem instructions
cat data/instructions.txt | head -3
```

---

## 📊 **Expected vs Actual**

### ✅ **Good Output**
```
[RAW OUTPUT] Content: question: Đại học Bách khoa là gì?, answer: Trường công lập hàng đầu [SEP] question: ...
```

### ❌ **Bad Output**
```
[RAW OUTPUT] Content: question question question question
[RAW OUTPUT] Content:  (trống)
[RAW OUTPUT] Content: không đúng format
```

---

## 🚀 **Các bước tiếp theo**

1. **Check terminal log** → Xem model sinh gì thực tế
2. **Modify generation params** nếu output bị lạ
3. **Retrain model** nếu output trống (xem PERFORMANCE_ANALYSIS.md)
4. **Use different model** nếu vẫn không tốt

---

## 💡 **Tips**

- Luôn **check terminal output** khi debug
- **Save raw output** từ test script để so sánh
- Test với **context khác nhau** để xem consistency
- Xem **model logs** khi training để check learning

---

**Cần giúp? Check logs và message tôi! 🎯**
