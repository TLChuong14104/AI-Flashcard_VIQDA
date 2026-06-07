# 🚀 Hướng dẫn chạy Gradio App

## Setup

### 1. Cài đặt Gradio
```bash
pip install gradio
```

### 2. Chạy app
```bash
cd ViQAG
python app_gradio.py
```

### 3. Truy cập giao diện
- **Local:** http://localhost:7860
- **Public:** Sẽ có link public để chia sẻ

---

## 📝 Cách sử dụng

### Bước 1: Tải Model
1. Nhập đường dẫn model (default: `./models/vit5-flashcard/epoch_10`)
2. Nhấn **"📥 Tải Model"**
3. Chờ cho đến khi thấy ✅ thành công

### Bước 2: Sinh Q&A
1. Nhập đoạn văn bản tiếng Việt (tối thiểu 30 ký tự)
2. Chọn số cặp câu hỏi-câu trả lời muốn sinh (1-5)
3. Nhấn **"🚀 Sinh Q&A"**
4. Chờ kết quả

### Bước 3: Xem kết quả
- **Phần trên:** Kết quả định dạng đẹp (Markdown)
- **JSON tab:** Dữ liệu dạng JSON để debug

---

## ⚙️ Tùy chỉnh

Chỉnh sửa các tham số trong code:

```python
outputs = self.model.generate(
    inputs,
    max_length=256,              # Độ dài tối đa output
    num_beams=4,                 # Beam search width
    num_return_sequences=3,      # Số sequence sinh ra
    temperature=0.7,             # Sáng tạo (cao = tự do)
    top_p=0.9                    # Diversity parameter
)
```

---

## 🐛 Troubleshooting

### Model không load được
```
❌ Không thể load model: ...
```
**Giải pháp:** Kiểm tra đường dẫn model có tồn tại không:
```bash
ls -la ./models/vit5-flashcard/epoch_10
```

### GPU không được dùng
Chỉnh sửa code:
```python
self.device = "cuda"  # Force GPU
# hoặc
self.device = "cpu"   # Force CPU
```

### App chạy chậm
- Giảm `max_length` 
- Giảm `num_beams` xuống 2
- Sử dụng GPU thay CPU

---

## 📊 Output Format

```json
{
  "context": "Đoạn văn bản gốc...",
  "num_generated": 3,
  "qa_pairs": [
    {
      "id": 1,
      "question": "Câu hỏi 1?",
      "answer": "Câu trả lời 1"
    },
    {
      "id": 2,
      "question": "Câu hỏi 2?",
      "answer": "Câu trả lời 2"
    }
  ]
}
```

---

## 💡 Tips

- **Test nhanh:** Dùng phần "Examples" ở dưới
- **Batch processing:** Có thể modify code để xử lý nhiều text cùng lúc
- **Export:** Copy JSON để lưu kết quả

---

**Vui lòng report bugs hoặc feature requests!** 🎉
