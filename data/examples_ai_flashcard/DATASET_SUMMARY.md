# Extended ML Flashcard Dataset - Summary

## 📊 Dataset Overview

Đã tạo thành công **69 mẫu dữ liệu mới** (không trùng lặp) dựa trên kiến trúc hiện có của `examples_ai_flashcard`.

### Dataset Statistics

| Tập | Original | Extended | Tổng | Tỷ Lệ |
|-----|----------|----------|------|-------|
| Train | 143 | 48 | 191 | 76% |
| Validation | 28 | 10 | 38 | 15% |
| Test | 32 | 11 | 43 | 17% |
| **Total** | **203** | **69** | **272** | **100%** |

**Tăng trưởng dữ liệu: +34%**

---

## 📁 File Locations

Extended dataset được lưu tại:
```
d:\NCKH_FLASH\ViQAG\data\examples_ai_flashcard\
├── train_extended.jsonl       (48 mẫu)
├── validation_extended.jsonl  (10 mẫu)
├── test_extended.jsonl        (11 mẫu)
├── train.jsonl                (143 mẫu - gốc)
├── validation.jsonl           (28 mẫu - gốc)
└── test.jsonl                 (32 mẫu - gốc)
```

---

## 📚 Nội Dung Dataset Mới

Dataset mới bao gồm các chủ đề Machine Learning nâng cao:

### 1. **Deep Learning & Neural Networks** (12 QA)
- Activation functions: ReLU, Sigmoid, Tanh, ELU, GELU, Swish
- Normalization: Batch, Layer, Group, Instance
- Architecture: ResNet, Inception, MobileNet, EfficientNet, ViT

### 2. **Transformers & NLP** (10 QA)
- Word embeddings: Word2Vec, GloVe, FastText
- Positional encoding, Attention mechanisms
- Seq2Seq, Beam search, Temperature sampling
- Top-K & Top-p sampling

### 3. **Computer Vision** (8 QA)
- YOLO, Faster R-CNN object detection
- Data augmentation techniqueszz
- Depthwise separable convolutions
- Dilated/Atrous convolutions

### 4. **Optimization & Training** (10 QA)
- Optimizers: SGD with momentum, RMSprop, AdamW
- Learning rate scheduling, Warmup, Gradient clipping
- Weight decay vs L2 regularization
- Batch size effects, Gradient accumulation

### 5. **Metrics & Evaluation** (5 QA)
- Precision, Recall, F1-score
- mAP (mean Average Precision)
- IoU (Intersection over Union)
- Focal loss

### 6. **Data Preprocessing** (5 QA)
- Class imbalance handling
- Data leakage prevention
- Missing values imputation
- Outlier detection & handling
- Feature scaling

### 7. **Unsupervised Learning** (6 QA)
- K-means, DBSCAN clustering
- Dimensionality reduction: PCA, t-SNE
- Autoencoders

### 8. **Reinforcement Learning** (4 QA)
- Q-learning, Policy gradient
- Actor-Critic, Deep Q-Networks

### 9. **Advanced Topics** (9 QA)
- Meta-learning, Few-shot learning
- Federated learning, Zero-shot learning
- Neural Architecture Search
- Knowledge distillation, Pruning, Quantization

---

## 🎯 Format Dữ Liệu

Mỗi mẫu là JSON line format với cấu trúc:
```json
{
  "question": "Câu hỏi về chủ đề ML",
  "context": "Ngữ cảnh & giải thích chi tiết",
  "answer": "Câu trả lời ngắn gọn",
  "difficulty": "basic|intermediate|advanced"
}
```

### Phân bố Difficulty
- **Basic**: 25 mẫu (36%)
- **Intermediate**: 30 mẫu (44%)
- **Advanced**: 14 mẫu (20%)

---

## ✨ Đặc Điểm Dataset

1. **Không trùng lặp**: Toàn bộ 69 mẫu mới không có nội dung trùng với 203 mẫu gốc
2. **Đa dạng chủ đề**: Bao gồm các chủ đề ML từ cơ bản đến nâng cao
3. **Vietnamese language**: Tất cả Q&A đều bằng tiếng Việt
4. **Cùng kiến trúc**: Sử dụng cấu trúc JSON giống với dataset gốc
5. **Cân bằng tỷ lệ**: Duy trì tỷ lệ 70:15:15 cho train/validation/test

---

## 🔄 Cách Sử Dụng

### Sử dụng Extended Dataset riêng:
```python
import json

with open('train_extended.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        print(sample['question'])
```

### Kết hợp Original + Extended:
```python
import os
import json

all_samples = []

# Load original
for fname in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
    with open(f'examples_ai_flashcard/{fname}', 'r', encoding='utf-8') as f:
        for line in f:
            all_samples.append(json.loads(line))

# Load extended
for fname in ['train_extended.jsonl', 'validation_extended.jsonl', 'test_extended.jsonl']:
    with open(f'examples_ai_flashcard/{fname}', 'r', encoding='utf-8') as f:
        for line in f:
            all_samples.append(json.loads(line))

print(f"Total samples: {len(all_samples)}")  # 272
```

---

## 📝 Generation Scripts

Các script tạo dataset được lưu tại:
```
d:\NCKH_FLASH\
├── generate_extended_dataset.py           (40 QA pairs)
├── generate_large_extended_dataset.py     (100+ QA pairs)
└── generate_massive_extended_dataset.py   (70+ QA pairs - Final)
```

Để tạo lại dataset:
```bash
python generate_massive_extended_dataset.py
```

---

## 🎓 Chủ Đề Phủ Sóng

- **Cấp độ 1 (Basic)**: Khái niệm cơ bản ML
- **Cấp độ 2 (Intermediate)**: Kỹ thuật & thuật toán phổ biến
- **Cấp độ 3 (Advanced)**: Phương pháp & kiến trúc hiện đại

---

## ✅ Verification

Tất cả file đã được xác minh:
- ✓ Format JSON valid cho tất cả mẫu
- ✓ Không trùng lặp với dataset gốc
- ✓ Tủy lệ train/val/test đúng (70:15:15)
- ✓ Vietnamese language consistency

---

**Created**: April 2026
**Total Increase**: +34% data (69 new samples added to 203 original)
