# Hướng dẫn sử dụng trên Kaggle

## Bước 1: Clone repository

```python
!git clone https://github.com/TLChuong14104/AI-Flashcard_VIQDA.git
%cd /kaggle/working/AI-Flashcard_VIQDA
```

## Bước 2: QUAN TRỌNG - Cài đặt packages với versions tương thích

**LỖI THƯỜNG GẶP:** `TypeError: argument 'vocab': 'dict' object cannot be converted to 'Sequence'`

**GIẢI PHÁP:** Cài đặt transformers >= 4.36.0 (version mới hơn)

```python
!pip install --upgrade \
  'transformers>=4.36.0' \
  'sentencepiece>=0.2.0' \
  'torch>=2.0.0' \
  'protobuf>=3.20.0'
```

## Bước 3: Cài đặt các package khác

```python
!pip install -q -r requirements_kaggle.txt
```

## Bước 4: Import và chạy training

```python
import sys
sys.path.insert(0, '/kaggle/working/AI-Flashcard_VIQDA')

from ViQAG.plms.trainer import Trainer

# Chạy training
trainer = Trainer(
    model_name='VietAI/vit5-base',
    batch_size=4,
    learning_rate=0.0001,
    num_epochs=10,
    gradient_accumulation_steps=4
)
```

## Troubleshooting

### Lỗi: "TokenizerFast" not available
**Giải pháp:** Đã fix trong code, nó sẽ tự fallback sang slow tokenizer

### Lỗi: HuggingFace Hub rate limit
**Giải pháp:** Thêm HF_TOKEN (optional):
```python
import os
os.environ['HF_TOKEN'] = 'your_hf_token_here'
```

### Lỗi: Out of memory
**Giải pháp:** Giảm batch_size:
```python
batch_size=2  # hoặc thậm chí 1
gradient_accumulation_steps=8  # tăng lên để compensate
```

## Key Settings

- **Model**: VietAI/vit5-base (Vietnamese T5)
- **Dataset location**: `data/examples_ai_flashcard/`
- **Training split**: 70% train / 15% validation / 15% test
- **Recommended config**:
  - Epochs: 10
  - Batch size: 4 (hoặc 2 nếu OOM)
  - Learning rate: 0.0001
  - Gradient accumulation: 4

## Support

Nếu vẫn gặp lỗi tokenizer, thử:
1. Restart kernel
2. Xóa cache: `!pip cache purge`
3. Reinstall transformers: `!pip install --force-reinstall transformers>=4.36.0`
