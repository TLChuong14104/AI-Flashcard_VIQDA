#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEBUG SCRIPT: Chẩn đoán lỗi "Model sinh output trống"

Chạy: python debug_empty_output.py [model_path]
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "./models/vit5-flashcard/epoch_10"

TEST_CONTEXTS = [
    "Học máy là lĩnh vực nghiên cứu trong trí tuệ nhân tạo. Nó giúp máy tính học từ dữ liệu mà không cần lập trình rõ ràng cho mọi tác vụ.",
    "Mạng nơ-ron nhân tạo được lấy cảm hứng từ não người. Chúng gồm nhiều tầng xử lý thông tin theo cấu trúc phân cấp.",
]

PREFIXES = [
    "generate question and answer: ",   # TASK_PREFIX['qag'] từ language_model.py
    "qag: ",                             # prefix ngắn (app_gradio dùng)
    "",                                  # không prefix
]

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)

def check_model_exists(path):
    section("1. KIỂM TRA MODEL PATH")
    if not os.path.exists(path):
        print(f"❌ KHÔNG TÌM THẤY: {path}")
        print("   → Model chưa được train hoặc sai đường dẫn!")
        return False
    files = os.listdir(path)
    print(f"✅ Tìm thấy: {path}")
    print(f"   Files: {files}")
    required = ["config.json"]
    for r in required:
        status = "✅" if r in files else "❌ THIẾU"
        print(f"   {status} {r}")
    return "config.json" in files

def check_config(path):
    section("2. KIỂM TRA CONFIG")
    try:
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        cfg_dict = config.to_dict()
        print(f"✅ model_type  : {cfg_dict.get('model_type', '???')}")
        print(f"   add_prefix  : {cfg_dict.get('add_prefix', 'KHÔNG CÓ ← ĐÂY CÓ THỂ LÀ VẤN ĐỀ')}")
        print(f"   vocab_size  : {cfg_dict.get('vocab_size')}")
        print(f"   decoder_start_token_id: {cfg_dict.get('decoder_start_token_id')}")
        print(f"   eos_token_id: {cfg_dict.get('eos_token_id')}")
        return config
    except Exception as e:
        print(f"❌ Lỗi load config: {e}")
        return None

def load_model_and_tokenizer(path):
    section("3. LOAD MODEL & TOKENIZER")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        print(f"✅ Tokenizer loaded | vocab_size={tokenizer.vocab_size}")
        print(f"   eos_token    : {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
        print(f"   pad_token    : {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
        print(f"   decoder_start: {tokenizer.decode([tokenizer.pad_token_id or 0])}")
    except Exception as e:
        print(f"❌ Lỗi load tokenizer: {e}")
        return None, None, device

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to(device)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded  | params={total_params/1e6:.1f}M")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return tokenizer, None, device

    return tokenizer, model, device

def test_prefix(tokenizer, model, device, context, prefix, prefix_label):
    input_text = f"{prefix}{context}"
    inputs = tokenizer(
        input_text, return_tensors="pt",
        max_length=512, truncation=True, padding=True
    ).to(device)

    n_input_tokens = inputs['input_ids'].shape[1]
    print(f"\n  [{prefix_label}]")
    print(f"   Input ({n_input_tokens} tokens): {repr(input_text[:80])}...")

    try:
        with torch.no_grad():
            # Strategy A: beam search thuần
            out_beam = model.generate(
                **inputs, max_new_tokens=200, num_beams=4,
                early_stopping=True, no_repeat_ngram_size=3
            )
            text_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
            raw_beam  = tokenizer.decode(out_beam[0], skip_special_tokens=False)
            print(f"   [Beam]    tokens={out_beam.shape[1]:3d} | text={repr(text_beam[:80])}")
            if not text_beam.strip():
                print(f"            raw_ids={out_beam[0].tolist()}")
                print(f"            with_specials={repr(raw_beam[:100])}")

            # Strategy B: greedy
            out_greedy = model.generate(
                **inputs, max_new_tokens=200, num_beams=1, do_sample=False
            )
            text_greedy = tokenizer.decode(out_greedy[0], skip_special_tokens=True)
            print(f"   [Greedy]  tokens={out_greedy.shape[1]:3d} | text={repr(text_greedy[:80])}")

            # Strategy C: sampling
            out_sample = model.generate(
                **inputs, max_new_tokens=200, do_sample=True,
                temperature=0.7, top_p=0.9, num_beams=1
            )
            text_sample = tokenizer.decode(out_sample[0], skip_special_tokens=True)
            print(f"   [Sample]  tokens={out_sample.shape[1]:3d} | text={repr(text_sample[:80])}")

        return text_beam  # trả về kết quả beam search
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_tests(tokenizer, model, device):
    section("4. THỬ GENERATE VỚI CÁC PREFIX KHÁC NHAU")
    # ❤️ Thứ tự quan trọng: prefix đúng (từ TASK_PREFIX trong language_model.py) phải lên đầu
    prefixes = [
        ("generate question and answer: ", "TASK_PREFIX[qag] ← ĐÚNG (training format)"),
        ("generate question: ",            "TASK_PREFIX[qg]"),
        ("qag: ",                          "short_prefix (app cũ dùng - SAI)"),
        ("",                               "no_prefix"),
    ]
    results = []
    for ctx in TEST_CONTEXTS:
        print(f"\n📄 Context: {repr(ctx[:60])}...")
        for prefix, label in prefixes:
            text = test_prefix(tokenizer, model, device, ctx, prefix, label)
            results.append({
                "prefix": label, "prefix_val": prefix,
                "context": ctx[:40], "output": text,
                "empty": not (text and text.strip())
            })
    return results

def diagnose(results):
    section("5. CHẨN ĐOÁN")
    all_empty = all(r["empty"] for r in results)
    some_empty = any(r["empty"] for r in results)
    
    # Nhóm theo prefix
    prefix_summary = {}
    for r in results:
        key = r["prefix"]
        if key not in prefix_summary:
            prefix_summary[key] = []
        prefix_summary[key].append(r["empty"])
    
    print("\n📊 Kết quả theo prefix:")
    working_prefixes = []
    for label, empties in prefix_summary.items():
        success_rate = sum(1 for e in empties if not e) / len(empties)
        status = "✅" if success_rate > 0.5 else "❌"
        print(f"   {status} [{label}] → {success_rate:.0%} có output")
        if success_rate > 0.5:
            working_prefixes.append(label)
    
    if all_empty:
        print("\n❌ TẤT CẢ output đều trống!")
        print("➡️ Đây không phải vấn đề prefix — model có vấn đề sâu hơn")
        print("\n🔧 Nguyên nhân có thể:")
        print("   1. Training bị lỗi (loss không giảm, checkpoint hỏng)")
        print("   2. Dataset đầu ra không đúng format")
        print("   3. eos_token_id / decoder_start_token_id sai")
        print("\n💡 Giải pháp:")
        print("   python debug_empty_output.py VietAI/vit5-base  ← test base model")
        print("   Nếu base model cũng trống → vấn đề tokenizer/config")
        print("   Nếu base model có output → training có lỗi, cần train lại")
    elif "TASK_PREFIX[qag] ← ĐÚNG (training format)" in working_prefixes:
        print("\n✅ XÁC NHẬN: Lỗi là PREFIX MISMATCH!")
        print("➡️ Model hoạt động đúng khi dùng prefix 'generate question and answer: '")
        print("➡️ app_gradio.py đã được sửa — chạy lại Gradio app để kiểm tra!")
    elif some_empty:
        print("\n⚠️ Một số prefix cho kết quả trống.")
        if working_prefixes:
            wl = working_prefixes[0]
            sample = next(r for r in results if r["prefix"] == wl and not r["empty"])
            print(f"   Prefix hoạt động: '{sample.get('prefix_val', wl)}'")
            print(f"   Output mẫu: {repr(str(sample['output'])[:100])}")
    else:
        print("\n✅ Model đang hoạt động bình thường!")
        for r in results[:2]:
            print(f"   [{r['prefix']}] → {repr(str(r['output'])[:80])}")

if __name__ == "__main__":
    print(f"\n🔬 DEBUG: {MODEL_PATH}")
    
    ok = check_model_exists(MODEL_PATH)
    if not ok:
        print("\n🛑 Dừng lại — Model không tồn tại")
        sys.exit(1)
    
    config = check_config(MODEL_PATH)
    tokenizer, model, device = load_model_and_tokenizer(MODEL_PATH)
    
    if model is None:
        print("\n🛑 Dừng lại — Không load được model")
        sys.exit(1)
    
    results = run_tests(tokenizer, model, device)
    diagnose(results)
    
    print(f"\n{'='*65}")
    print("  DEBUG COMPLETE")
    print('='*65)
