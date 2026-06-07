#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script to check if model is working
Run: python test_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_model(model_path: str = "./models/vit5-flashcard/epoch_10"):
    """Test model generation"""
    
    print(f"\n{'='*70}")
    print(f"🧪 MODEL TEST SCRIPT")
    print(f"{'='*70}\n")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    
    # Load model
    print(f"\n📥 Loading model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        model.eval()
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test context
    test_contexts = [
        "Đại học Bách khoa Hà Nội là trường công lập hàng đầu ở Việt Nam.",
        "Machine Learning là lĩnh vực của trí tuệ nhân tạo giúp máy tính học từ dữ liệu.",
        "Python là ngôn ngữ lập trình phổ biến trong khoa học dữ liệu."
    ]
    
    for idx, context in enumerate(test_contexts, 1):
        print(f"\n{'='*70}")
        print(f"📝 TEST {idx}")
        print(f"{'='*70}")
        
        print(f"\n[INPUT] Context length: {len(context)}")
        print(f"[INPUT] Preview: {context[:80]}...\n")
        
        # Prepare input
        input_text = f"qag: {context}"
        print(f"[FORMAT] Full input: {input_text[:100]}...")
        
        try:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(device)
            
            print(f"[TOKENIZED] Shape: {inputs['input_ids'].shape}")
            print(f"[TOKENIZED] First 15 tokens: {inputs['input_ids'][0][:15].tolist()}")
            
            # Generate
            print(f"\n[GENERATING...]")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=256,
                    num_beams=3,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9
                )
            
            print(f"[GENERATED] Output shape: {outputs.shape}")
            print(f"[GENERATED] Output tokens count: {outputs.shape[1]}")
            
            if outputs.shape[1] > 0:
                print(f"[GENERATED] All tokens: {outputs[0].tolist()}")
            else:
                print(f"[WARNING] 0 tokens generated!")
            
            # Decode
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result_with_special = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            print(f"\n[DECODED] Length: {len(result)} chars")
            print(f"[DECODED] Output (skip_special=True): {repr(result)}")
            print(f"[DECODED] Output (skip_special=False): {repr(result_with_special[:150])}")
            
            if len(result) == 0:
                print(f"⚠️ OUTPUT IS EMPTY!")
            elif "question:" in result.lower():
                print(f"✅ Output contains 'question:' - good format!")
            else:
                print(f"⚠️ Output doesn't match expected format")
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"✅ TEST COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import sys
    
    model_path = "./models/vit5-flashcard/epoch_10"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    test_model(model_path)
