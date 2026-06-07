# -*- coding: utf-8 -*-
"""
Gradio Interface for ViQAG - Vietnamese Question & Answer Generation
"""

import gradio as gr
import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# State storage for outputs
class OutputStorage:
    def __init__(self):
        self.raw_output = ""
        self.has_output = False
    
    def set(self, text):
        self.raw_output = text
        self.has_output = True
    
    def get(self):
        if self.has_output:
            return self.raw_output
        else:
            return "⏳ Chưa chạy sinh Q&A hoặc có lỗi capture"
    
    def reset(self):
        self.raw_output = ""
        self.has_output = False

output_storage = OutputStorage()


# Prefix đúng theo TASK_PREFIX['qag'] trong language_model.py
TASK_PREFIX_QAG = "generate question and answer"


class QAGModel:
    def __init__(self, model_path: str):
        """Load model and tokenizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
            self.model.eval()
            
            # Đọc add_prefix từ config (được lưu lúc train bởi TransformersQG.save())
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
                self.add_prefix = getattr(cfg, 'add_prefix', True)  # default True cho T5
            except Exception:
                self.add_prefix = True  # T5/ViT5 luôn cần prefix
            
            print(f"✅ Model loaded from {model_path}")
            print(f"   add_prefix = {self.add_prefix}")
            print(f"   Prefix sẽ dùng: '{TASK_PREFIX_QAG}: '" if self.add_prefix else "   Không dùng prefix")
        except Exception as e:
            raise Exception(f"❌ Không thể load model: {str(e)}")
    
    def generate(self, context: str, num_return_sequences: int = 3) -> list:
        """Generate Q&A from context - single inference pass"""
        try:
            if self.add_prefix:
                input_text = f"{TASK_PREFIX_QAG}: {context}"
            else:
                input_text = context
            print(f"\n[INPUT] Length: {len(input_text)} | Preview: {input_text[:80]}...")
            
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            ).to(self.device)
            
            print(f"[TOKENIZED] Input IDs shape: {inputs['input_ids'].shape}")
            print(f"[TOKENIZED] Input IDs sample (first 10): {inputs['input_ids'][0][:10].tolist()}")
            
            # Generate - config đã test và hoạt động (xem debug_empty_output.py)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=256,
                    num_beams=4,
                    num_return_sequences=1,
                    early_stopping=True,       # ✅ BẮT BUỘC True với T5
                    no_repeat_ngram_size=3,    # ✅ ngăn loop <pad>, KHÔNG block multi-pair
                    # ❌ KHÔNG dùng repetition_penalty: phạt context words → pad degeneration
                    # ❌ KHÔNG dùng early_stopping=False: gây loop <pad> đến max_new_tokens
                )
            
            output_token_ids = outputs[0].tolist()
            
            if outputs.shape[1] <= 1:
                result_text = ""
                print(f"[WARNING] Model chỉ sinh {outputs.shape[1]} token(s) - output trống!")
            else:
                result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                result_with_special = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                print(f"[DECODED] With special tokens: {repr(result_with_special[:200])}")
            
            print(f"[DECODED] Full output: {repr(result_text)}")
            
            if result_text.strip():
                output_storage.set(result_text)
            else:
                debug_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                output_storage.set(f"(Output trống) Token IDs: {output_token_ids[:30]} | With specials: {repr(debug_text[:100])}")
            
            return self._parse_combined_qa(result_text, num_return_sequences)
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"\n[EXCEPTION] {error_msg}")
            import traceback
            traceback.print_exc()
            output_storage.set(f"EXCEPTION: {error_msg}")
            raise Exception(error_msg)

    def generate_multi(self, context: str, num_pairs: int = 3) -> list:
        """Sentence-level inference: workaround khi model chỉ sinh 1 cặp/lần.
        
        Chạy model lần lượt cho từng câu → gom kết quả → dedup → lấy num_pairs cặp.
        Phù hợp khi training data ít, model chưa học được multi-pair generation.
        """
        import re
        
        # Tách câu đơn giản cho tiếng Việt
        sentences = re.split(r'(?<=[.!?])\s+', context.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Nếu chỉ có 1 câu hoặc đoạn quá ngắn, dùng toàn bộ context
        if len(sentences) <= 1:
            sentences = [context]
        
        print(f"\n[MULTI] Tách thành {len(sentences)} câu, cần {num_pairs} cặp")
        
        all_qa = []
        seen_questions = []  # dedup: tránh câu hỏi giống nhau
        raw_outputs = []
        
        for i, sentence in enumerate(sentences):
            if len(all_qa) >= num_pairs:
                break
            
            print(f"\n[MULTI] Câu {i+1}/{len(sentences)}: {repr(sentence[:60])}...")
            try:
                if self.add_prefix:
                    input_text = f"{TASK_PREFIX_QAG}: {sentence}"
                else:
                    input_text = sentence
                
                inputs = self.tokenizer(
                    input_text, return_tensors="pt",
                    max_length=512, truncation=True, padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=256,
                        num_beams=4,
                        num_return_sequences=1,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                    )
                
                result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"[MULTI] Output: {repr(result_text[:80])}")
                
                if not result_text.strip():
                    continue
                
                raw_outputs.append(result_text)
                pairs = self._parse_combined_qa(result_text, num_pairs=5)
                
                for qa in pairs:
                    q = qa.get("question", "")
                    a = qa.get("answer", "")
                    
                    # ✅ Validity filter: chỉ giữ cặp có câu hỏi thật và có answer
                    if not self._is_valid_qa(qa):
                        print(f"[MULTI] Bỏ qua cặp không hợp lệ: Q={repr(q[:40])} A={repr(a[:20])}")
                        continue
                    
                    # Dedup: bỏ qua nếu câu hỏi quá giống một câu đã có
                    is_dup = any(
                        self._similarity(q, seen_q) > 0.7
                        for seen_q in seen_questions
                    )
                    if not is_dup:
                        all_qa.append(qa)
                        seen_questions.append(q)
                
            except Exception as e:
                print(f"[MULTI] Lỗi câu {i+1}: {e}")
                continue
        
        # Lưu tất cả raw outputs để debug
        output_storage.set(" [SEP] ".join(raw_outputs) if raw_outputs else "(Không có output nào)")
        
        if not all_qa:
            # Fallback: chạy lại với toàn bộ context
            return self.generate(context, num_pairs)
        
        return all_qa[:num_pairs]

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Tính độ tương đồng đơn giản giữa 2 câu (Jaccard trên từ)"""
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _is_valid_qa(qa: dict) -> bool:
        """Kiểm tra cặp Q&A có hợp lệ không.
        
        Lọc bỏ các trường hợp:
        - Model copy nguyên câu context làm "câu hỏi" (không có dấu ?)
        - Model không sinh được answer (answer là placeholder)
        - Câu hỏi quá dài (> 200 ký tự) → nhiều khả năng là context bị copy
        """
        INVALID_QUESTIONS = {"⚠️ Model sinh output trống", "(Lỗi parse)", ""}
        INVALID_ANSWERS   = {"(Không có)", "(Lỗi parse)", ""}
        
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        
        if q in INVALID_QUESTIONS:
            return False
        if a in INVALID_ANSWERS:
            return False
        if not q.endswith("?"):          # Không phải câu hỏi thật
            return False
        if len(q) > 200:                 # Câu hỏi quá dài → copy context
            return False
        return True


    
    def _parse_combined_qa(self, text: str, num_pairs: int = 3) -> list:
        """Parse combined Q&A format from model output"""
        results = []
        
        if not text or len(text.strip()) == 0:
            return [{
                "question": "⚠️ Model sinh output trống",
                "answer": "(Xem Raw Output để debug)"
            }]
        
        # Split by [SEP] token (model có thể sinh nhiều cặp)
        sep_variants = [" [SEP] ", "[SEP]", " </s> ", "</s>"]
        pairs_text = [text]  # mặc định 1 cặp
        for sep in sep_variants:
            if sep in text:
                pairs_text = [p for p in text.split(sep) if p.strip()]
                break
        
        for pair_text in pairs_text:
            pair_text = pair_text.strip()
            if not pair_text:
                continue
            
            qa = self._extract_single_qa(pair_text)
            if qa["question"] not in ["", "(Lỗi parse)"] or qa["answer"] not in ["", "(Không có)"]:
                results.append(qa)
            
            if len(results) >= num_pairs:
                break
        
        if not results:
            results.append({
                "question": "Không parse được Q&A",
                "answer": text[:300] if len(text) > 300 else text
            })
        
        return results[:num_pairs]
    
    def _extract_single_qa(self, text: str) -> dict:
        """Extract single Q&A from text - preserve original case"""
        text = text.strip()
        text_lower = text.lower()  # chỉ dùng để tìm vị trí, KHÔNG dùng để extract
        q = ""
        a = ""
        
        if "question:" in text_lower:
            q_start = text_lower.find("question:") + len("question:")
            
            # Tìm separator ", answer:" (có dấu phẩy trước)
            for ans_sep in [", answer:", ",answer:", " answer:"]:
                sep_pos = text_lower.find(ans_sep, q_start)
                if sep_pos != -1:
                    # Trích từ text GỐC để giữ nguyên hoa/thường
                    q = text[q_start:sep_pos].strip().rstrip(',').strip()
                    a = text[sep_pos + len(ans_sep):].strip()
                    break
            
            if not q:  # fallback: chỉ có question không có answer
                ans_pos = text_lower.find("answer:", q_start)
                if ans_pos != -1:
                    q = text[q_start:ans_pos].strip().rstrip(',').strip()
                    a = text[ans_pos + len("answer:"):].strip()
                else:
                    q = text[q_start:].strip()
        
        # Fallback: không có "question:" prefix
        if not q and not a and "," in text:
            parts = text.split(",", 1)
            q = parts[0].strip()
            a = parts[1].strip() if len(parts) > 1 else ""
        
        # Last resort
        if not q and text:
            q = text[:100] + ("?" if not text.endswith("?") else "")
            a = ""
        
        return {
            "question": q if q else "(Lỗi parse)",
            "answer": a if a else "(Không có)"
        }


# Global model instance
model_instance = None

def initialize_model(model_path: str):
    """Initialize model"""
    global model_instance
    try:
        model_instance = QAGModel(model_path)
        return "✅ Model đã tải thành công!"
    except Exception as e:
        return f"❌ {str(e)}"


def generate_qa(context: str, num_pairs: int = 3) -> tuple:
    """
    Generate questions and answers from context
    Returns: (formatted_text, raw_output, json_data)
    """
    global model_instance
    
    if model_instance is None:
        return "", "❌ Model chưa được tải", {"error": "Model chưa được tải"}
    
    if not context.strip():
        return "", "", {"error": "Vui lòng nhập đoạn văn bản"}
    
    if len(context) < 30:
        return "", "", {"error": "Đoạn văn bản quá ngắn"}
    
    try:
        print(f"\n{'='*70}")
        print(f"🔄 BẮT ĐẦU SINH Q&A | Context: {len(context)} chars")
        print(f"{'='*70}")
        
        # Reset storage
        output_storage.reset()
        
        # Generate Q&A - dùng sentence-level inference để ra nhiều cặp hơn
        # (workaround khi model chỉ sinh 1 cặp do training data ít)
        qa_results = model_instance.generate_multi(context, num_pairs=num_pairs)
        
        # Get raw output from storage
        raw_output_text = output_storage.get()
        
        print(f"\n[RESULTS] Generated {len(qa_results)} Q&A pairs")
        print(f"[RESULTS] Raw output length: {len(raw_output_text)}")
        
        # Format output
        output = {
            "status": "success",
            "context_length": len(context),
            "num_generated": len(qa_results),
            "raw_output": raw_output_text,
            "raw_output_length": len(raw_output_text),
            "qa_pairs": []
        }
        
        for i, qa in enumerate(qa_results, 1):
            output["qa_pairs"].append({
                "id": i,
                "question": qa['question'],
                "answer": qa['answer']
            })
        
        # Format text output
        result_text = f"""📄 **Đoạn văn bản ({len(context)} ký tự):**
---
{context}

✨ **Kết quả sinh Q&A ({len(qa_results)} cặp):**
"""
        
        for qa in output["qa_pairs"]:
            result_text += f"""
---
❓ **Câu hỏi {qa['id']}:** 
{qa['question']}

✅ **Câu trả lời {qa['id']}:** 
{qa['answer']}
"""
        
        print(f"\n{'='*70}")
        print(f"✅ HOÀN THÀNH | Output: {len(raw_output_text)} chars | Q&A: {len(qa_results)}")
        print(f"{'='*70}\n")
        
        return result_text, raw_output_text, output
        
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        print(f"\n{error_msg}\n")
        import traceback
        traceback.print_exc()
        output_storage.reset()
        return "", error_msg, {"status": "error", "error": str(e)}


# Create Gradio Interface
with gr.Blocks(title="ViQAG - Vietnamese QA Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🎯 ViQAG - Vietnamese Question & Answer Generator
    
**Tự động sinh câu hỏi và câu trả lời từ đoạn văn bản tiếng Việt**

Hãy:
1. Nhập đường dẫn model (hoặc dùng default)
2. Nhấn "Tải Model"
3. Nhập đoạn văn bản
4. Nhấn "Sinh Q&A"
""")
    
    with gr.Row():
        model_path_input = gr.Textbox(
            label="🤖 Đường dẫn Model",
            value="./models/vit5-flashcard/epoch_10",
            placeholder="Ví dụ: ./models/vit5-flashcard/epoch_10"
        )
        load_btn = gr.Button("📥 Tải Model", variant="primary")
        status_text = gr.Textbox(label="Trạng thái", interactive=False)
    
    load_btn.click(
        fn=initialize_model,
        inputs=[model_path_input],
        outputs=[status_text]
    )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Nhập đoạn văn bản")
            context_input = gr.Textbox(
                label="Đoạn văn bản",
                placeholder="Nhập đoạn văn bản tiếng Việt (tối thiểu 30 ký tự)...",
                lines=8,
                max_lines=20
            )
            
            num_pairs_slider = gr.Slider(
                label="Số cặp Q&A cần sinh",
                minimum=1,
                maximum=5,
                value=3,
                step=1
            )
            
            with gr.Row():
                generate_btn = gr.Button("🚀 Sinh Q&A", variant="primary", scale=2)
                clear_btn = gr.Button("🔄 Xóa", scale=1)
        
        with gr.Column():
            gr.Markdown("### 📊 Kết quả")
            output_text = gr.Markdown(
                label="Câu hỏi và câu trả lời được sinh"
            )
    
    gr.Markdown("---")
    with gr.Row():
        raw_output = gr.Textbox(
            label="🔍 Raw Output từ Model (Debug Info)",
            lines=5,
            max_lines=10,
            interactive=False,
            placeholder="Sẽ hiển thị output thô từ model sau khi sinh"
        )
    
    with gr.Row():
        output_json = gr.JSON(label="📋 Dữ liệu JSON (Full Format)")
    
    # Event handlers
    def generate_and_format(context, num_pairs):
        return generate_qa(context, num_pairs)
    
    generate_btn.click(
        fn=generate_and_format,
        inputs=[context_input, num_pairs_slider],
        outputs=[output_text, raw_output, output_json]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", {}),
        outputs=[output_text, raw_output, output_json]
    )
    
    # Example
    gr.Examples(
        examples=[
            ["""Đại học Bách khoa Hà Nội là một trường đại học công lập hàng đầu ở Việt Nam. 
            Trường được thành lập năm 1956 với mục đích đào tạo các kỹ sư giỏi cho đất nước. 
            Hiện nay, trường có hơn 15.000 sinh viên và cung cấp các chương trình đào tạo trong lĩnh vực 
            kỹ thuật, công nghệ thông tin, kinh tế và quản lý.""", 3]
        ],
        inputs=[context_input, num_pairs_slider],
        outputs=[output_text, raw_output, output_json],
        fn=generate_and_format,
        cache_examples=False
    )


if __name__ == "__main__":
    print("🚀 Khởi động Gradio interface...")
    print("📱 Truy cập: http://localhost:7860")
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

