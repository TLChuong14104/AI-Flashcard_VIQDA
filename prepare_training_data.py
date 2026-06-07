"""
Quick script to prepare training data from examples_ai_flashcard
Converts: {question, context, answer} → {text_input: "qag: context", text_output: "question: Q, answer: A"}
"""

import json
import os
from pathlib import Path

def prepare_data():
    input_dir = "data/examples_ai_flashcard"
    output_dir = "data/processed_data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'validation', 'test']:
        input_file = os.path.join(input_dir, f"{split}.jsonl")
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        
        if not os.path.exists(input_file):
            print(f"⚠️  {input_file} not found, skipping...")
            continue
        
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                count = 0
                for line in f_in:
                    try:
                        obj = json.loads(line.strip())
                        
                        # Convert format
                        text_input = f"qag: {obj.get('context', '')}"
                        question = obj.get('question', '').strip()
                        answer = obj.get('answer', '').strip()
                        
                        if not question or not answer:
                            continue
                        
                        # Format: "question: ..., answer: ..."
                        text_output = f"question: {question}, answer: {answer}"
                        
                        output_obj = {
                            'text_input': text_input,
                            'text_output': text_output,
                            'context': obj.get('context', ''),
                            'question': question,
                            'answer': answer
                        }
                        
                        f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                        count += 1
                    except Exception as e:
                        print(f"Error processing line: {e}")
                
                print(f"✅ {split}: {count} examples → {output_file}")

if __name__ == "__main__":
    prepare_data()
    print("\n✅ Data preparation complete!")
    print("Now run training:")
    print("python train.py fine_tuning --model VietAI/vit5-base --dataset_path data/processed_data --epoch 50 --batch 8 --lr 5e-5 --gradient_accumulation_steps 2 --checkpoint_dir ./models/vit5-flashcard-v2")
