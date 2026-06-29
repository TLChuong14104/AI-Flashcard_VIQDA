import json
import os
import re
from glob import glob
from tqdm import tqdm
from typing import Dict
from underthesea import sent_tokenize
from difflib import SequenceMatcher
import fire

HIGHLIGHT_TOKEN = '<hl>'


def find_best_span(paragraph: str, answer: str, min_ratio: float = 0.5) -> int:
    """Find the best matching span in paragraph for the given answer.
    Returns the start position of the best match, or -1 if no good match found.
    """
    answer_len = len(answer)
    best_pos = -1
    best_ratio = min_ratio
    
    # Sliding window: try spans of similar length to the answer
    for window_mult in [1.0, 0.8, 1.2, 0.6, 1.5]:
        window = max(10, int(answer_len * window_mult))
        for i in range(0, len(paragraph) - window + 1, 3):  # step=3 for speed
            candidate = paragraph[i:i + window]
            ratio = SequenceMatcher(None, answer.lower(), candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i
    
    return best_pos

class QGDataProcessor:
    def __init__(self):
        self.input_dir: str = 'data/examples'
        self.output_dir: str = 'data/processed_data'

    def get_sentence(self, document: str):
        """Tokenize Vietnamese text into sentences using Underthesea"""
        try:
            sentences = sent_tokenize(document)
            return [s.strip() for s in sentences if s.strip()]
        except:
            return [document]

    def jsonline_reader(self, filename: str):
        """Safely read JSONL file with error reporting"""
        examples = []
        with open(filename, 'r', encoding='utf-8') as f_reader:
            for line_num, line in enumerate(f_reader, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON at {filename}:{line_num} - {e}")
                    continue
        return examples

    def process_single_data(self, data: Dict):
        """ Convert single raw json data into QG format """
        example = {'question': data["question"], 'paragraph': data["context"], 'answer': data["answer"]}

        # get sentence
        position = example['paragraph'].find(example['answer'])
        
        # Fallback 1: try _original_answer if available
        if position == -1 and '_original_answer' in data:
            orig = data['_original_answer']
            pos2 = example['paragraph'].find(orig)
            if pos2 != -1:
                example['answer'] = orig
                position = pos2
        
        # Fallback 2: fuzzy find best matching span
        if position == -1:
            best_pos = find_best_span(example['paragraph'], example['answer'], min_ratio=0.55)
            if best_pos != -1:
                # Extract the actual span from paragraph
                ans_len = len(example['answer'])
                # Try a few window sizes around the best position
                best_span = None
                best_r = 0
                for mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    w = max(10, int(ans_len * mult))
                    end = min(best_pos + w, len(example['paragraph']))
                    candidate = example['paragraph'][best_pos:end]
                    r = SequenceMatcher(None, example['answer'].lower(), candidate.lower()).ratio()
                    if r > best_r:
                        best_r = r
                        best_span = candidate
                        position = best_pos
                if best_span and best_r >= 0.55:
                    example['answer'] = best_span
                else:
                    position = -1
        
        if position == -1:
            return None
        
        before_tmp = self.get_sentence(example['paragraph'][:position])
        if len(before_tmp) == 0:
            before = ''
            before_sentence = ''
        else:
            if before_tmp[-1].endswith('.'):
                before = ' '.join(before_tmp)
                before_sentence = ''
            else:
                before = ' '.join(before_tmp[:-1])
                before_sentence = before_tmp[-1]
                before_sentence = before_sentence if before_sentence.endswith(' ') else '{} '.format(before_sentence)
        after_tmp = self.get_sentence(example['paragraph'][position + len(example['answer']):])
        if len(after_tmp) == 0:
            after = ''
            after_sentence = ''
        else:
            after = ' '.join(after_tmp[1:])
            after_sentence = after_tmp[0]
            after_sentence = after_sentence if after_sentence.startswith(' ') else ' {}'.format(after_sentence)
        example['sentence'] = '{}{}{}'.format(before_sentence, example['answer'], after_sentence)

        # get paragraph_sentence
        before = '' if before == '' else '{} '.format(before)
        after = '' if after == '' else ' {}'.format(after)
        source_text = '{0}{1} {2} {1}{3}'.format(before, HIGHLIGHT_TOKEN, example['sentence'], after)
        example['paragraph_sentence'] = re.sub(r'\s+', ' ', source_text)

        # get paragraph_answer
        source_text = '{0}{1} {2} {1}{3}'.format(
            example['paragraph'][:position], HIGHLIGHT_TOKEN, example['answer'],
            example['paragraph'][position + len(example['answer']):])
        example['paragraph_answer'] = re.sub(r'\s+', ' ', source_text)

        # get sentence_answer
        if len(before_tmp) == 0 or before_tmp[-1].endswith('.'):
            before = ''
        else:
            before = before_tmp[-1] if before_tmp[-1].endswith(' ') else '{} '.format(before_tmp[-1])
        if len(after_tmp) == 0:
            after = ''
        else:
            after = after_tmp[0] if after_tmp[0].startswith(' ') else ' {}'.format(after_tmp[0])
        source_text = '{0}{1} {2} {1}{3}'.format(before, HIGHLIGHT_TOKEN, example['answer'], after)
        example['sentence_answer'] = re.sub(r'\s+', ' ', source_text)
        
        # Add difficulty if present
        if 'difficulty' in data:
            example['difficulty'] = data['difficulty']

        # Add paragraph_question for QA training (question + context)
        example['paragraph_question'] = f"question: {example['question']}, context: {example['paragraph']}"

        return example

    def process_data(self, input_dir='data/examples', output_dir='data'):
        self.input_dir = input_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        
        # Support both 'validation' and 'dev' naming
        splits = {}
        for split_name in ['train', 'validation', 'dev', 'test']:
            path = os.path.join(input_dir, f'{split_name}.jsonl')
            if os.path.exists(path):
                splits[split_name] = path
        
        for split, filepath in splits.items():
            # Determine output split name
            output_split = 'dev' if split == 'validation' else split
            
            print(f"Reading {filepath}...")
            json_data = self.jsonline_reader(filepath)
            
            processed_data = []
            for single_data in tqdm(json_data, desc=f"Processing {split}"):
                try:
                    processed = self.process_single_data(single_data)
                    if processed is not None:  # Skip if answer not found in paragraph
                        processed_data.append(processed)
                except Exception as e:
                    print(f"Warning: Skipping sample due to error: {e}")
                    continue
            
            output_file = os.path.join(self.output_dir, f'{output_split}.jsonl')
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"OK {split}: {len(processed_data)}/{len(json_data)} samples processed -> {output_file}")

if __name__ == '__main__':
    fire.Fire(QGDataProcessor)