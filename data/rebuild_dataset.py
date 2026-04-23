"""
Dataset Rebuilder: Convert weakly-aligned flashcard data to strict extractive QA
Pipeline: normalize → fuzzy locate span in normalized → map back to original

Usage:
    python rebuild_dataset.py --input_dir examples_ai_flashcard --output_dir examples_ai_flashcard_fixed
"""

import json
import os
import re
import unicodedata
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from difflib import SequenceMatcher

# Try to use rapidfuzz for better fuzzy matching, fallback to difflib
try:
    from rapidfuzz import fuzz
    USE_RAPIDFUZZ = True
except ImportError:
    USE_RAPIDFUZZ = False
    print("⚠️  rapidfuzz not found. Using difflib (slower). Install: pip install rapidfuzz")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetRebuilder:
    """Rebuild dataset to ensure answers are 100% substrings of context."""
    
    def __init__(self, threshold: float = 85.0):
        """
        Args:
            threshold: Fuzzy match score threshold (0-100)
                      85-90 recommended for balance between cleaning and preservation
        """
        self.threshold = threshold
        self.stats = {
            "total": 0,
            "exact": 0,
            "fuzzy_fixed": 0,
            "dropped": 0,
            "avg_fuzzy_score": 0.0
        }
        self.dropped_samples = []
        self.fuzzy_scores = []
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text for search only:
        - Lowercase
        - Unicode NFC normalization (preserve diacritics - critical for Vietnamese!)
        - Strip whitespace
        - Replace multiple spaces with single space
        
        ⚠️  DO NOT remove diacritics: "học" vs "hóc" vs "họa" are different!
        """
        # Unicode normalization NFC (compose characters, preserve diacritics)
        text = unicodedata.normalize('NFC', text)
        # Lowercase and strip
        text = text.lower().strip()
        # Replace multiple spaces
        text = ' '.join(text.split())
        return text
    
    def find_span_in_text(self, answer: str, context: str) -> Tuple[Optional[str], float, str]:
        """
        Find answer span in context using SequenceMatcher on ORIGINAL text.
        
        🎯 Golden rule: NEVER map normalized → original via heuristic
        
        Returns:
            (extracted_answer, score, status)
            - extracted_answer: exact substring from original context (or None)
            - score: match score 0-100
            - status: "exact", "fuzzy", or "dropped"
        """
        # Step 1: Try exact substring match on ORIGINAL text (fastest, safest path)
        if answer in context:
            return answer, 100.0, "exact"
        
        # Step 2: Use SequenceMatcher on ORIGINAL text for fuzzy matching
        extracted, score = self._extract_span_sequencematcher(answer, context)
        
        if extracted and score >= self.threshold:
            self.fuzzy_scores.append(score)
            return extracted, score, "fuzzy"
        
        # Step 3: No match - drop sample
        return None, score if score else 0.0, "dropped"
    
    def _extract_span_sequencematcher(self, answer: str, context: str) -> Tuple[Optional[str], float]:
        """
        Extract span using SequenceMatcher directly on ORIGINAL text.
        
        🎯 Golden principle:
        - Work on ORIGINAL text only
        - No normalized ↔ original mapping
        - Direct substring extraction
        - Strict validation to avoid partial/semantic corruption
        
        Returns: (extracted_span, score)
            - extracted_span: actual substring from context
            - score: % of answer that matched (0-100)
        """
        # Find longest matching block between answer and context
        matcher = SequenceMatcher(None, answer, context)
        match = matcher.find_longest_match(0, len(answer), 0, len(context))
        
        if match.size == 0:
            return None, 0.0
        
        # ✅ Fix 3: Reject partial matches that are too small
        # Require at least 60% of answer to be matched (avoid semantic corruption)
        coverage_answer = (match.size / len(answer)) if len(answer) > 0 else 0
        if coverage_answer < 0.60:  # Less than 60% match = too risky
            return None, coverage_answer * 100
        
        # Extract span directly from context using match position
        span_start = match.b
        span_end = match.b + match.size
        
        # Expand to word boundaries for better semantic meaning
        span = self._expand_to_word_boundary(context, span_start, span_end)
        
        # ✅ Fix 1: Validate span is actually in context (sanity check)
        if span not in context:
        ✅ Fix 4: Support Vietnamese with diacritics + unicode word chars
        
        Example:
            text: "The machine learning algorithm works well"
            span: [4:19] = "machine learning"
            → no expansion needed, already at word boundaries
        """
        # Vietnamese word character pattern (includes diacritics)
        # \w in regex with UNICODE flag matches most unicode letters/digits
        word_char_pattern = re.compile(r'\w', re.UNICODE)
        
        # Expand left to word boundary
        while start > 0 and word_char_pattern.match(text[start - 1]):
            start -= 1
        
        # Expand right to word boundary
        while end < len(text) and word_char_pattern.match(text[end]
        Expand span to complete words (avoid cutting mid-word).
        
        Example:
            text: "The machine learning algorithm works well"
            span: [4:19] = "machine learning"
            → no expansion needed, already at word boundaries
        """
        # Expand left to word boundary
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        
        # Expand right to word boundary
        while end < len(text) and text[end].isalnum():
            end += 1
        
        return text[start:end].strip()
    
    def process_file(self, input_path: str, output_path: str) -> Dict:
        """
        Process a single JSONL file.
        
        Returns:
            stats dictionary for this file
        """
        file_stats = {"exact": 0, "fuzzy_fixed": 0, "dropped": 0, "total": 0}
        samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    logger.warning(f"Skipping line {line_num}: invalid JSON")
                    continue
                
                context = data.get('context', '').strip()
                answer = data.get('answer', '').strip()
                
                if not context or not answer:
                    logger.warning(f"Line {line_num}: missing context or answer")
                    continue
                
                file_stats["total"] += 1
                self.stats["total"] += 1
                
                # Extract substring match
                extracted_answer, score, status = self.find_span_in_text(answer, context)
                
                if status == "exact":
                    file_stats["exact"] += 1
                    self.stats["exact"] += 1
                    data['answer'] = extracted_answer
                    data['_match_type'] = 'exact'
                    data['_match_score'] = 100.0
                    samples.append(data)
                    
                elif status == "fuzzy":
                    file_stats["fuzzy_fixed"] += 1
                    self.stats["fuzzy_fixed"] += 1
                    data['answer'] = extracted_answer
                    data['_match_type'] = 'fuzzy'
                    data['_match_score'] = score
                    data['_original_answer'] = answer
                    samples.append(data)
                    
                else:  # dropped
                    file_stats["dropped"] += 1
                    self.stats["dropped"] += 1
                    self.dropped_samples.append({
                        "file": Path(input_path).name,
                        "line": line_num,
                        "context": context[:100] + "..." if len(context) > 100 else context,
                        "answer": answer,
                        "fuzzy_score": score
                    })
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"📄 {Path(input_path).name}: "
                   f"Total={file_stats['total']}, "
                   f"Exact={file_stats['exact']}, "
                   f"Fuzzy={file_stats['fuzzy_fixed']}, "
                   f"Dropped={file_stats['dropped']}")
        
        return file_stats
    
    def rebuild(self, input_dir: str, output_dir: str):
        """
        Rebuild all dataset files (train/val/test).
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔄 Starting dataset rebuild...")
        logger.info(f"   Input:  {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Threshold: {self.threshold}")
        
        # Process each split
        for split in ['train', 'validation', 'test']:
            input_file = input_path / f"{split}.jsonl"
            output_file = output_path / f"{split}_fixed.jsonl"
            
            if not input_file.exists():
                logger.warning(f"⚠️  {input_file} not found, skipping")
                continue
            
            self.process_file(str(input_file), str(output_file))
        
        # Calculate final stats
        if self.stats["fuzzy_fixed"] > 0:
            self.stats["avg_fuzzy_score"] = sum(self.fuzzy_scores) / len(self.fuzzy_scores)
        
        self._print_summary(output_path)
        self._save_metadata(output_path)
    
    def _print_summary(self, output_dir: Path):
        """Print detailed summary."""
        logger.info("\n" + "="*70)
        logger.info("📊 DATASET REBUILD SUMMARY")
        logger.info("="*70)
        logger.info(f"Total samples:     {self.stats['total']}")
        logger.info(f"✅ Exact match:    {self.stats['exact']} ({self.stats['exact']*100/max(self.stats['total'],1):.1f}%)")
        logger.info(f"🔧 Fuzzy fixed:    {self.stats['fuzzy_fixed']} ({self.stats['fuzzy_fixed']*100/max(self.stats['total'],1):.1f}%)")
        logger.info(f"❌ Dropped:        {self.stats['dropped']} ({self.stats['dropped']*100/max(self.stats['total'],1):.1f}%)")
        
        if self.stats['fuzzy_fixed'] > 0:
            logger.info(f"📈 Avg fuzzy score: {self.stats['avg_fuzzy_score']:.1f}/100")
        
        logger.info(f"\n✨ Threshold: {self.threshold}")
        logger.info(f"📁 Output: {output_dir}")
        logger.info("="*70)
    
    def _save_metadata(self, output_dir: Path):
        """Save metadata and dropped samples."""
        # Stats JSON
        metadata = {
            "threshold": self.threshold,
            "stats": self.stats,
            "dropped_count": len(self.dropped_samples)
        }
        
        metadata_path = output_dir / "rebuild_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"📝 Metadata saved: {metadata_path}")
        
        # Dropped samples (for manual review)
        if self.dropped_samples:
            dropped_path = output_dir / "dropped_samples.jsonl"
            with open(dropped_path, 'w', encoding='utf-8') as f:
                for sample in self.dropped_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"⚠️  Dropped samples: {dropped_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rebuild dataset for extractive QA")
    parser.add_argument("--input_dir", default="data/examples_ai_flashcard", 
                       help="Input directory with original JSONL files")
    parser.add_argument("--output_dir", default="data/examples_ai_flashcard_fixed",
                       help="Output directory for fixed JSONL files")
    parser.add_argument("--threshold", type=float, default=85.0,
                       help="Fuzzy match threshold (85-90 recommended)")
    
    args = parser.parse_args()
    
    rebuilder = DatasetRebuilder(threshold=args.threshold)
    rebuilder.rebuild(args.input_dir, args.output_dir)
