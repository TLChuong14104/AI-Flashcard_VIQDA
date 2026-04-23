"""
Dataset Rebuilder: Convert weakly-aligned flashcard data to strict extractive QA

Strategy: 
1. Try exact substring match first
2. If fail → use SequenceMatcher + extract exact matched span (no expansion)
3. Strict: require ≥80% match coverage + ≥90% threshold
4. Drop if span too long (preserve semantic integrity, never truncate)
5. Log answer length distribution before/after

Semantics (SQuAD-style extractive QA):
- Answer = exact substring from context
- _answer_start = starting position in context
- No word boundary expansion (preserves training signal purity)

Usage:
    python rebuild_dataset.py --input_dir examples_ai_flashcard --output_dir examples_ai_flashcard_fixed --threshold 90
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetRebuilder:
    """Rebuild dataset to ensure answers are 100% substrings of context."""
    
    # Global config
    THRESHOLD_DEFAULT = 90.0
    COVERAGE_MIN = 0.80  # Require ≥80% of answer matched
    ANSWER_MAX_LEN = 200  # Drop if too long (preserve semantic integrity, not truncate)
    
    def __init__(self, threshold: float = THRESHOLD_DEFAULT):
        """
        Args:
            threshold: Fuzzy match score threshold (0-100)
                      90 recommended for strict semantic preservation
        """
        self.threshold = threshold
        self.stats = {
            "total": 0,
            "exact": 0,
            "fuzzy_fixed": 0,
            "dropped": 0,
            "avg_fuzzy_score": 0.0,
            "avg_answer_length_before": 0.0,
            "avg_answer_length_after": 0.0,
        }
        self.dropped_samples = []
        self.fuzzy_scores = []
        self.answer_lengths_before = []
        self.answer_lengths_after = []
    
    def find_span_in_text(self, answer: str, context: str) -> Tuple[Optional[str], float, str, Optional[int]]:
        """
        Find answer span in context using pure extractive QA semantics.
        
        Strategy:
        1. Try exact match first (fastest, safest)
        2. If fail → use SequenceMatcher to find best match
        3. Extract exact span (NO word boundary expansion to preserve training signal)
        4. Strict: require ≥80% match coverage + ≥90% threshold
        5. Drop if span too long (preserve semantic precision)
        
        Returns:
            (extracted_answer, score, status, answer_start_index)
        """
        self.answer_lengths_before.append(len(answer))
        
        # Step 1: Try exact substring match
        if answer in context:
            self.answer_lengths_after.append(len(answer))
            return answer, 100.0, "exact", context.index(answer)
        
        # Step 2: Fuzzy matching with window extraction
        extracted, score, start_idx = self._extract_span_with_window(answer, context)
        
        if extracted and score >= self.threshold:
            self.fuzzy_scores.append(score)
            self.answer_lengths_after.append(len(extracted))
            return extracted, score, "fuzzy", start_idx
        
        # Step 3: No good match - drop sample
        return None, score if score else 0.0, "dropped", None
    
    def _extract_span_with_window(self, answer: str, context: str) -> Tuple[Optional[str], float, Optional[int]]:
        """
        Extract answer using SequenceMatcher + exact span extraction (SQuAD-style).
        
        🎯 Pure extractive QA semantics:
        - Use SequenceMatcher to find best match
        - Extract EXACT matched span (no word boundary expansion)
        - Why no expansion? To preserve exact semantics + avoid training noise
        - Drop if span too long (preserve semantic precision)
        
        Returns: (extracted_span, score, start_index_in_context)
        """
        # Find longest matching block
        matcher = SequenceMatcher(None, answer, context)
        match = matcher.find_longest_match(0, len(answer), 0, len(context))
        
        if match.size == 0:
            return None, 0.0, None
        
        # Calculate coverage: what % of answer was matched
        coverage_answer = (match.size / len(answer)) if len(answer) > 0 else 0
        score = coverage_answer * 100
        
        # Strict: require at least COVERAGE_MIN of answer matched
        if coverage_answer < self.COVERAGE_MIN:
            return None, score, None
        
        # Extract EXACT matched span (pure extractive QA - no expansion)
        # This preserves semantic integrity for model training
        span_start = match.b
        span_end = match.b + match.size
        span = context[span_start:span_end].strip()
        
        # Check length: DROP if too long (don't truncate/corrupt semantic meaning)
        if len(span) > self.ANSWER_MAX_LEN:
            return None, score, span_start  # Return score for logging but reject span
        
        # Edge case warning: log if coverage suggests possible multi-span answer
        # (longest block found, but answer might have multiple important parts)
        if 0.8 <= coverage_answer < 0.95 and match.size < len(answer) * 0.5:
            # Low coverage block - might be multi-span, log for review
            logger.debug(f"Edge case: low-coverage match ({score:.1f}%) - possible multi-span answer")
        
        return span if span else None, score, span_start
    
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
                extracted_answer, score, status, answer_start = self.find_span_in_text(answer, context)
                
                if status == "exact":
                    file_stats["exact"] += 1
                    self.stats["exact"] += 1
                    data['answer'] = extracted_answer
                    data['_match_type'] = 'exact'
                    data['_match_score'] = 100.0
                    data['_answer_start'] = answer_start
                    samples.append(data)
                    
                elif status == "fuzzy":
                    file_stats["fuzzy_fixed"] += 1
                    self.stats["fuzzy_fixed"] += 1
                    data['answer'] = extracted_answer
                    data['_match_type'] = 'fuzzy'
                    data['_match_score'] = score
                    data['_answer_start'] = answer_start
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
        """Rebuild all dataset files (train/val/test)."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔄 Starting dataset rebuild...")
        logger.info(f"   Input:  {input_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Threshold: {self.threshold}")
        logger.info(f"   Coverage min: {self.COVERAGE_MIN * 100}%")
        logger.info(f"   Answer max len: {self.ANSWER_MAX_LEN} (drop if exceeds)")
        
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
        
        if self.answer_lengths_before:
            self.stats["avg_answer_length_before"] = sum(self.answer_lengths_before) / len(self.answer_lengths_before)
        if self.answer_lengths_after:
            self.stats["avg_answer_length_after"] = sum(self.answer_lengths_after) / len(self.answer_lengths_after)
        
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
        
        logger.info(f"\n📏 Answer Length Stats:")
        logger.info(f"   Before: {self.stats['avg_answer_length_before']:.1f} chars (avg)")
        logger.info(f"   After:  {self.stats['avg_answer_length_after']:.1f} chars (avg)")
        
        logger.info(f"\n✨ Config:")
        logger.info(f"   Threshold: {self.threshold}")
        logger.info(f"   Coverage min: {self.COVERAGE_MIN * 100}%")
        logger.info(f"   Answer max len: {self.ANSWER_MAX_LEN} (drop if exceeds)")
        logger.info(f"📁 Output: {output_dir}")
        logger.info("="*70)
    
    def _save_metadata(self, output_dir: Path):
        """Save metadata and dropped samples."""
        # Stats JSON
        metadata = {
            "threshold": self.threshold,
            "coverage_min": self.COVERAGE_MIN,
            "answer_max_len": self.ANSWER_MAX_LEN,
            "extraction_strategy": "exact_span_no_expansion",
            "notes": {
                "semantics": "Pure extractive QA (SQuAD-style): answer = exact substring from context",
                "word_expansion": "DISABLED - preserves training signal purity",
                "multi_span_limitation": "SequenceMatcher finds longest contiguous block only. Multi-span answers (A...B) rely on coverage threshold (0.8+)"
            },
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
    parser.add_argument("--threshold", type=float, default=90.0,
                       help="Fuzzy match score threshold (90 recommended)")
    
    args = parser.parse_args()
    
    rebuilder = DatasetRebuilder(threshold=args.threshold)
    rebuilder.rebuild(args.input_dir, args.output_dir)
