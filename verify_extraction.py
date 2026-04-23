import json

print("="*70)
print("VERIFICATION: Exact span extraction + _answer_start indices")
print("="*70)

errors = []
total_checked = 0

with open('data/examples_ai_flashcard_fixed/train_fixed.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:  # Check first 10
            break
        data = json.loads(line)
        total_checked += 1
        context = data['context']
        answer = data['answer']
        start_idx = data.get('_answer_start')
        match_type = data.get('_match_type')
        match_score = data.get('_match_score')
        
        # Verify substring
        if answer not in context:
            errors.append(f"Sample {i+1}: Answer NOT substring!")
            continue
        
        # Verify _answer_start index
        if start_idx is not None:
            expected_answer = context[start_idx:start_idx + len(answer)]
            if expected_answer != answer:
                errors.append(f"Sample {i+1}: _answer_start index mismatch")
                errors.append(f"  Expected at idx {start_idx}: {repr(expected_answer)}")
                errors.append(f"  Got: {repr(answer)}")
                continue
        
        print(f"\n✅ Sample {i+1} ({match_type}, score={match_score}):")
        print(f"   Context: {repr(context[:60])}")
        print(f"   Answer:  {repr(answer)}")
        print(f"   Index:   {start_idx} (verified ✓)")

print(f"\n{'='*70}")
print(f"Checked: {total_checked} samples")
if errors:
    print(f"❌ ERRORS FOUND:")
    for e in errors:
        print(f"  {e}")
else:
    print(f"✅ ALL CORRECT: Exact span + valid _answer_start indices")
print(f"{'='*70}")

print("\nMetadata check:")
with open('data/examples_ai_flashcard_fixed/rebuild_metadata.json', encoding='utf-8') as f:
    meta = json.load(f)
    print(f"Extraction strategy: {meta['extraction_strategy']}")
    if 'notes' in meta:
        for k, v in meta['notes'].items():
            print(f"  {k}: {v}")
