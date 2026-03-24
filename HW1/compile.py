#!/usr/bin/env python
"""
HW1 Compile Script - Test and Build Notebook Outputs

This script validates your code and generates cache files so the notebook
can run on CPU without GPU.

Usage:
  python compile.py [section]

  section:
    section1  - Compile Section 1 (Probing) only
    section2  - Compile Section 2 (Logit Lens) only  
    section4  - Compile Section 4 (KV Cache) only
    all       - Compile all sections (default)

Examples:
  # Compile just Section 1 after implementing probing.py:
  srun --partition general --gres gpu:a100:1 --time 0:20:00 python compile.py section1

  # Compile all sections at once:
  srun --partition general --gres gpu:a100:1 --time 1:00:00 python compile.py all

Note: The autograder always runs first to validate your code.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import json
from pathlib import Path
import time
import subprocess
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Compile HW1 sections')
parser.add_argument('section', nargs='?', default='all',
                    choices=['section1', 'section2', 'section4', 'all'],
                    help='Which section to compile (default: all)')
args = parser.parse_args()

print("="*80)
print(f"HW1 COMPILE - {args.section.upper()}")
print("="*80)

# ============================================================================
# STEP 0: Run Autograder
# ============================================================================
print("\nSTEP 0: Running Autograder")
print("-"*80)

try:
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode == 0:
        passed = result.stdout.count(' PASSED')
        print(f"✓ All {passed} tests passed!")
    else:
        print("✗ Some tests failed:")
        print(result.stdout[-1000:])
        print("\nPlease fix failing tests before generating cache files.")
        sys.exit(1)

except Exception as e:
    print(f"⚠️  Could not run autograder: {e}")
    print("Continuing with cache generation...")

print()
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
print(f"Cache directory: {CACHE_DIR}\n")

# ============================================================================
# SECTION 1: Probing
# ============================================================================
if args.section in ['section1', 'all']:
    print("="*80)
    print("STEP 1: Compiling Section 1 Outputs (Probing)")
    print("="*80)

    PROBING_CACHE = CACHE_DIR / 'probing_results.json'

    if PROBING_CACHE.exists():
        print("✓ cache/probing_results.json already exists - SKIPPING")
    else:
        print("\nRunning Section 1 (may take 5-10 minutes)...")
        start = time.time()
        
        from src.probing import (get_model_and_tokenizer, get_data, build_classifier,
                                 get_sentence_repr, train_probe, evaluate_probe)
        import numpy as np
        from tqdm import tqdm
        
        torch.manual_seed(42)
        train_data, test_data = get_data(seed=42)
        model_p, tokenizer_p, emb_dim = get_model_and_tokenizer('EleutherAI/pythia-160m', device)
        
        print('Extracting representations...')
        train_repr = [get_sentence_repr(s, model_p, tokenizer_p, device) 
                      for s in tqdm(train_data['sentence'], desc='Train set')]
        test_repr  = [get_sentence_repr(s, model_p, tokenizer_p, device) 
                      for s in tqdm(test_data['sentence'], desc='Test set')]
        
        n_layers             = train_repr[0].shape[0]
        final_idx, third_idx = n_layers - 1, 3
        
        def mean_pool(lst, idx):
            return [torch.tensor(np.mean(r[idx], 0), dtype=torch.float32).to(device) for r in lst]
        
        lbl_tr = [torch.tensor(int(x), dtype=torch.long).to(device) for x in train_data['label']]
        lbl_te = [torch.tensor(int(x), dtype=torch.long).to(device) for x in test_data['label']]
        
        print('\nTraining final-layer probe...')
        clf_f, crit_f, opt_f = build_classifier(emb_dim, 2, str(device))
        trL_f, trA_f = train_probe(mean_pool(train_repr, final_idx), lbl_tr, clf_f, crit_f, opt_f)
        teL_f, teA_f = evaluate_probe(mean_pool(test_repr, final_idx), lbl_te, clf_f, crit_f)
        
        print('\nTraining third-layer probe...')
        clf_t, crit_t, opt_t = build_classifier(emb_dim, 2, str(device))
        trL_t, trA_t = train_probe(mean_pool(train_repr, third_idx), lbl_tr, clf_t, crit_t, opt_t)
        teL_t, teA_t = evaluate_probe(mean_pool(test_repr, third_idx), lbl_te, clf_t, crit_t)
        
        probing_results = {
            'final_layer_idx': final_idx,
            'final': {'train_acc': trA_f, 'test_acc': teA_f, 'train_loss': trL_f, 'test_loss': teL_f},
            'third': {'train_acc': trA_t, 'test_acc': teA_t, 'train_loss': trL_t, 'test_loss': teL_t},
        }
        
        with open(PROBING_CACHE, 'w') as f:
            json.dump(probing_results, f, indent=2)
        
        elapsed = time.time() - start
        print(f'\n✓ Saved to {PROBING_CACHE} ({elapsed:.1f}s)')
        
        del model_p, tokenizer_p, clf_f, clf_t
        torch.cuda.empty_cache()

# ============================================================================
# SECTION 2: Logit Lens  
# ============================================================================
if args.section in ['section2', 'all']:
    print("\n" + "="*80)
    print("STEP 2: Compiling Section 2 Outputs (Logit Lens)")
    print("="*80)

    LOGIT_CACHE = CACHE_DIR / 'logit_lens_results.pt'

    if LOGIT_CACHE.exists():
        print("✓ cache/logit_lens_results.pt already exists - SKIPPING")
    else:
        print("\nRunning Section 2 (may take 1-2 minutes)...")
        start = time.time()
        
        from src.logit_lens import get_logit_lens_predictions, get_token_rank_by_layer
        from nnsight import LanguageModel
        
        PROMPT_A = 'The currency in the United States of America is the dollar.'
        PROMPT_B = 'The result of two plus two is four.'
        
        print('Loading pythia-410m with nnsight...')
        lm = LanguageModel('EleutherAI/pythia-410m', device_map='auto', dispatch=True)
        
        def encode_single(s, tok):
            for cand in [s, ' ' + s]:
                ids = tok.encode(cand)
                if len(ids) == 1: return ids[0]
            return tok.encode(' ' + s)[-1]
        
        print('Running logit lens on prompt A...')
        mp_a, tt_a, iw_a, probs_a = get_logit_lens_predictions(PROMPT_A, lm)
        
        print('Running logit lens on prompt B...')
        mp_b, tt_b, iw_b, probs_b = get_logit_lens_predictions(PROMPT_B, lm)
        
        id_dollar = encode_single('dollar', lm.tokenizer)
        id_four   = encode_single('four',   lm.tokenizer)
        ranks_a   = get_token_rank_by_layer(probs_a, id_dollar)
        ranks_b   = get_token_rank_by_layer(probs_b, id_four)
        
        torch.save({
            'mp_a': mp_a.cpu(), 'tt_a': tt_a, 'iw_a': iw_a,
            'mp_b': mp_b.cpu(), 'tt_b': tt_b, 'iw_b': iw_b,
            'ranks_a': ranks_a, 'ranks_b': ranks_b,
            'id_dollar': id_dollar, 'id_four': id_four,
        }, LOGIT_CACHE)
        
        elapsed = time.time() - start
        print(f'\n✓ Saved to {LOGIT_CACHE} ({elapsed:.1f}s)')
        
        del lm
        torch.cuda.empty_cache()

# ============================================================================
# SECTION 4: KV Cache
# ============================================================================
if args.section in ['section4', 'all']:
    print("\n" + "="*80)
    print("STEP 3: Compiling Section 4 Outputs (KV Cache)")
    print("="*80)

    KV_BENCH_CACHE = CACHE_DIR / 'kv_cache_benchmark.json'

    if KV_BENCH_CACHE.exists():
        print("✓ cache/kv_cache_benchmark.json already exists - SKIPPING")
    else:
        print("\nRunning Section 4 (may take 2-3 minutes)...")
        start = time.time()
        
        from src.kv_cache import MultiLayerKVCache, generate_reference, generate_with_budget, benchmark
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print('Loading pythia-160m...')
        model_kv     = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m').to(device)
        tokenizer_kv = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
        model_kv.eval()
        
        L = model_kv.config.num_hidden_layers
        FULL_BUDGET = L * 256
        
        budgets = [
            FULL_BUDGET // 4,
            FULL_BUDGET // 2,
            FULL_BUDGET * 3 // 4,
            FULL_BUDGET
        ]
        
        # Load ranks from Section 2 cache
        LOGIT_CACHE = CACHE_DIR / 'logit_lens_results.pt'
        if LOGIT_CACHE.exists():
            data = torch.load(LOGIT_CACHE, map_location='cpu', weights_only=False)
            ranks_a = data['ranks_a']
        else:
            print('⚠️  Section 2 cache not found, using dummy layer_scores')
            ranks_a = list(range(24, 0, -1))
        
        layer_scores = ranks_a[:L] if len(ranks_a) >= L else ranks_a
        
        GEN_PROMPT = ('The transformer architecture was introduced in "Attention is All You Need". '
                      'It uses self-attention to process sequences in parallel. Since then,')
        MAX_NEW_TOKS = 50
        
        print('Running benchmark...')
        raw = benchmark(
            model_kv, tokenizer_kv, GEN_PROMPT,
            total_budgets=budgets,
            layer_scores=layer_scores,
            max_new_tokens=MAX_NEW_TOKS,
        )
        
        kv_bench = {str(b): v for b, v in raw.items()}
        
        with open(KV_BENCH_CACHE, 'w') as f:
            json.dump(kv_bench, f, indent=2)
        
        elapsed = time.time() - start
        print(f'\n✓ Saved to {KV_BENCH_CACHE} ({elapsed:.1f}s)')

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
if args.section == 'all':
    print("COMPILE COMPLETE - All Outputs Ready")
else:
    print(f"COMPILE COMPLETE - {args.section.upper()} Ready")
print("="*80)
print()

# Check which cache files exist
PROBING_CACHE = CACHE_DIR / 'probing_results.json'
LOGIT_CACHE = CACHE_DIR / 'logit_lens_results.pt'
KV_BENCH_CACHE = CACHE_DIR / 'kv_cache_benchmark.json'

print("Cache files:")
for cache_file in [PROBING_CACHE, LOGIT_CACHE, KV_BENCH_CACHE]:
    if cache_file.exists():
        size = cache_file.stat().st_size
        print(f"  ✓ {cache_file.name} ({size:,} bytes)")
    else:
        print(f"  ⏳ {cache_file.name} (not created yet)")

print()
if all(f.exists() for f in [PROBING_CACHE, LOGIT_CACHE, KV_BENCH_CACHE]):
    print("✓ All cache files ready! Notebook can run on CPU.")
else:
    missing = [f.name for f in [PROBING_CACHE, LOGIT_CACHE, KV_BENCH_CACHE] if not f.exists()]
    print(f"⏳ Still need: {', '.join(missing)}")
    print("   Run compile.py with the corresponding section arguments.")
print("="*80)
