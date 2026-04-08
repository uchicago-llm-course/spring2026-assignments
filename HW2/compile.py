#!/usr/bin/env python
"""
HW2 Compile Script — Train Models and Save Notebook Outputs

This script validates your code and trains/evaluates the models for each
section, saving the results to cache/ so the notebook can run on CPU.

Usage:
  python compile.py [section]

  section:
    section1  — SFT training            → cache/sft_results.json
    section2  — Instruction Tuning      → cache/it_results.json
    section3  — RLHF (GRPO) training    → cache/rlhf_results.json
    all       — All three sections (default)

Examples:
  srun --partition general --gres gpu:a100:1 --time 1:00:00 --mem 32G \\
       python compile.py section1

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

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Compile HW2 sections')
parser.add_argument('section', nargs='?', default='all',
                    choices=['section1', 'section2', 'section3', 'all'],
                    help='Which section to compile (default: all)')
args = parser.parse_args()

print("=" * 80)
print(f"HW2 COMPILE — {args.section.upper()}")
print("=" * 80)

# ── Autograder ────────────────────────────────────────────────────────────────
print("\nSTEP 0: Running Autograder")
print("-" * 80)

try:
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
        capture_output=True, text=True, timeout=180
    )
    if result.returncode == 0:
        passed = result.stdout.count(' PASSED')
        print(f"✓ All {passed} tests passed!")
    else:
        print("✗ Some tests failed:")
        print(result.stdout[-2000:])
        print("\nPlease fix failing tests before generating cache files.")
        sys.exit(1)
except Exception as e:
    print(f"⚠️  Could not run autograder: {e}")
    print("Continuing with cache generation...")

print()
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
print()

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
print(f"Cache directory: {CACHE_DIR}\n")

# ── Shared training config ────────────────────────────────────────────────────
MODEL_NAME  = "gpt2"
NUM_TRAIN   = 5000
NUM_VAL     = 500
MAX_LENGTH  = 256    # token budget per example (keep training fast)
BATCH_SIZE  = 8
LR          = 5e-5
NUM_EPOCHS  = 5      # Increased for smoother training curves
LOG_INTERVAL = 100   # Log every N batches for detailed curves

# ============================================================================
# SECTION 1: SFT
# ============================================================================
if args.section in ['section1', 'all']:
    print("=" * 80)
    print("STEP 1: Compiling Section 1 Outputs (SFT)")
    print("=" * 80)

    SFT_CACHE = CACHE_DIR / 'sft_results.json'

    if SFT_CACHE.exists():
        print("✓ cache/sft_results.json already exists — SKIPPING")
    else:
        print("\nRunning Section 1 (may take 20–30 minutes)...")
        start = time.time()

        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from src.sft import (
            get_data, get_model_and_tokenizer, SFTDataset,
            train_epoch, evaluate_rouge
        )

        train_data, val_data = get_data(num_train=NUM_TRAIN, num_val=NUM_VAL)
        model, tokenizer = get_model_and_tokenizer(MODEL_NAME, str(device))

        train_ds = SFTDataset(train_data, tokenizer, max_length=MAX_LENGTH)
        val_ds   = SFTDataset(val_data,   tokenizer, max_length=MAX_LENGTH)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        optimizer = AdamW(model.parameters(), lr=LR)

        sft_results = {
            'model_name': MODEL_NAME,
            'train_steps': [],      # Global step numbers
            'train_losses': [],     # Loss at each logged step
            'epoch_losses': [],     # Mean loss per epoch (for backward compatibility)
            'epoch_rouge': [],      # ROUGE per epoch
            'global_step': 0,       # Track global training step
        }

        batches_per_epoch = len(train_loader)

        def log_fn(batch_in_epoch, loss):
            sft_results['global_step'] += LOG_INTERVAL
            sft_results['train_steps'].append(sft_results['global_step'])
            sft_results['train_losses'].append(loss)

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            train_loss = train_epoch(model, train_loader, optimizer, device,
                                    log_callback=log_fn, log_interval=LOG_INTERVAL)
            sft_results['epoch_losses'].append(train_loss)
            print(f"  Train loss : {train_loss:.4f}")

            rouge = evaluate_rouge(
                model, tokenizer,
                val_data['article'], val_data['highlights'],
                device, max_new_tokens=80, num_examples=100
            )
            sft_results['epoch_rouge'].append(rouge['rouge1'])
            print(f"  Val ROUGE-1: {rouge['rouge1']:.4f}")

        # Save some qualitative examples
        rouge_final = evaluate_rouge(
            model, tokenizer,
            val_data['article'], val_data['highlights'],
            device, max_new_tokens=80, num_examples=5
        )
        sft_results['examples'] = [
            {'article': val_data[i]['article'][:500],
             'reference': rouge_final['references'][i],
             'prediction': rouge_final['predictions'][i]}
            for i in range(5)
        ]

        # Save model checkpoint
        SFT_CKPT_DIR = CACHE_DIR / 'sft_checkpoint'
        model.save_pretrained(SFT_CKPT_DIR)
        tokenizer.save_pretrained(SFT_CKPT_DIR)
        sft_results['checkpoint'] = str(SFT_CKPT_DIR)

        # Remove internal tracking variable before saving
        sft_results.pop('global_step', None)

        with open(SFT_CACHE, 'w') as f:
            json.dump(sft_results, f, indent=2)

        elapsed = time.time() - start
        print(f'\n✓ Saved to {SFT_CACHE} ({elapsed:.1f}s)')

        del model
        torch.cuda.empty_cache()

# ============================================================================
# SECTION 2: Instruction Tuning
# ============================================================================
if args.section in ['section2', 'all']:
    print("\n" + "=" * 80)
    print("STEP 2: Compiling Section 2 Outputs (Instruction Tuning)")
    print("=" * 80)

    IT_CACHE = CACHE_DIR / 'it_results.json'

    if IT_CACHE.exists():
        print("✓ cache/it_results.json already exists — SKIPPING")
    else:
        print("\nRunning Section 2 (may take 20–30 minutes)...")
        start = time.time()

        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from src.sft import (
            get_data, get_model_and_tokenizer, evaluate_rouge
        )
        from src.instruction_tuning import (
            ITDataset, format_instruction, INSTRUCTION_TEMPLATE
        )
        from src.sft import compute_sft_loss
        import torch.nn as nn

        train_data, val_data = get_data(num_train=NUM_TRAIN, num_val=NUM_VAL)
        model, tokenizer = get_model_and_tokenizer(MODEL_NAME, str(device))

        train_ds = ITDataset(train_data, tokenizer, max_length=MAX_LENGTH)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=LR)

        it_results = {
            'model_name': MODEL_NAME,
            'train_steps': [],      # Global step numbers
            'train_losses': [],     # Loss at each logged step
            'epoch_losses': [],     # Mean loss per epoch
            'epoch_rouge': [],      # ROUGE per epoch
            'global_step': 0,       # Track global training step
        }

        def it_train_epoch(model, loader, opt, log_fn=None):
            model.train()
            total = 0.0
            for batch_idx, batch in enumerate(loader):
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbl  = batch['labels'].to(device)
                opt.zero_grad()
                loss = compute_sft_loss(model, ids, mask, lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()

                # Periodic logging
                if log_fn is not None and (batch_idx + 1) % LOG_INTERVAL == 0:
                    log_fn(batch_idx + 1, loss.item())
            return total / len(loader)

        def log_fn(batch_in_epoch, loss):
            it_results['global_step'] += LOG_INTERVAL
            it_results['train_steps'].append(it_results['global_step'])
            it_results['train_losses'].append(loss)

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            train_loss = it_train_epoch(model, train_loader, optimizer, log_fn=log_fn)
            it_results['epoch_losses'].append(train_loss)
            print(f"  Train loss : {train_loss:.4f}")

            # Evaluate: prompt with instruction template, no summary
            formatted_val_articles = [format_instruction(art) for art in val_data['article']]

            rouge = evaluate_rouge(
                model, tokenizer,
                formatted_val_articles, val_data['highlights'],
                device, max_new_tokens=80, num_examples=100
            )
            it_results['epoch_rouge'].append(rouge['rouge1'])
            print(f"  Val ROUGE-1: {rouge['rouge1']:.4f}")

        # Save examples + checkpoint
        rouge_final = evaluate_rouge(
            model, tokenizer,
            val_data['article'], val_data['highlights'],
            device, max_new_tokens=80, num_examples=5
        )
        it_results['examples'] = [
            {'article': val_data[i]['article'][:500],
             'reference': rouge_final['references'][i],
             'prediction': rouge_final['predictions'][i]}
            for i in range(5)
        ]

        IT_CKPT_DIR = CACHE_DIR / 'it_checkpoint'
        model.save_pretrained(IT_CKPT_DIR)
        tokenizer.save_pretrained(IT_CKPT_DIR)
        it_results['checkpoint'] = str(IT_CKPT_DIR)

        # Remove internal tracking variable before saving
        it_results.pop('global_step', None)

        with open(IT_CACHE, 'w') as f:
            json.dump(it_results, f, indent=2)

        elapsed = time.time() - start
        print(f'\n✓ Saved to {IT_CACHE} ({elapsed:.1f}s)')

        del model
        torch.cuda.empty_cache()

# ============================================================================
# SECTION 3: RLHF via GRPO
# ============================================================================
if args.section in ['section3', 'all']:
    print("\n" + "=" * 80)
    print("STEP 3: Compiling Section 3 Outputs (RLHF — GRPO)")
    print("=" * 80)

    RLHF_CACHE = CACHE_DIR / 'rlhf_results.json'

    if RLHF_CACHE.exists():
        print("✓ cache/rlhf_results.json already exists — SKIPPING")
    else:
        print("\nRunning Section 3 (may take 20–30 minutes)...")
        start = time.time()

        from src.sft import get_data, get_model_and_tokenizer, evaluate_rouge
        from src.rlhf import run_grpo_training, build_grpo_dataset
        from datasets import Dataset as HFDataset

        train_data, val_data = get_data(num_train=NUM_TRAIN, num_val=NUM_VAL)

        # Load the instruction-tuned checkpoint (or base model if not available)
        IT_CKPT_DIR = CACHE_DIR / 'it_checkpoint'
        if IT_CKPT_DIR.exists():
            print(f"Loading IT checkpoint from {IT_CKPT_DIR}")
            ckpt_path = str(IT_CKPT_DIR)
        else:
            print("⚠️  IT checkpoint not found; using base GPT-2 (run section2 first for best results)")
            ckpt_path = MODEL_NAME

        _, tokenizer = get_model_and_tokenizer(ckpt_path, 'cpu')

        # Build dataset for GRPOTrainer
        grpo_records = build_grpo_dataset(train_data, tokenizer, max_article_tokens=200)
        grpo_dataset = HFDataset.from_list(grpo_records)

        print("Starting GRPO training (calls your run_grpo_training implementation)...")
        log_history = run_grpo_training(
            ckpt_path   = ckpt_path,
            train_dataset = grpo_dataset,
            tokenizer   = tokenizer,
            output_dir  = str(CACHE_DIR / "grpo_output"),
        )

        steps   = [x.get("step", i) for i, x in enumerate(log_history) if "loss"   in x]
        losses  = [x["loss"]          for x   in log_history             if "loss"   in x]
        rsteps  = [x.get("step", i) for i, x in enumerate(log_history) if "reward" in x]
        rewards = [x["reward"]        for x   in log_history             if "reward" in x]

        # Reload the saved checkpoint to evaluate ROUGE
        from transformers import AutoModelForCausalLM
        grpo_out_dir = CACHE_DIR / "grpo_output"
        # GRPOTrainer saves to checkpoint-{max_steps} subdirectory
        checkpoint_dir = grpo_out_dir / "checkpoint-200"
        if not checkpoint_dir.exists():
            # Fallback to grpo_output if no checkpoint subdirs
            checkpoint_dir = grpo_out_dir
        rlhf_model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir)).to(device)
        rlhf_model.eval()

        rouge = evaluate_rouge(
            rlhf_model, tokenizer,
            val_data['article'], val_data['highlights'],
            device if torch.cuda.is_available() else torch.device('cpu'),
            max_new_tokens=80, num_examples=100
        )

        # Load SFT and IT ROUGE for comparison
        sft_rouge = None
        it_rouge  = None
        if (CACHE_DIR / 'sft_results.json').exists():
            with open(CACHE_DIR / 'sft_results.json') as f:
                sft_data = json.load(f)
            sft_rouge = (sft_data.get('epoch_rouge') or sft_data.get('val_rouge', []))[-1] if sft_data.get('epoch_rouge') or sft_data.get('val_rouge') else None
        if (CACHE_DIR / 'it_results.json').exists():
            with open(CACHE_DIR / 'it_results.json') as f:
                it_data = json.load(f)
            it_rouge = (it_data.get('epoch_rouge') or it_data.get('val_rouge', []))[-1] if it_data.get('epoch_rouge') or it_data.get('val_rouge') else None

        rlhf_results = {
            'model_name'   : ckpt_path,
            'train_steps'  : steps,
            'train_losses' : losses,
            'reward_steps' : rsteps,
            'rewards'      : rewards,
            'val_rouge'    : rouge['rouge1'],
            'comparison'   : {
                'sft_rouge1'  : sft_rouge,
                'it_rouge1'   : it_rouge,
                'rlhf_rouge1' : rouge['rouge1'],
            }
        }

        with open(RLHF_CACHE, 'w') as f:
            json.dump(rlhf_results, f, indent=2)

        elapsed = time.time() - start
        print(f'\n✓ Saved to {RLHF_CACHE} ({elapsed:.1f}s)')

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
if args.section == 'all':
    print("COMPILE COMPLETE — All Outputs Ready")
else:
    print(f"COMPILE COMPLETE — {args.section.upper()} Ready")
print("=" * 80)
print()

SFT_CACHE  = CACHE_DIR / 'sft_results.json'
IT_CACHE   = CACHE_DIR / 'it_results.json'
RLHF_CACHE = CACHE_DIR / 'rlhf_results.json'

print("Cache files:")
for cache_file in [SFT_CACHE, IT_CACHE, RLHF_CACHE]:
    if cache_file.exists():
        size = cache_file.stat().st_size
        print(f"  ✓ {cache_file.name} ({size:,} bytes)")
    else:
        print(f"  ⏳ {cache_file.name} (not created yet)")

print()
if all(f.exists() for f in [SFT_CACHE, IT_CACHE, RLHF_CACHE]):
    print("✓ All cache files ready! Notebook can run on CPU.")
else:
    missing = [f.name for f in [SFT_CACHE, IT_CACHE, RLHF_CACHE] if not f.exists()]
    print(f"⏳ Still need: {', '.join(missing)}")
    print("   Run compile.py with the corresponding section argument.")
print("=" * 80)
