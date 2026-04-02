"""
Section 3: RLHF via GRPO (Group Relative Policy Optimization)

After instruction tuning, RLHF further improves the model by optimising a
reward signal rather than simple maximum-likelihood.  Here we use ROUGE-1 as
a proxy reward: the model is rewarded for generating summaries that overlap
with human-written references.

Algorithm: GRPO (Shao et al., 2024 — DeepSeekMath).
For each prompt, G completions are sampled.  The reward of each completion is
normalised relative to the other G-1 completions from the same prompt
(group-relative advantage), then a clipped surrogate objective from PPO is
optimised.  This avoids the need for a separate value/critic network.

Reference: https://arxiv.org/abs/2402.03300

Dataset : CNN/DailyMail   (same subset as Sections 1–2)
Model   : instruction-tuned GPT-2 checkpoint from Section 2
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


# Your implementations

def compute_rouge_reward(
    completions: List[str],
    ground_truths: List[str],
) -> List[float]:
    """Compute ROUGE-1 F1 reward for each (completion, ground_truth) pair.
    (Problem 3.1 — 5 pts)

    Use the `evaluate` library:
        import evaluate
        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=[pred], references=[ref])
        score  = result["rouge1"]   # float in [0, 1]

    Call rouge.compute once per pair (not in batch) so that each pair gets
    an independent scalar reward.

    Args:
        completions:   List of model-generated summary strings.
        ground_truths: Corresponding list of reference summary strings.

    Returns:
        List of float rewards, one per pair, each in [0, 1].
    """
    # YOUR CODE HERE
    raise NotImplementedError


def compute_group_normalized_rewards(
    rewards: List[float],
    group_size: int,
) -> torch.Tensor:
    """Compute group-relative advantages for GRPO.  (Problem 3.2 — 8 pts)

    GRPO groups the rewards for the G completions of a single prompt together
    and normalises within each group:

        advantage_i = (reward_i - mean(group)) / (std(group) + eps)

    where eps = 1e-8 prevents division by zero.

    Steps:
        1. Convert rewards to a FloatTensor of shape (N,), where
           N = len(rewards) should be divisible by group_size.
        2. Reshape to (N // group_size, group_size).
        3. Compute per-group mean and std along dim=1, keeping dims.
        4. Normalise: advantages = (rewards - mean) / (std + eps).
        5. Reshape back to (N,) and return.

    Args:
        rewards:    List of N float rewards (N % group_size == 0).
        group_size: Number of completions per prompt (G in the paper).

    Returns:
        FloatTensor of shape (N,) containing the normalised advantages.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def compute_grpo_clip_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Compute the GRPO clipped surrogate loss.  (Problem 3.3 — 7 pts)

    This is the PPO-clip objective applied per completion:

        ratio          = exp(log_probs - old_log_probs)
        clipped_ratio  = clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        loss           = -mean( min(ratio * advantages,
                                    clipped_ratio * advantages) )

    The negative sign converts the maximisation objective into a minimisation
    loss (standard PyTorch convention).

    Args:
        log_probs:     FloatTensor (N,) — log probs under the current policy.
        old_log_probs: FloatTensor (N,) — log probs under the reference policy
                       (detached; no gradient flows through this tensor).
        advantages:    FloatTensor (N,) — group-normalised advantages.
        clip_epsilon:  Clipping threshold (default 0.2, same as PPO paper).

    Returns:
        Scalar loss tensor.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def run_grpo_training(
    ckpt_path: str,
    train_dataset,
    tokenizer,
    output_dir: str,
) -> list:
    """Set up and run GRPO fine-tuning with trl's GRPOTrainer.  (Problem 3.4 — 5 pts)

    Wire together the three building blocks you implemented above:
      1. Build the reward function with build_grpo_reward_fn().
      2. Create a GRPOConfig with the hyperparameters listed below.
      3. Instantiate a GRPOTrainer and call trainer.train().
      4. Return trainer.state.log_history (a list of dicts, one per logged step).

    Required hyperparameters:
        learning_rate               = 1e-5
        max_steps                   = 200
        per_device_train_batch_size = 4
        num_generations             = 4      # G = 4 completions per prompt
        max_new_tokens              = 60
        num_train_epochs            = 1
        logging_steps               = 10
        report_to                   = "none"
        output_dir                  = output_dir   (use the argument)

    The train_dataset is already formatted (each row has "prompt" and
    "ground_truth" columns) — pass it directly to GRPOTrainer.

    Args:
        ckpt_path:     Path (or HuggingFace model ID) to the model to fine-tune.
                       This should be the instruction-tuned checkpoint from Section 2.
        train_dataset: A HuggingFace Dataset with "prompt" and "ground_truth" columns.
                       Use build_grpo_dataset() to create this from raw CNN/DailyMail data.
        tokenizer:     The tokenizer matching ckpt_path.
        output_dir:    Directory where trl saves checkpoints.

    Returns:
        log_history: List of dicts logged during training
                     (from trainer.state.log_history).
                     Each dict contains keys like "step", "loss", "reward", etc.
    """
    # YOUR CODE HERE
    raise NotImplementedError


# ── Provided helpers ──────────────────────────────────────────────────────────

def get_per_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-token log-probabilities from a CausalLM.

    Runs a forward pass and returns log_softmax over the vocabulary for each
    token position, then gathers the log-prob of the actual next token.

    Args:
        model:          A HuggingFace CausalLM.
        input_ids:      LongTensor of shape (B, T).
        attention_mask: LongTensor of shape (B, T).

    Returns:
        FloatTensor of shape (B, T-1) — log P(token_t | token_{<t}) for each t.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, vocab)
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]          # (B, T-1, vocab)
    shift_labels = input_ids[:, 1:]           # (B, T-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log-prob of actual tokens
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)                              # (B, T-1)
    return token_log_probs


def build_grpo_reward_fn(ground_truths: List[str]):
    """Build a reward function compatible with trl's GRPOTrainer.

    trl expects the reward function to accept keyword arguments:
        prompts, completions, **kwargs
    and return a list of float tensors.  Any extra columns in the dataset are
    passed through **kwargs; we retrieve ground_truth from there.

    Args:
        ground_truths: Not used here — the ground_truth column is passed via
                       **kwargs by trl when the dataset has a "ground_truth"
                       column.

    Returns:
        A reward function compatible with GRPOTrainer.
    """
    def reward_fn(prompts, completions, ground_truth=None, **kwargs):
        if ground_truth is None:
            raise ValueError(
                "Dataset must contain a 'ground_truth' column. "
                "See build_grpo_dataset() for how to add it."
            )
        scores = compute_rouge_reward(completions, ground_truth)
        return [torch.tensor(s, dtype=torch.float32) for s in scores]
    return reward_fn


def build_grpo_dataset(raw_data, tokenizer, max_article_tokens: int = 400):
    """Convert a CNN/DailyMail split into a format suitable for GRPOTrainer.

    Returns a list of dicts, each with:
        "prompt"        — the formatted instruction prompt (no summary)
        "ground_truth"  — the reference summary string

    trl's GRPOTrainer will tokenise "prompt" and pass "ground_truth" to the
    reward function via **kwargs.
    """
    from src.instruction_tuning import INSTRUCTION_TEMPLATE
    records = []
    for item in raw_data:
        # Truncate article so the prompt fits in the model's context window
        article_ids = tokenizer.encode(item["article"])[:max_article_tokens]
        article     = tokenizer.decode(article_ids, skip_special_tokens=True)
        prompt      = INSTRUCTION_TEMPLATE.format(article=article)
        records.append({
            "prompt":       prompt,
            "ground_truth": item["highlights"],
        })
    return records
