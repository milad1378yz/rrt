"""Generate base-policy responses locally and judge every rubric criterion.

This intentionally uses a plain in-process Transformers model. Distributed inference and cluster orchestration are deployment details rather than part of RRT.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.cache import (
    coerce_messages,
    extract_rubrics,
    judged_example,
    load_splits,
    prompt_text,
    save_cache,
)
from core.judge import JudgeConfig, RubricJudge, normalized_score


logger = logging.getLogger(__name__)


def render_prompt(tokenizer, row, enable_thinking=True):
    """Apply the policy's native chat template to one dataset prompt."""

    return tokenizer.apply_chat_template(
        coerce_messages(row["prompt"]),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def load_policy(args):
    """Load the local policy once for every requested split."""

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model, device


def generate_responses(rows, args, policy=None):
    """Generate ``args.rollouts`` responses per prompt with a local model."""

    tokenizer, model, device = policy or load_policy(args)

    all_responses = []
    for row_index, row in enumerate(tqdm(rows, desc="generating prompts")):
        text = render_prompt(tokenizer, row, args.enable_thinking)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_tokens,
        ).to(device)
        prompt_length = inputs["input_ids"].shape[1]
        responses = []
        for rollout_index in range(args.rollouts):
            torch.manual_seed(args.seed + row_index * args.rollouts + rollout_index)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_response_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            responses.append(tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True))
        all_responses.append(responses)
    return all_responses


def judge_responses(rows, responses, config):
    """Judge all rollouts with one bounded worker pool."""

    judge = RubricJudge(config)
    rubrics = [extract_rubrics(row) for row in rows]
    prompts = [prompt_text(row) for row in rows]
    presence = [[None] * len(responses[row]) for row in range(len(rows))]

    tasks = []
    for row_index, row_responses in enumerate(responses):
        for rollout_index, response in enumerate(row_responses):
            tasks.append(
                (
                    row_index,
                    rollout_index,
                    prompts[row_index],
                    response,
                    rubrics[row_index],
                )
            )

    def grade(task):
        row_index, rollout_index, prompt, response, rollout_rubrics = task
        try:
            verdicts = [
                judge.grade_criterion(prompt, response, rubric) for rubric in rollout_rubrics
            ]
        except Exception as error:
            logger.warning("judge call failed: %s", error)
            verdicts = None
        return row_index, rollout_index, verdicts

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        results = pool.map(grade, tasks)
        for row_index, rollout_index, verdicts in tqdm(
            results, total=len(tasks), desc="judging rollouts"
        ):
            presence[row_index][rollout_index] = verdicts
    return [judged_example(row, verdicts) for row, verdicts in zip(rows, presence)]


def cache_summary(cache):
    """Small integrity and score summary for a judged cache."""

    summaries = {}
    for split, examples in cache.items():
        scores = []
        missing = 0
        for example in examples:
            rubrics = [
                {"criterion": criterion["text"], "points": criterion["points"]}
                for criterion in example["criteria"]
            ]
            for verdicts in example["presence"]:
                if verdicts is None:
                    missing += 1
                else:
                    scores.append(normalized_score(rubrics, verdicts))
        summaries[split] = {
            "prompts": len(examples),
            "graded_rollouts": len(scores),
            "missing_rollouts": missing,
            "mean_normalized_score": float(np.mean(scores)) if scores else None,
        }
    return summaries


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--output", required=True, help="Output directory or rollout_cache.pkl path."
    )
    parser.add_argument("--model", required=True, help="Local Hugging Face policy checkpoint.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-prompt-tokens", type=int, default=6144)
    parser.add_argument("--max-response-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="Zero disables top-k sampling.")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    defaults = JudgeConfig()
    parser.add_argument("--judge-model", default=defaults.model)
    parser.add_argument("--judge-timeout-seconds", type=float, default=defaults.timeout_seconds)
    parser.add_argument("--judge-retries", type=int, default=defaults.retries)
    parser.add_argument("--judge-concurrency", type=int, default=defaults.concurrency)
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if min(args.rollouts, args.max_prompt_tokens, args.max_response_tokens) <= 0:
        raise SystemExit("rollouts and token limits must be positive")
    if args.judge_concurrency <= 0 or args.judge_retries < 0:
        raise SystemExit("judge concurrency must be positive and retries non-negative")
    rows_by_split = load_splits(args.data_dir, args.splits, args.limit)
    judge_config = JudgeConfig(
        model=args.judge_model,
        timeout_seconds=args.judge_timeout_seconds,
        retries=args.judge_retries,
        concurrency=args.judge_concurrency,
    )
    policy = load_policy(args)
    cache = {}
    for split, rows in rows_by_split.items():
        logger.info("building %s cache for %d prompts", split, len(rows))
        responses = generate_responses(rows, args, policy)
        cache[split] = judge_responses(rows, responses, judge_config)

    output = Path(args.output)
    cache_path = output if output.suffix == ".pkl" else output / "rollout_cache.pkl"
    save_cache(cache, cache_path)
    summary = cache_summary(cache)
    summary_path = cache_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("wrote %s", cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
