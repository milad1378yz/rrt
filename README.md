# Rubric Rewards from Item Response Theory

Reference implementation of **Rubric Response Theory (RRT)**. This repository provides
dataset preparation, rollout judging, Response Parameter Network (RPN) fitting, RRT
rewards, adaptive criterion selection, and held-out evaluation.

A policy trainer is not included. Integrate `reward.py` with your existing GRPO or PPO
training loop.

## Install

```bash
conda create --name rrt python=3.10 -y
conda activate rrt
python -m pip install -e ".[pipeline,dev]"
export OPENAI_API_KEY="your-api-key"
```

Set `JUDGE_MODEL` if you want to override the default judge model. A CUDA GPU is
recommended for rollout generation and RPN fitting.

## Run the pipeline

The commands below use RubricHub Science as the example.

### 1. Prepare data

Choose one converter:

```bash
# RubricHub Medical
python -m data_prep.convert_rubrichub --domain Medical

# RubricHub Science
python -m data_prep.convert_rubrichub --domain Science

# Rubrics as Rewards Science
python -m data_prep.convert_rar_science

# RubricBench
python -m data_prep.convert_rubricbench
```

Use `--out-dir /path/to/output` to change a converter's output directory.

### 2. Generate and judge rollouts

```bash
python build_rollout_cache.py \
  --data-dir data/datasets/rubrichub_science_irt \
  --output data/caches/science_base \
  --model /path/to/base-policy-checkpoint
```

This command uses the policy checkpoint's native chat template and calls the configured
judge for each rubric criterion. Use `python build_rollout_cache.py --help` to change
generation, judging, split, or device options.

### 3. Fit the RPN

```bash
python fit_rpn.py \
  --cache data/caches/science_base \
  --output data/rpn/science
```

The fitted checkpoint is written to `data/rpn/science/rpn.pt`. Use
`python fit_rpn.py --help` to select another embedding model or training configuration.

### 4. Add RRT to policy training

Collect all rollout records for a policy step, train the policy with their rewards, and
then update and save the RPN:

```python
from reward import RRTReward

rrt = RRTReward("data/rpn/science/rpn.pt", online=True)

records = [
    rrt.score_from_verdicts(
        prompt,
        rubrics,
        presence,
        response_tokens=response_tokens,
    )
    for prompt, rubrics, presence, response_tokens in policy_step_rollouts
]

run_policy_update([record.reward for record in records])
rrt.update(records)
rrt.save("checkpoints/step_001/rpn.pt")
```

Each `rubrics` value is a list of dictionaries with `criterion` and `points` fields.
Each `presence` value is the aligned Boolean verdict vector. Replace
`policy_step_rollouts` and `run_policy_update` with the corresponding values and call
from your trainer.

To let this package judge a response, pass a `RubricJudge` to `RRTReward` and call
`score_response`.

### 5. Use adaptive criterion selection

Adaptive selection uses a frozen RPN:

```python
from core.judge import RubricJudge
from reward import RRTReward

rrt = RRTReward(
    "data/rpn/science/rpn.pt",
    online=False,
    judge=RubricJudge(),
)

records, selected = rrt.score_group_adaptive(
    prompt,
    responses,
    rubrics,
    budget=criterion_budget,
    response_tokens=response_token_counts,
)
```

### 6. Evaluate policies

First build a test cache for each policy:

```bash
python build_rollout_cache.py \
  --data-dir data/datasets/rubrichub_science_irt \
  --output data/caches/science_base_test \
  --model /path/to/base-policy-checkpoint \
  --splits test

python build_rollout_cache.py \
  --data-dir data/datasets/rubrichub_science_irt \
  --output data/caches/science_rrt_test \
  --model /path/to/rrt-policy-checkpoint \
  --splits test
```

Then compare the judged caches:

```bash
python evaluate.py \
  --data data/datasets/rubrichub_science_irt/test.parquet \
  --reference base \
  --output results/science.json \
  base=data/caches/science_base_test \
  rrt=data/caches/science_rrt_test
```

Use `python evaluate.py --help` for evaluation options.

## Test

```bash
python -m pytest -q
```
