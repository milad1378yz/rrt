# Rubric Response Theory

Reference implementation of **Rubric Rewards from Item Response Theory**. The accompanying [paper repository](https://github.com/milad1378yz/rubric_rl) contains the derivations, experiment details, and configuration tables.

This repository is a compact, method-focused release. It contains only the code needed to prepare the paper datasets, generate and judge local-policy rollouts, fit the Response Parameter Network (RPN), compute RRT rewards, and evaluate judged policy outputs. Shell launchers, model-serving infrastructure, cluster configuration, tracking integrations, and archived experiments are intentionally excluded.

## Method overview

For rollout $i$ and rubric criterion $j$, RRT models a favorable binary verdict with a two-parameter probit item-response model:

$$
P(G_{ij}=1 \mid z_i) = \Phi\left(a_{ij}(z_i-b_{ij})\right).
$$

The RPN predicts discrimination $a_{ij}>0$ and difficulty $b_{ij}$ from the prompt and criterion text. A standard-normal prior on rollout quality yields the reward as the posterior mode:

$$
R_i = \hat z_i = \arg\max_z \log p(z \mid G_i,a_i,b_i).
$$

The implementation uses a probit response function, an initial quality bracket of `[-4, 4]`, automatic bracket expansion, and 40 bisection iterations. During policy training, the RPN can be recalibrated with one stochastic partial M-step after each policy update.

## Repository layout

```text
.
├── build_rollout_cache.py   # Generate local responses and judge every criterion
├── fit_rpn.py               # Warm-start the prompt-conditioned RPN
├── reward.py                # RRT reward and optional online hard-EM update
├── evaluate.py              # Paired evaluation of held-out judged caches
├── core/
│   ├── cache.py             # Parquet and judged-cache representations
│   ├── judge.py             # Minimal per-criterion OpenAI judge
│   ├── rpn.py               # Frozen text embedder and parameter-prediction MLPs
│   └── rrt.py               # MAP inference, Fisher information, and M-step
├── data_prep/               # Deterministic paper dataset converters
└── tests/                   # Numerical and data-format regression tests
```

All runnable pipeline programs are at the repository root. Shared implementation code is under `core/`, and dataset-specific code is under `data_prep/`.

## Requirements

- Python 3.10 or newer.
- A CUDA-capable GPU is strongly recommended for rollout generation and RPN fitting. The numerical core and tests also run on CPU.
- Access to the policy and embedding checkpoints used for an experiment. Local paths and Hugging Face model identifiers are accepted by Transformers.
- An OpenAI API key for rubric judging.
- Network access for the dataset converters unless the source files are already in the Hugging Face cache.

This release generates rollouts with an in-process Transformers model. It does not require vLLM, a separate inference server, or a distributed training stack.

## Setup

### 1. Create an environment

From the repository root:

```bash
conda create --name rrt python=3.10 -y
conda activate rrt
python -m pip install --upgrade pip
```

If your machine requires a particular CUDA build of PyTorch, install that build first. Then install the project and all pipeline dependencies:

```bash
python -m pip install -e ".[pipeline]"
```

For development and tests, include the `dev` extra:

```bash
python -m pip install -e ".[pipeline,dev]"
```

The editable installation also exposes these optional commands: `rrt-build-cache`, `rrt-fit-rpn`, `rrt-evaluate`, `rrt-prepare-rubrichub`, `rrt-prepare-rar-science`, and `rrt-prepare-rubricbench`. The examples below call the root Python files directly so the execution path is explicit.

### 2. Configure paths and judging

The data converters write to `data/datasets/` by default. To store generated artifacts elsewhere, set `DATA_ROOT`. Set the judge credentials before generating a cache:

```bash
export DATA_ROOT="$PWD/data"
export OPENAI_API_KEY="your-api-key"
export JUDGE_MODEL="gpt-5.5"
```

`JUDGE_MODEL` is optional and defaults to `gpt-5.5`. `.env.example` documents the same variables, but Python does not load `.env` files automatically; export or source those values in the active shell.

### 3. Verify the installation

```bash
python -c "import core, reward; print('RRT installation OK')"
python build_rollout_cache.py --help
```

With the development dependencies installed, run the regression suite with:

```bash
python -m pytest -q
```

## Quick start

The following small run exercises the complete offline preparation path while limiting generation and judge usage. It is intended as a smoke test, not a paper reproduction.

Prepare a Science split:

```bash
python -m data_prep.convert_rubrichub --domain Science
```

Generate two rollouts for two prompts from both the training and validation splits:

```bash
python build_rollout_cache.py \
  --data-dir data/datasets/rubrichub_science_irt \
  --output data/caches/science_smoke \
  --model /path/to/policy-checkpoint \
  --splits train val \
  --limit 2 \
  --rollouts 2 \
  --max-response-tokens 512
```

Then fit a small smoke-test RPN:

```bash
python fit_rpn.py \
  --cache data/caches/science_smoke \
  --output data/rpn/science_smoke \
  --embed-model /path/to/embedding-checkpoint \
  --epochs 1 \
  --hidden 32 \
  --layers 1
```

The smoke run should produce `rollout_cache.pkl`, `summary.json`, `rpn.pt`, and `metrics.json`. A real experiment should use the full splits, rollout count, token budget, and RPN configuration described below.

## Paper workflow

### 1. Prepare deterministic dataset splits

Run one or more converters from the repository root:

```bash
# RubricHub Medical: 5,000 train / 500 validation / 500 test
python -m data_prep.convert_rubrichub --domain Medical

# RubricHub Science: 5,000 train / 500 validation / 500 test
python -m data_prep.convert_rubrichub --domain Science

# RaR Science: 5,000 train / 500 validation / 500 test
python -m data_prep.convert_rar_science

# RubricBench: 847 train / 150 validation / 150 test
python -m data_prep.convert_rubricbench
```

All converters shuffle with seed 42, preserve each prompt's complete rubric, and write `train.parquet`, `val.parquet`, and `test.parquet`. Their default output directories are:

```text
data/datasets/rubrichub_medical_irt/
data/datasets/rubrichub_science_irt/
data/datasets/rar_science_irt/
data/datasets/rubricbench_irt/
```

Use `--out-dir` to select a dataset-specific destination. Existing files are left untouched unless `--force` is supplied.

### 2. Generate and judge warm-start rollouts

Create judged rollouts for the `train` and `val` splits:

```bash
python build_rollout_cache.py \
  --data-dir data/datasets/rubrichub_science_irt \
  --output data/caches/science_base \
  --model /path/to/policy-checkpoint
```

The default generation configuration matches the paper setup exposed by this release:

| Setting | Default |
| --- | ---: |
| Rollouts per prompt | 8 |
| Prompt limit | 6,144 tokens |
| Response limit | 32,768 tokens |
| Temperature | 1.0 |
| Top-p | 1.0 |
| Top-k | disabled |
| Repetition penalty | 1.1 |
| Random seed | 42 |
| Judge concurrency | 8 |

Generation uses the checkpoint's native chat template. Thinking is enabled by default; pass `--no-enable-thinking` for checkpoints whose template does not use that mode. Use `--device cpu` only for small debugging runs—full generation on CPU is generally impractical.

The output directory contains:

```text
data/caches/science_base/
├── rollout_cache.pkl   # Prompts, rubrics, responses, and binary verdicts
└── summary.json        # Prompt counts, completed grades, missing grades, and mean score
```

Judge failures are recorded as missing rollouts instead of terminating the whole job. Inspect `summary.json` before fitting the RPN.

### 3. Warm-start the RPN

Fit the prompt-conditioned response parameters on `train` and select the checkpoint by `val` negative log-likelihood:

```bash
python fit_rpn.py \
  --cache data/caches/science_base \
  --output data/rpn/science \
  --embed-model /path/to/embedding-checkpoint
```

The default embedder is `Qwen/Qwen3-Embedding-4B`. The text encoder is frozen; separate MLPs predict discrimination and difficulty. The default RPN uses three hidden layers of width 1,024 with dropout 0.3. For the paper's RubricBench configuration, use:

```bash
python fit_rpn.py \
  --cache data/caches/rubricbench_base \
  --output data/rpn/rubricbench \
  --embed-model /path/to/embedding-checkpoint \
  --hidden 32 \
  --layers 1
```

Each output directory contains the validation-selected `rpn.pt` checkpoint and `metrics.json` with the epoch history and rollout counts. The input cache must contain both `train` and `val` splits with at least one completely judged rollout in each.

### 4. Integrate RRT into a policy-training loop

`reward.py` is intentionally framework-independent. The policy trainer needs to:

1. Judge each generated response against the prompt's rubric.
2. Convert the presence vector to an RRT reward.
3. Use those rewards for the policy update.
4. Recalibrate the RPN once after that policy update.

```python
from reward import RRTReward

rrt = RRTReward("data/rpn/science/rpn.pt", online=True)

# `rollout_presence` contains one complete Boolean criterion vector per response.
records = [
    rrt.score_from_verdicts(
        prompt,
        rubrics,
        presence,
        response_tokens=response_tokens,
    )
    for presence, response_tokens in zip(rollout_presence, rollout_token_counts)
]

grpo_rewards = [record.reward for record in records]
run_policy_update(grpo_rewards)

# Exactly one online partial M-step after the policy update.
rpn_loss = rrt.update(records)
rrt.save("checkpoints/step_001/rpn.pt")
```

Positive-point criteria are favorable when present. Negative-point criteria represent pitfalls and are favorable when absent. `record.reward` is the MAP quality estimate minus the soft response-length penalty.

The online defaults are AdamW with learning rate `2e-5`, weight decay `0.1`, mini-batch size 16, discrimination regularization `lambda_a=0.05`, and gradient clipping at 1.0. Saving the RRT checkpoint beside the policy checkpoint preserves both the RPN and its optimizer state.

To let this package call the judge itself, construct `RubricJudge` and pass it as the `judge` argument, then call `score_response`. `score_group_adaptive` implements the paper's sequential group-shared Fisher selection and must be used with `online=False`.

### 5. Build held-out caches

Generate an independently judged `test` cache for every policy being compared. Use the same test parquet, ordering, rollout count, generation settings, and judge model for all policies:

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

### 6. Evaluate policies

Compare the held-out caches with paired prompt-level bootstrap intervals:

```bash
python evaluate.py \
  --data data/datasets/rubrichub_science_irt/test.parquet \
  --reference base \
  --output results/science.json \
  base=data/caches/science_base_test \
  rrt=data/caches/science_rrt_test
```

The evaluator verifies exact prompt and rubric identity before comparing caches. It reports the mean unweighted criterion score, the normalized point-weighted score, and paired differences against the named reference. By default it uses eight complete rollouts per prompt and 10,000 bootstrap samples.

Evaluation writes:

```text
results/
├── science.json                 # Aggregate means and paired confidence intervals
└── science.per_prompt.parquet   # Scores retained at prompt level
```

## Artifact overview

A complete run typically has the following local structure:

```text
data/
├── datasets/<dataset>/{train,val,test}.parquet
├── caches/<policy>/{rollout_cache.pkl,summary.json}
└── rpn/<experiment>/{rpn.pt,metrics.json}
results/
├── <experiment>.json
└── <experiment>.per_prompt.parquet
```

Generated data, caches, checkpoints, and results are ignored by Git by default.

## CLI reference

Use `--help` for the complete options of any command:

```bash
python build_rollout_cache.py --help
python fit_rpn.py --help
python evaluate.py --help
python -m data_prep.convert_rubrichub --help
python -m data_prep.convert_rar_science --help
python -m data_prep.convert_rubricbench --help
```

The most useful controls for small runs are `--limit`, `--rollouts`, and `--max-response-tokens` on `build_rollout_cache.py`. The most useful RPN memory controls are `--hidden`, `--layers`, `--batch-size`, and `--embed-batch-size` on `fit_rpn.py`.

## Minimal numerical example

The inference core can be used without the data pipeline or API judge:

```python
import numpy as np

from core.rrt import estimate_quality, oriented_labels

presence = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=float)
favorable, _ = oriented_labels(presence, points=[1, 1, 1])
a = np.array([1.4, 0.8, 1.1])
b = np.array([-0.5, 0.2, 0.9])

n, k = favorable.shape
quality = estimate_quality(
    favorable.ravel(),
    np.tile(a, n),
    np.tile(b, n),
    np.repeat(np.arange(n), k),
    n,
)
print(quality)
```

## Troubleshooting

**The judge reports an authentication error.**  Confirm that `OPENAI_API_KEY` is exported in the same shell that launches `build_rollout_cache.py`. If the configured judge model is unavailable to the account, select an accessible model with `--judge-model` or `JUDGE_MODEL`.

**Rollout generation runs out of GPU memory.**  Start with `--limit 2 --rollouts 2 --max-response-tokens 512`, use a smaller policy checkpoint, or reduce the prompt and response token limits. Rollouts are generated sequentially, so `--rollouts` mainly changes total runtime rather than model memory.

**RPN fitting runs out of GPU memory.**  Reduce `--embed-batch-size` first, then `--batch-size`, `--hidden`, or `--layers`. Do not change the architecture when loading an existing checkpoint; its architecture is stored with the checkpoint.

**`fit_rpn.py` says the cache is incomplete.**  Build the default `train val` cache and check `summary.json`. Rollouts with failed or unparseable judge verdicts are excluded, and both splits need at least one complete rollout.

**Evaluation says caches do not match.**  Rebuild every policy cache from the exact same test parquet and row order. The evaluator deliberately rejects comparisons with missing, reordered, or modified prompts and rubrics.

**A dataset converter skips existing files.**  This protects reproducibility. Pass `--force` only when you intentionally want to replace the generated split files.

## Scope and assumptions

RRT assumes rubric criteria are monotone indicators of a shared latent quality target. Explicit trade-offs should retain normative point weighting, and strongly dependent criteria require a model that accounts for that dependence. Distributed policy training, high-throughput serving, and job scheduling are deployment choices and are therefore outside the scope of this reference repository.
