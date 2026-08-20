# Rubric Rewards from Item Response Theory

Reference implementation of **Rubric Response Theory (RRT)**. RRT turns binary rubric
verdicts into a scalar reward for group relative policy optimization (GRPO) and uses the
same model to select which criteria to judge.

This repository contains the method-specific workflow: prepare the paper datasets,
generate and judge rollouts, warm-start the Response Parameter Network (RPN), compute
RRT rewards, update the RPN online, select criteria by Fisher information, and evaluate
held-out policies. It does not include the paper's distributed policy-training,
model-serving, or cluster infrastructure.

## Method

For prompt $q$, rollout $i$, and rubric criterion $j$, the RPN predicts criterion
discrimination $a_j>0$ and difficulty $b_j$ from the prompt and criterion text:

$$
(a_j,b_j)=\psi(q,c_j), \qquad
P_{ij}:=P(G_{ij}=1\mid z_i)=\Phi\!\left(a_j(z_i-b_j)\right).
$$

$G_{ij}=1$ always denotes a favorable verdict. RRT combines the verdict likelihoods
with a standard normal prior and uses the posterior mode as the reward:

$$
R_i=\hat z_i=\arg\max_z\left[
-\frac{z^2}{2}+\sum_j G_{ij}\log P_{ij}+(1-G_{ij})\log(1-P_{ij})
\right].
$$

The implementation uses the Gaussian CDF, an initial quality bracket of $[-4,4]$ with
automatic expansion, and 40 bisection steps. After each policy update, one stochastic
partial M-step can recalibrate the RPN to the current rollout distribution. With a
frozen RPN, adaptive selection repeatedly judges the remaining criterion with the most
total Fisher information, $\sum_i I_j(\hat z_i)$, over the rollout group.

Under this model, the rubric likelihood score is locally efficient for measuring shared
quality. Fixed rubric points generally are not: they express normative importance, not
how strongly a verdict distinguishes the current rollouts.

Rubric point signs orient verdicts: satisfying a positive criterion is favorable, while
avoiding a negative-point pitfall is favorable. Point magnitudes are retained for the
normalized points evaluation metric, but they do not weight the RRT reward.

RRT assumes local independence: rubric criteria are conditionally independent given one
shared quality target, and each is a monotone indicator of that target. Explicit
tradeoffs should retain point weights, and known criterion dependencies require a model
of that dependence.

## Main paper results

- Under repeated judging, RRT increased the share of rollout pairs with a stable nonzero
  ordering from 54.1% to 59.5%, a gain of 5.4 percentage points.
- With Qwen3.5-4B across Medical, Science, RaR Science, and RubricBench, RRT with an
  online RPN reached a 75.1% macro criterion score, compared with 73.4% for Vanilla
  GRPO.
- A 0.50 adaptive Fisher budget reduced judge requests by 49.0% on Medical and Science.
  Across all four datasets, its macro criterion score gain from the base policy was 2.3
  points, compared with 2.4 points for Vanilla GRPO with full judging.

## Installation

Python 3.10 or newer is required. A CUDA GPU is recommended for rollout generation and
RPN fitting.

```bash
conda create --name rrt python=3.10 -y
conda activate rrt
python -m pip install -e ".[pipeline,dev]"
export OPENAI_API_KEY="your-api-key"
```

The judge defaults to GPT-5.5 with reasoning disabled. Set `JUDGE_MODEL` to override
it. Dataset outputs default to `data/datasets/`; set `DATA_ROOT` to change the artifact
root.

## Workflow

### 1. Prepare a dataset

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

The converters use seed 42 and preserve each prompt's complete rubric. Medical,
Science, and RaR Science each use 5,000 training, 500 validation, and 500 test prompts.
RubricBench uses 847, 150, and 150.

### 2. Generate and judge warm-start rollouts

```bash
python build_rollout_cache.py \
  --data-dir data/datasets/rubrichub_science_irt \
  --output data/caches/science_base \
  --model /path/to/base-policy-checkpoint
```

This builds the required `train` and `val` cache. The defaults follow the paper's Qwen
generation setup: eight rollouts per prompt, 6,144 prompt tokens, 32,768 response
tokens, temperature 1.0, top-p 1.0, top-k disabled, and repetition penalty 1.1. Judging
uses concurrency 8, a 50-second timeout, and three retries.

The full command makes many judge requests. For a smoke test, add
`--limit 2 --rollouts 2 --max-response-tokens 512`. Pass `--no-enable-thinking` for
Qwen3.5-2B and Llama-3.1-8B-Instruct, and use `--repetition-penalty 1.0` for Llama, as in
the paper.

### 3. Warm-start the RPN

```bash
python fit_rpn.py \
  --cache data/caches/science_base \
  --output data/rpn/science
```

The standard RPN uses the frozen `Qwen/Qwen3-Embedding-4B` encoder and separate MLPs for
$a_j$ and $b_j$. Training uses complete rollout verdicts from `train` and retains the
checkpoint with the lowest `val` negative log-likelihood.

### 4. Use RRT during policy training

`reward.py` is independent of the policy-training framework. Collect reward records
across one policy step, update the policy, and then update the RPN once:

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
rrt.update(records)  # one stochastic partial M-step after the policy update
rrt.save("checkpoints/step_001/rpn.pt")
```

`record.quality` is the MAP reward from the paper. `record.reward` additionally applies
the shared soft length penalty used for policy training. To judge responses inside this
package, pass a `RubricJudge` to `RRTReward` and call `score_response`.

For partial judging, construct `RRTReward(..., online=False, judge=judge)` and call
`score_group_adaptive(..., budget=m)`, where `m` is the number of criteria to judge.
The paper freezes the RPN for adaptive Fisher experiments.

### 5. Evaluate held-out policies

Build a `test` cache for each policy with `build_rollout_cache.py --splits test`, then
compare caches on the same prompts and rubrics:

```bash
python evaluate.py \
  --data data/datasets/rubrichub_science_irt/test.parquet \
  --reference base \
  --output results/science.json \
  base=data/caches/science_base_test \
  rrt=data/caches/science_rrt_test
```

The evaluator reports criterion score, normalized points score, and paired prompt-level
bootstrap intervals. Run `python <command> --help` for all options.

## Tests

```bash
python -m pytest -q
```
