# Rubric Rewards from Item Response Theory

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white) ![Microsoft icon](https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Microsoft_icon.svg/20px-Microsoft_icon.svg.png)

**Explore:** [🎬 RRT in action](#rrt-in-action) · [⚙️ Installation](#installation) · [🧪 Experiments](#experiments) · [🎯 Policy training](#policy-training) · [📊 Evaluation](#evaluation)

**Rubric Response Theory (RRT)** converts binary rubric verdicts into rewards for language model training. This implementation provides dataset preparation, rollout judging, Response Parameter Network (RPN) fitting, frozen and online RRT rewards, adaptive criterion selection, and paired evaluation on held-out prompts.

> [!NOTE]
> Integrate [`RRTReward`](reward.py) with your GRPO or PPO trainer. RRT computes rewards and updates the RPN; your trainer collects rollouts and optimizes the policy.

## RRT in action

RRT uses learned criterion difficulty and discrimination to turn each response's verdict pattern into a quality estimate. In this example, responses with the same rubric point total receive different rewards.

![RRT assigns different quality rewards to responses with equal rubric point totals, using the difficulty and discrimination of the criteria they pass.](assets/GMatrixReward.gif)

<details>
<summary>From verdicts to a reward: quality inference</summary>

The E step combines the criterion verdicts with a Gaussian quality prior and finds the maximum a posteriori (MAP) quality by bisection. This inferred quality is the reward.

![Criterion verdicts reshape the quality posterior, then bisection finds its mode to obtain the RRT reward.](assets/EStepSearch.gif)

</details>

## Installation

Run the commands below from the repository root.

```bash
conda create --name rrt python=3.10 -y
conda activate rrt
python -m pip install -e ".[pipeline]"
export OPENAI_API_KEY="your-api-key"
```

The `pipeline` extra installs the dataset and judge dependencies. Set `JUDGE_MODEL` in your shell to override the default judge model. A CUDA GPU is recommended for rollout generation and RPN fitting; both scripts accept `--device`.

## Experiments

The commands below use RubricHub Science and a local policy checkpoint. Prepare the data, build a judged cache, and fit the RPN before integrating rewards into policy training.

**Workflow:** 📚 Prepare data → 🧪 Judge rollouts → 🧠 Fit the RPN → 🎯 Train the policy → 📊 Evaluate

### Dataset preparation

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

Each converter writes `train.parquet`, `val.parquet`, and `test.parquet`:

| Dataset | Default output directory |
| --- | --- |
| RubricHub Medical | `data/datasets/rubrichub_medical_irt` |
| RubricHub Science | `data/datasets/rubrichub_science_irt` |
| Rubrics as Rewards Science | `data/datasets/rar_science_irt` |
| RubricBench | `data/datasets/rubricbench_irt` |

Use `--out-dir /path/to/output` to choose a destination, or set `DATA_ROOT` to replace the default `data` directory for dataset conversion. The parquet rows include chat messages in `prompt`, rubric criteria in `reward_model`, and the original prompt and full rubric in `extra_info`.

### Rollout generation and judging

```bash
python build_rollout_cache.py \
  --data-dir data/datasets/rubrichub_science_irt \
  --output data/caches/science_base \
  --model /path/to/base-policy-checkpoint
```

By default, this command generates eight responses per prompt for the `train` and `val` splits using the policy checkpoint's native chat template. The judge checks each rubric criterion for each response.

The output directory contains `rollout_cache.pkl` with the verdicts and `summary.json` with counts of graded and missing rollouts. Use `--rollouts` and `--splits` to change the cache contents, or `python build_rollout_cache.py --help` for generation and judging options.

### RPN fitting

The RPN predicts criterion difficulty and discrimination from the prompt and criterion text. It uses a frozen text encoder, `Qwen/Qwen3-Embedding-4B` by default, and trains two small prediction networks. In each M step, it fits the observed verdicts while holding the inferred quality targets fixed.

![The M step adjusts a criterion's difficulty and discrimination to fit observed pass and fail verdicts at fixed quality targets.](assets/MStepFit.gif)

```bash
python fit_rpn.py \
  --cache data/caches/science_base \
  --output data/rpn/science
```

Fitting requires complete judged rollouts in both the `train` and `val` cache splits. It saves the checkpoint with the lowest validation negative log likelihood to `data/rpn/science/rpn.pt` and the training history to `data/rpn/science/metrics.json`. The checkpoint stores the RPN prediction networks and the encoder configuration; loading it also loads the referenced encoder.

Use `--embed-model` to choose another encoder, or `python fit_rpn.py --help` for training options.

### Policy training

Collect all rollout records for a policy step, train the policy with their rewards, and then update and save the RPN:

```python
from reward import RRTReward

rrt = RRTReward("data/rpn/science/rpn.pt", online=True)

records = [
    rrt.score_from_verdicts(
        prompt,
        rubrics,
        presence,
    )
    for prompt, rubrics, presence in policy_step_rollouts
]

run_policy_update([record.reward for record in records])
rrt.update(records)
rrt.save("checkpoints/step_001/rpn.pt")
```

Each entry in `policy_step_rollouts` is `(prompt, rubrics, presence)` for one response. `rubrics` is a list of dictionaries with `criterion` and `points` fields, and `presence` is the aligned Boolean verdict vector. The sign of `points` determines the favorable outcome: satisfying a positive criterion or avoiding a negative criterion. Replace `policy_step_rollouts` and `run_policy_update` with your trainer's rollout data and policy update.

Each returned record exposes `reward`, `quality`, and `uncertainty`. The reward equals the inferred MAP quality, and uncertainty estimates the posterior standard deviation. With `online=True`, `rrt.update(records)` performs one RPN optimizer step over the policy step's collected records. Set `online=False` to keep the RPN frozen.

To let this package judge a response, pass a `RubricJudge` to `RRTReward` and call `score_response`.

#### Minimal GRPO with veRL

Prepare a veRL checkout at `/path/to/verl` using its [installation guide](https://verl.readthedocs.io/en/latest/start/install.html), including FSDP and your chosen rollout backend. Install RRT in that same Python environment. The dataset converters above provide the `prompt`, `reward_model`, and `extra_info` fields used by the adapter.

```bash
python -m pip install -e "/absolute/path/to/rrt[pipeline]"
```

For the smallest integration, use a frozen RPN through veRL's custom reward function. Save this adapter as `verl_rrt_reward.py` in the repository root:

```python
from functools import lru_cache

from core.judge import RubricJudge
from reward import RRTReward


@lru_cache(maxsize=None)
def load_rrt(checkpoint, device):
    return RRTReward(
        checkpoint,
        device=device,
        online=False,
        judge=RubricJudge(),
    )


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    checkpoint="",
    device="cpu",
):
    info = extra_info or {}
    return load_rrt(checkpoint, device).score_response(
        info["user_prompt"],
        solution_str,
        info["rubrics_full"],
    ).reward
```

Start from [veRL's FSDP GRPO launcher](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_8b_fsdp.sh) and add the RRT reward overrides:

```bash
RRT_ROOT=/absolute/path/to/rrt
VERL_ROOT=/absolute/path/to/verl

cd "$VERL_ROOT"
PYTHONPATH="$RRT_ROOT:${PYTHONPATH:-}" \
MODEL_PATH=/path/to/base-policy-checkpoint \
VERL_USE_UV=0 \
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files="$RRT_ROOT/data/datasets/rubrichub_science_irt/train.parquet" \
  data.val_files="$RRT_ROOT/data/datasets/rubrichub_science_irt/val.parquet" \
  reward.custom_reward_function.path="$RRT_ROOT/verl_rrt_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers=1 \
  "+reward.custom_reward_function.reward_kwargs.checkpoint=$RRT_ROOT/data/rpn/science/rpn.pt" \
  "+reward.custom_reward_function.reward_kwargs.device=cpu" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$RRT_ROOT" \
  'trainer.logger=["console"]'
```

`VERL_USE_UV=0` makes the launcher use the active Python environment where RRT and veRL are installed. Adjust the launcher's device count, batch sizes, and token limits for your hardware and dataset.

This adapter returns the RRT reward from a frozen RPN. For online RPN updates, integrate `rrt.update(records)` once after each policy step in your trainer, following the loop above.

#### Adaptive criterion selection

With a frozen RPN, adaptive selection judges the criterion with the largest total Fisher information across the response group, updates the quality estimates, and repeats until the criterion budget is reached. The animation illustrates selection on a rubric with five criteria.

![Adaptive Fisher selection chooses informative criteria and updates response quality estimates after each judging step.](assets/FisherInfo.gif)

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
)
```

`responses` contains the rollouts for one prompt, and `criterion_budget` is an integer from one to the number of rubric criteria. The result contains one reward record per response and the selected criteria's zero-based indices. Each selected criterion is judged across the whole response group.

### Evaluation

First build a test cache for each policy using the same held-out dataset, judge settings, and number of rollouts:

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

The report in `results/science.json` contains the mean favorable criterion rate (`criterion_score`), the rubric score weighted by absolute points (`normalized_score`), and paired differences from the reference policy with 95% bootstrap confidence intervals. Per-prompt scores are saved to `results/science.per_prompt.parquet`.

Evaluation uses eight judged responses per prompt by default and includes only prompts with enough complete verdicts in every cache. Set `--rollouts` to match the caches if you generated a different number, or use `python evaluate.py --help` for more options.
