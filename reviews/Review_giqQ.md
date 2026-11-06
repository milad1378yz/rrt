# Reviews

## Reviewer giqQ

### Paper Summary:

This paper investigates Reward Aggregation and judging costs in rubric-based reinforcement learning for LLMs. The authors propose Rubric Response Theory (RRT), which uses Item Response Theory to model the difficulty and discrimination of different rubric criteria and estimates the latent quality of responses from multiple criterion verdicts. In addition, the authors introduce a Response Parameter Network (RPN) and use Fisher Information to adaptively select criteria for evaluation, thereby reducing LLM judge costs. Experimental results show that the proposed method can improve rubric satisfaction rates in certain settings.

### Presentation: 2: Generally understandable, but requires substantial effort in places.

### Informativeness: 3: I gained a substantially deeper understanding of the problem or question.

### Soundness: 2: I have minor or fixable concerns about the paper’s soundness.

### Critical Strengths:

1. The paper addresses an important research problem.

2. The paper is well written and easy to follow.

3. The paper develops a reward estimation model based on Item Response Theory and provides relatively comprehensive mathematical derivations and theoretical analyses.

4. The experiments are extensive, covering multiple datasets and policy models. The paper also compares RRT with existing reward aggregation methods, such as POW3R and DIVA, and includes analyses of parameter estimation, reward stability, and criterion selection.

5. Fisher-based Criterion Selection achieves promising results. By reducing judging requests by approximately half, it maintains performance close to GRPO with full rubric judging. The paper also provides an analysis of actual training-time overhead.

### Critical Weaknesses:

1. The main technical components, including Two-parameter Item Response Theory, MAP Estimation, Online EM, and Fisher Information-based Selection, are well-established methods. This paper mainly combines and applies these techniques to rubric-based RL. The authors should better clarify the technical novelty compared with traditional IRT and existing adaptive reward aggregation approaches. Furthermore, Online RPN provides only limited improvements over Frozen RPN, raising questions about its necessity.

2. RRT assumes that different rubric criteria are monotonic indicators of a single latent quality variable and are conditionally independent given that variable. However, real-world rubrics may involve different or even conflicting dimensions, such as Correctness, Safety, Completeness, and Conciseness. It remains unclear whether these dimensions can be adequately represented by a single latent quality variable. Although the authors analyze criterion dependence, the applicability of this core assumption is not sufficiently validated.

3. Although RRT improves over Vanilla GRPO by 1.7 percentage points on Qwen3.5-4B, its performance slightly decreases on Qwen3.5-2B and improves by only 0.1 percentage points on Llama-3.1-8B. Therefore, the experimental results do not sufficiently demonstrate consistent improvements across different model scales and architectures.

4. The paper explicitly states that each experimental condition involves only one training run. Given the sensitivity of reinforcement learning to random seeds, sampling outcomes, and reward estimation noise, a single run is insufficient to establish the stability of relatively small performance improvements. The authors should conduct multiple independent training runs and report the mean and variance.

5. The code is not publicly available, which limits the reproducibility of the work.

6. The paper shows that Static Fisher and Adaptive Fisher selection can skip approximately 19.3% and 21.0% of criteria, respectively, under the same reward fidelity requirement. The difference is relatively small, making it unclear whether the additional complexity of adaptive selection is justified. The authors should further analyze its benefits across different training stages and computational budgets.

7. The comparison in the abstract is somewhat misleading. The proposed method is compared against GRPO with conventional rubric-based rewards, rather than GRPO itself.

### Questions For Authors:

see the weaknesses

### Flag For Ethics Review: No

### Rating: 2: Weak rejection

### Confidence: 2: I have broad familiarity with this field but little or no direct research experience on this topic.

