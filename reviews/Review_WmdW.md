# Reviews

## Reviewer WmdW

### Paper Summary:

The paper proposes new method called Rubric Response Theory (RRT), which employs a reward inferred from 2-parameter item response to replace conventional method (sums the points assigned to satisfied criteria) used in rubric-based GRPO. In RRT, each binary criterion verdict $G_{\mathrm{ij}}$ is modelled as Bernoulli $(P_{ij})$ under local independence (Equation 1), maximizer is found by bisection using the log posterior gradient (Theorem 3 in Appendix A.1), $R_i = \hat{z}_i$ is employed as the GRPO rewards (Appendix B.2), criterion difficulty $b_j$ and discrimination $a_j$ are predicted from prompt and criterion text by a RPN (Section 3.1), RRT calibrates RPN from new rollout verdicts using 1 online EM sweep per-policy step (Section 3.2). The theoritical contributions are (1) fisher information of a rubric criterion (Theorem 1), (2) a local SNR bound showing that the rubric likelihood score is up to a nonzero affine map, so that no fixed vector of rubric points attains the bound at every quality level once criterion parameters differs (Theorem 2), and (3) the result that under the logistic link the reward order depends only on sum of $a_j \times G_i j$ and not on $b_j$ (Theorem 4), whereas difficulty can reverse RRT reward order of 2-fixed verdict vectors (Theorem 5). Empirically, with Qwen3.5-4B and a GPT-5.5 judge on Medical, Science, RaR science, and RubricBench. RRT + online RPN reaches 75.1 macro criterion score versus 73.4 for vanilla GRPO and 74.3 for POW3R (Table 2). Adaptive Fisher selection at budget 0.50 cuts judge requests by 49.0% and step time by 23.5-28.7% while matching full judging vanilla GRPO.

### Presentation: 2: Generally understandable, but requires substantial effort in places.

### Informativeness: 2: I learned something interesting or useful.

### Soundness: 2: I have minor or fixable concerns about the paper’s soundness.

### Critical Strengths:

#### S1. Reward, calibration, and criteria selection is unified by single principled model

- 2PL probit models yields the reward (MAP Quality), online EM calibration, and fisher-based criterion ranking.

#### S2. Clear and correct theoritical characterization

- Theorem 2 provides well-posed argument that additive rubric points cannot attain the local information bound at every quality level once criterion parameters differ

- Theorems 4-5 give a concrete, testable reason to prefer probit over logistic link

#### S3. Practical reduction in judge cost

- At criterion budget 0.50, adaptive FIsher selection reduces judge request by 49.0% and step time by 23.5-28.7% while matching Vanilla GRPO with full judging (Table 4).

### Critical Weaknesses:

#### W1. One training run per condition

- Each condition has one training run (Section 4.1)

- The reported boostrap intervals are computed over prompt groups with the trained policies fixed, so they do not capture training-seed variance

#### W2. The source of the gain is unclear

- RRT with estimated parameters (batch pass rate 73.6, cached pass rate 73.4) is distinguishable from vanilla GRPO (73.4)

- Table 1 shows identical within-criterion ROC-AUC (65.8) for RRT

- The improvement appears only when the text-conditioned RPN supplies the parameters

- There is possibility that the gain may come from the RPN's text prior rather than likelihood aggregation (core of the paper)

### Questions For Authors:

Q1. Could you please report at least 3 training seeds for Vanilla GRPO, POW3R, RRT + frozen RPN, and RRT + online RPN on Qwen3.5-4B, with mean +- std and a test of the paired difference?

Q2. How many judge requests produced the cached verdicts used for the RPN warm start (rollouts per training prompt × criteria)? Is the 49.0% saving still positive after amortizing this cost over a training run?

Q3. Selected checkpoint steps differ notably across conditions (e.g., Medical: 145 frozen vs 115 online, Table 28). How sensitive are the Table 2 differences to evaluating every condition at the same step?

### Flag For Ethics Review: No

### Rating: 3: Weak acceptance

### Confidence: 2: I have broad familiarity with this field but little or no direct research experience on this topic.

