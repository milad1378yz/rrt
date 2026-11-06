# Reviews

## Reviewer Ve73

### Paper Summary:

This paper proposes Rubric Rewards from Item Response Theory (RRT) that treats multiple rubric criteria as noisy measurements of a shared latent response quality, rather than directly aggregating their binary verdicts through weighted summation. It introduces a Response Parameter Network (RPN) to estimate criterion difficulty and discrimination, and infers latent quality scores from the observed verdict patterns to serve as rewards for GRPO training. Additionally, the framework uses Fisher information to adaptively select informative criteria, achieving competitive policy performance while substantially reducing judging costs. Experiments on RLHF demonstrate the effectiveness of RRT.

### Presentation: 2: Generally understandable, but requires substantial effort in places.

### Informativeness: 3: I gained a substantially deeper understanding of the problem or question.

### Soundness: 3: I have no substantive concerns about the paper’s soundness.

### Critical Strengths:

1. Simply summing or averaging multiple criterion rewards in rubric reward + GRPO training settings can overlook differences in criterion difficulty and informativeness. The paper offers an interesting perspective on how these signals could be aggregated more meaningfully.

2. The IRT-based formulation is intuitive and well motivated. Treating rubric criteria as noisy measurements of an underlying response quality provides a principled alternative to conventional weighted aggregation. Modeling criterion difficulty and discrimination also allows the framework to make better use of different verdict patterns.

### Critical Weaknesses:

1. The assumption of a shared latent quality may be too strong for general rubric-based RL. RRT assumes that all criteria are monotonic indicators of a single underlying response quality. However, real-world rubrics often involve different objectives, which may have genuine trade-offs.

2. It is unclear whether the inferred latent quality better reflects actual response quality or human preferences. The paper demonstrates that RRT can distinguish different verdict patterns and reduce reward ties, but greater reward discrimination does not necessarily mean better reward quality. In particular, criterion difficulty and statistical informativeness are not necessarily equivalent to criterion importance. Some independent evaluation against human preferences would make this claim more convincing. For example, I suggest that the authors could conduct additional experiments on human preference datasets (like helpsteer3) to examine whether the inferred latent-quality scores better align with human preference judgments than conventional additive reward aggregation.

3. I am also somewhat concerned about the additional complexity introduced by the Response Parameter Network (RPN). Although the authors report relatively low computational overhead for online RPN updates, the framework still requires a separate warm-start procedure and an additional parameterized model that is continuously updated alongside the policy. Since the RPN directly determines the inferred reward, its initialization quality and calibration stability could potentially affect the overall RL training dynamics. Moreover, the improvement from online RPN updates over a frozen RPN appears relatively modest (Table 2). It would be helpful to further investigate sensitivity to RPN initialization, update frequency, and training hyperparameters, as well as whether the additional online calibration consistently improves policy performance across different settings.

4. The downstream RL improvements are relatively modest and not fully consistent across models. (That's fine, but experiments on LLMs with more parameters would be better, like 32B.)

### Questions For Authors:

Please refer to the weaknesses.

### Flag For Ethics Review: No

### Rating: 3: Weak acceptance

### Confidence: 3: I am familiar with this topic and its relevant literature.
