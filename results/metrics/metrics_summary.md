# HalfCheetah-v5 Benchmark Metrics

All metrics aggregated over 3 random seeds per algorithm. Final reward = mean over the last 10% of evaluation points; AUC = area under the (smoothed, mean-across-seeds) learning curve, normalized so 1.0 corresponds to instantly hitting the global best final reward. Sample efficiency thresholds are fractions of the best algorithm's final reward.

| Algorithm | Final reward (mean ± std over seeds) | Best reward (smoothed) | AUC (normalized) | Steps to 50% of best | Steps to 80% of best | Across-seed CV | Final ep length |
|---|---|---|---|---|---|---|---|
| SAC | 11,369 ± 97 | 11,505.9 | 0.785 | 130,000 | 380,000 | 0.009 | 1000 |
| DDPG | 11,206 ± 1,147 | 11,956.7 | 0.764 | 180,000 | 370,000 | 0.102 | 1000 |
| TD3 | 10,810 ± 245 | 10,974.3 | 0.781 | 120,000 | 370,000 | 0.023 | 1000 |
| PPO | 1,405 ± 58 | 1,391.2 | 0.067 | — | — | 0.041 | 1000 |

**Reading the table.** *Final reward* is the headline performance number; the ± span is the spread across seeds (smaller = more stable). *Best reward* shows peak performance during training and is useful for spotting algorithms that learned but then collapsed. *AUC* rewards algorithms that learn fast and stay high. *Steps-to-X%* are the canonical sample-efficiency numbers — lower is better. *Across-seed CV* is the coefficient of variation of the final reward across seeds (lower = more reproducible). *Final ep length* should be 1000 (the HalfCheetah time limit) for a healthy policy; lower means the agent is somehow terminating early.