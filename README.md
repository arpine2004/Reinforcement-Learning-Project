# Reinforcement Learning Benchmark: HalfCheetah-v5

Comparative study of four deep RL algorithms on the MuJoCo **HalfCheetah-v5** continuous-control task. Each algorithm is trained with 3 random seeds over 1 million timesteps.

**Algorithms compared:** SAC · TD3 · DDPG · PPO

---

## Results

| Algorithm | Final Reward (mean ± std) | Best Reward | AUC (normalized) | Steps to 50% | Steps to 80% | Seed CV |
|-----------|--------------------------|-------------|------------------|--------------|--------------|---------|
| SAC  | 11,369 ± 97   | 11,505.9 | 0.785 | 130,000 | 380,000 | 0.009 |
| DDPG | 11,206 ± 1,147 | 11,956.7 | 0.764 | 180,000 | 370,000 | 0.102 |
| TD3  | 10,810 ± 245  | 10,974.3 | 0.781 | 120,000 | 370,000 | 0.023 |
| PPO  | 1,405 ± 58    | 1,391.2  | 0.067 | —       | —       | 0.041 |

- **Final reward**: mean over the last 10% of evaluation checkpoints across 3 seeds
- **AUC**: area under the smoothed learning curve, normalized to the global best
- **Steps to X%**: sample efficiency — timesteps needed to reach X% of the best algorithm's final reward
- **Seed CV**: coefficient of variation across seeds (lower = more reproducible)

Off-policy algorithms (SAC, TD3, DDPG) substantially outperform on-policy PPO on this task. SAC achieves the highest stable final performance; DDPG reaches the highest peak reward but with much higher variance across seeds.

---

## Project Structure

```
.
├── train.py            # Train an algorithm across multiple seeds
├── evaluate.py         # Evaluate a saved checkpoint
├── plot_results.py     # Learning curves per algorithm + combined plot
├── compute_metrics.py  # Aggregate metrics (AUC, sample efficiency, etc.)
├── extra_plots.py      # Extended analysis plots
├── requirements.txt
├── checkpoints/        # Saved model weights (best + final per seed)
│   └── {algo}_seed{N}/
│       ├── best_model.zip
│       └── final_model.zip
├── results/            # Evaluation logs and plots
│   ├── {algo}_seed{N}/evaluations.npz
│   ├── metrics/
│   │   ├── metrics.csv
│   │   ├── metrics_per_seed.csv
│   │   └── metrics_summary.md
│   └── plots/
└── logs/               # TensorBoard logs
```

---

## Setup

**Prerequisites:** Python 3.10+, MuJoCo (installed automatically via `gymnasium[mujoco]`)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Train

```bash
# Train a single algorithm across 3 seeds (default)
python train.py --algo sac

# Specify seeds and timesteps
python train.py --algo td3 --seeds 0 1 2 --timesteps 1000000

# Available algorithms: sac, td3, ddpg, ppo
```

Models are saved to `checkpoints/{algo}_seed{N}/` and evaluation logs to `results/{algo}_seed{N}/`.

### Evaluate a saved model

```bash
python evaluate.py --algo sac --seed 0 --episodes 10
```

Loads `best_model.zip` if available, otherwise `final_model.zip`.

### Plot learning curves

```bash
# Per-algorithm plots + combined comparison plot
python plot_results.py

# Extended analysis (per-seed, bar charts, sample efficiency, stability)
python extra_plots.py
```

Plots are saved to `results/plots/` as PNG and PDF.

### Compute metrics

```bash
python compute_metrics.py
```

Writes `metrics.csv`, `metrics_per_seed.csv`, and `metrics_summary.md` to `results/metrics/`.

### TensorBoard

```bash
tensorboard --logdir logs/
```

---

## Algorithm Hyperparameters

| Setting | SAC | TD3 | DDPG | PPO |
|---------|-----|-----|------|-----|
| Batch size | 256 | 256 | 256 | 64 |
| Learning starts | 10,000 | 10,000 | 10,000 | — |
| Action noise | — | Normal(0, 0.1) | Normal(0, 0.1) | — |
| n_steps / n_epochs | — | — | — | 2048 / 10 |
| Parallel envs | 1 | 1 | 1 | 4 |

All algorithms use an MLP policy. Evaluation runs every 10,000 timesteps over 5 episodes.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `gymnasium[mujoco]` | HalfCheetah-v5 environment |
| `stable-baselines3[extra]` | SAC, TD3, DDPG, PPO implementations |
| `tensorboard` | Training monitoring |
| `numpy` / `pandas` | Data handling |
| `matplotlib` | Plotting |
