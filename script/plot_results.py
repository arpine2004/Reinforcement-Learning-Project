from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt

Path("results/plots").mkdir(parents=True, exist_ok=True)

algos = {
    "SAC": "results/sac_seed*/evaluations.npz",
    "TD3": "results/td3_seed*/evaluations.npz",
    "DDPG": "results/ddpg_seed*/evaluations.npz",
    "PPO": "results/ppo_seed*/evaluations.npz",
}

colors = {
    "SAC": "#01696f",
    "TD3": "#006494",
    "DDPG": "#a12c7b",
    "PPO": "#437a22",
}

loaded = {}

for algo, pattern in algos.items():
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"Skipping {algo}: no files matched {pattern}")
        continue

    rewards = []
    timesteps = None

    for path in paths:
        data = np.load(path, allow_pickle=True)
        rewards.append(data["results"].mean(axis=1))
        timesteps = data["timesteps"]

    rewards = np.array(rewards)

    loaded[algo] = {
        "timesteps": timesteps,
        "mean": rewards.mean(axis=0),
        "std": rewards.std(axis=0),
    }

if not loaded:
    raise RuntimeError("No evaluation files found. Train models first.")

# Separate plot for each algorithm
for algo, stats in loaded.items():
    plt.figure(figsize=(9, 5))
    plt.plot(
        stats["timesteps"],
        stats["mean"],
        label=algo,
        linewidth=2,
        color=colors[algo]
    )
    plt.fill_between(
        stats["timesteps"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        alpha=0.15,
        color=colors[algo],
    )

    plt.title(f"HalfCheetah-v5: {algo} Reward vs. Timestep")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{algo.lower()}_learning_curves.png", dpi=180)
    plt.savefig(f"results/plots/{algo.lower()}_learning_curves.pdf")
    plt.close()

# Combined plot
plt.figure(figsize=(10, 6))

for algo, stats in loaded.items():
    plt.plot(
        stats["timesteps"],
        stats["mean"],
        label=algo,
        linewidth=2.2,
        color=colors[algo]
    )
    plt.fill_between(
        stats["timesteps"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        alpha=0.12,
        color=colors[algo],
    )

plt.title("HalfCheetah-v5: SAC vs. TD3 vs. DDPG vs. PPO")
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Reward")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/all_algorithms_learning_curves.png", dpi=180)
plt.savefig("results/plots/all_algorithms_learning_curves.pdf")
plt.close()

print("Saved separate plots and combined plot to results/plots/")