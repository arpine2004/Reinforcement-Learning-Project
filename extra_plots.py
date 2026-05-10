from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ALGOS = ["SAC", "TD3", "DDPG", "PPO"]
PATTERN = "results/{algo}_seed*/evaluations.npz"
OUT_DIR = Path("results/plots/extra")

COLORS = {
    "SAC": "#01696f",
    "TD3": "#006494",
    "DDPG": "#a12c7b",
    "PPO": "#437a22",
}

FINAL_WINDOW_FRAC = 0.10

def load_algo(algo: str) -> dict | None:
    paths = sorted(glob.glob(PATTERN.format(algo=algo.lower())))
    if not paths:
        print(f"[skip] {algo}: no evaluations found")
        return None
    rewards, lengths, seed_ids = [], [], []
    timesteps = None
    for path in paths:
        seed_id = int(Path(path).parent.name.split("seed")[-1])
        seed_ids.append(seed_id)
        data = np.load(path, allow_pickle=True)
        rewards.append(data["results"].mean(axis=1))
        lengths.append(data["ep_lengths"].mean(axis=1))
        timesteps = data["timesteps"]
    return {
        "timesteps": np.asarray(timesteps),
        "rewards": np.asarray(rewards),
        "lengths": np.asarray(lengths),
        "seeds": seed_ids,
    }


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or x.shape[-1] < window:
        return x
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
    return np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="valid"), -1, padded
    )[..., : x.shape[-1]]


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR/f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR/name}.[png|pdf]")


def plot_smoothed_all(payloads: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, p in payloads.items():
        sm = smooth(p["rewards"], window=5)
        mean = sm.mean(axis=0)
        std = sm.std(axis=0)
        ax.plot(p["timesteps"], mean, label=algo, lw=2.2, color=COLORS[algo])
        ax.fill_between(
            p["timesteps"], mean - std, mean + std,
            alpha=0.15, color=COLORS[algo],
        )
    ax.set_title("HalfCheetah-v5: Smoothed mean reward (window=5)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean episode reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "smoothed_all_algorithms")


def plot_per_seed(algo: str, payload: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    base = COLORS[algo]
    for i, seed in enumerate(payload["seeds"]):
        ax.plot(
            payload["timesteps"],
            payload["rewards"][i],
            lw=1.4,
            color=base,
            alpha=0.45 + 0.2 * i,  
            label=f"seed {seed}",
        )
    mean = payload["rewards"].mean(axis=0)
    ax.plot(payload["timesteps"], mean, lw=2.5, color="black", label="mean")
    ax.set_title(f"HalfCheetah-v5: {algo} per-seed learning curves")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean episode reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, f"{algo.lower()}_per_seed")


def plot_final_bar(payloads: dict) -> None:
    algos = list(payloads.keys())
    finals_mean, finals_std, seed_dots = [], [], []
    for algo in algos:
        rew = payloads[algo]["rewards"]
        tail = max(1, int(rew.shape[1] * FINAL_WINDOW_FRAC))
        seed_finals = rew[:, -tail:].mean(axis=1)
        finals_mean.append(seed_finals.mean())
        finals_std.append(seed_finals.std(ddof=0))
        seed_dots.append(seed_finals)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(algos))
    bar_colors = [COLORS[a] for a in algos]
    ax.bar(xs, finals_mean, yerr=finals_std, capsize=6,
           color=bar_colors, alpha=0.75, edgecolor="black")
    for i, dots in enumerate(seed_dots):
        ax.scatter([i] * len(dots), dots, color="black", zorder=3, s=22)
    ax.set_xticks(xs)
    ax.set_xticklabels(algos)
    ax.set_ylabel("Final mean reward (last 10% of evals)")
    ax.set_title("Final performance ± across-seed std (dots = individual seeds)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, "final_performance_bar")


def plot_sample_efficiency(payloads: dict) -> None:
    global_best = max(
        p["rewards"][:, -max(1, int(p["rewards"].shape[1] * FINAL_WINDOW_FRAC)):]
        .mean(axis=1).mean()
        for p in payloads.values()
    )
    thresholds = [0.50, 0.80]
    bar_data: dict = {thr: [] for thr in thresholds}
    algos = list(payloads.keys())
    for algo in algos:
        sm_mean = smooth(payloads[algo]["rewards"], 5).mean(axis=0)
        ts = payloads[algo]["timesteps"]
        for thr in thresholds:
            target = thr * global_best
            above = np.where(sm_mean >= target)[0]
            bar_data[thr].append(int(ts[above[0]]) if above.size else np.nan)

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    xs = np.arange(len(algos))
    for j, thr in enumerate(thresholds):
        offset = (j - 0.5) * width
        vals = bar_data[thr]
        plot_vals = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(
            xs + offset, plot_vals, width=width,
            label=f"reach {int(thr * 100)}% of best",
            color="#888" if thr == 0.5 else "#222",
            edgecolor="black",
        )
        for x, v in zip(xs + offset, vals):
            if np.isnan(v):
                ax.text(x, ax.get_ylim()[1] * 0.02, "n/a",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(algos)
    ax.set_ylabel("Timesteps to reach threshold")
    ax.set_title(
        f"Sample efficiency — lower is better "
        f"(100% baseline = {global_best:,.0f})"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    save(fig, "sample_efficiency")


def plot_stability(payloads: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo, p in payloads.items():
        std = p["rewards"].std(axis=0, ddof=0)
        ax.plot(p["timesteps"], std, label=algo, lw=2, color=COLORS[algo])
    ax.set_title("Across-seed standard deviation over training")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Std of mean eval reward across seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "stability_over_time")


def plot_ep_length(payloads: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo, p in payloads.items():
        mean_len = p["lengths"].mean(axis=0)
        ax.plot(p["timesteps"], mean_len, label=algo, lw=2, color=COLORS[algo])
    ax.axhline(1000, color="grey", lw=1, ls="--", label="time-limit (1000)")
    ax.set_title("Mean episode length during evaluation")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode length (steps)")
    ax.set_ylim(0, 1100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "episode_length_curves")


def plot_best_so_far(payloads: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo, p in payloads.items():
        sm_mean = smooth(p["rewards"], 5).mean(axis=0)
        cummax = np.maximum.accumulate(sm_mean)
        ax.plot(p["timesteps"], cummax, label=algo, lw=2, color=COLORS[algo])
    ax.set_title("Best-so-far mean reward (peak performance over training)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Best smoothed mean reward up to step t")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "best_so_far")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {a: load_algo(a) for a in ALGOS}
    payloads = {a: p for a, p in payloads.items() if p is not None}
    if not payloads:
        raise SystemExit("No evaluation files found. Train models first.")

    print("Plotting smoothed combined curve...")
    plot_smoothed_all(payloads)

    print("Plotting per-seed curves...")
    for algo, p in payloads.items():
        plot_per_seed(algo, p)

    print("Plotting final-performance bar chart...")
    plot_final_bar(payloads)

    print("Plotting sample efficiency...")
    plot_sample_efficiency(payloads)

    print("Plotting across-seed stability...")
    plot_stability(payloads)

    print("Plotting episode lengths...")
    plot_ep_length(payloads)

    print("Plotting best-so-far curves...")
    plot_best_so_far(payloads)

    print(f"\nAll plots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
