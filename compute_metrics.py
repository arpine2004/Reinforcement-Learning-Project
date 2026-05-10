from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ALGOS = ["SAC", "TD3", "DDPG", "PPO"]
PATTERN = "results/{algo}_seed*/evaluations.npz"

FINAL_WINDOW_FRAC = 0.10
EFFICIENCY_THRESHOLDS = (0.50, 0.80)


def load_algo(algo: str) -> dict | None:
    """Load all seed runs for one algorithm. Returns None if no files found."""
    paths = sorted(glob.glob(PATTERN.format(algo=algo.lower())))
    if not paths:
        print(f"[skip] {algo}: no evaluations found")
        return None

    seed_curves = []  # mean reward over eval episodes, per eval point
    seed_lengths = []  # mean ep length over eval episodes
    timesteps = None
    seed_ids = []

    for path in paths:
        seed_id = int(Path(path).parent.name.split("seed")[-1])
        seed_ids.append(seed_id)

        data = np.load(path, allow_pickle=True)
        seed_curves.append(data["results"].mean(axis=1))
        seed_lengths.append(data["ep_lengths"].mean(axis=1))
        timesteps = data["timesteps"]

    return {
        "timesteps": np.asarray(timesteps),
        "rewards": np.asarray(seed_curves),   # shape (n_seeds, n_evals)
        "lengths": np.asarray(seed_lengths),  # shape (n_seeds, n_evals)
        "seeds": seed_ids,
    }


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average; mirrors the kind of smoothing TensorBoard uses."""
    if window <= 1 or x.shape[-1] < window:
        return x
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
    return np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="valid"), -1, padded
    )[..., : x.shape[-1]]


def steps_to_threshold(
    timesteps: np.ndarray, smoothed_mean: np.ndarray, target: float
) -> float | None:
    """First timestep at which the smoothed mean curve crosses `target`."""
    above = np.where(smoothed_mean >= target)[0]
    if above.size == 0:
        return None
    return int(timesteps[above[0]])


def per_algo_metrics(algo: str, payload: dict, global_best_final: float) -> dict:
    rewards = payload["rewards"]            # (n_seeds, n_evals)
    lengths = payload["lengths"]
    timesteps = payload["timesteps"]
    n_seeds, n_evals = rewards.shape

    tail = max(1, int(n_evals * FINAL_WINDOW_FRAC))
    seed_final = rewards[:, -tail:].mean(axis=1)              

    smoothed = smooth(rewards, window=5)
    seed_best = smoothed.max(axis=1)

    mean_curve = smoothed.mean(axis=0)

    metrics: dict = {
        "algorithm": algo,
        "n_seeds": n_seeds,
        "final_mean_reward": float(seed_final.mean()),
        "final_std_seeds": float(seed_final.std(ddof=0)),
        "final_cv_seeds": float(
            seed_final.std(ddof=0) / max(abs(seed_final.mean()), 1e-9)
        ),
        "best_mean_reward": float(seed_best.mean()),
        "worst_seed_final": float(seed_final.min()),
        "best_seed_final": float(seed_final.max()),
        "auc_normalized": float(
            mean_curve.sum() / (n_evals * max(global_best_final, 1e-9))
        ),
        "final_ep_length": float(lengths[:, -tail:].mean()),
    }
    for thr in EFFICIENCY_THRESHOLDS:
        target = thr * global_best_final
        steps = steps_to_threshold(timesteps, mean_curve, target)
        metrics[f"steps_to_{int(thr * 100)}pct"] = steps  
    return metrics


def per_seed_rows(algo: str, payload: dict) -> list[dict]:
    rewards = payload["rewards"]
    lengths = payload["lengths"]
    n_evals = rewards.shape[1]
    tail = max(1, int(n_evals * FINAL_WINDOW_FRAC))

    rows = []
    for i, seed in enumerate(payload["seeds"]):
        rows.append(
            {
                "algorithm": algo,
                "seed": seed,
                "final_reward": float(rewards[i, -tail:].mean()),
                "best_reward": float(smooth(rewards[i:i + 1], 5).max()),
                "final_ep_length": float(lengths[i, -tail:].mean()),
                "n_evals": n_evals,
            }
        )
    return rows


def write_markdown_summary(df: pd.DataFrame, path: Path) -> None:
    cols = [
        ("algorithm", "Algorithm"),
        ("final_mean_reward", "Final reward (mean ± std over seeds)"),
        ("best_mean_reward", "Best reward (smoothed)"),
        ("auc_normalized", "AUC (normalized)"),
        ("steps_to_50pct", "Steps to 50% of best"),
        ("steps_to_80pct", "Steps to 80% of best"),
        ("final_cv_seeds", "Across-seed CV"),
        ("final_ep_length", "Final ep length"),
    ]
    lines = ["# HalfCheetah-v5 Benchmark Metrics", ""]
    lines.append(
        "All metrics aggregated over 3 random seeds per algorithm. "
        "Final reward = mean over the last 10% of evaluation points; "
        "AUC = area under the (smoothed, mean-across-seeds) learning curve, "
        "normalized so 1.0 corresponds to instantly hitting the global best "
        "final reward. Sample efficiency thresholds are fractions of the best "
        "algorithm's final reward."
    )
    lines.append("")
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines.extend([header, sep])
    for _, row in df.iterrows():
        cells = []
        for key, _ in cols:
            v = row[key]
            if key == "final_mean_reward":
                cells.append(f"{v:,.0f} ± {row['final_std_seeds']:,.0f}")
            elif key == "algorithm":
                cells.append(str(v))
            elif key in ("steps_to_50pct", "steps_to_80pct"):
                cells.append("—" if pd.isna(v) else f"{int(v):,}")
            elif key == "final_ep_length":
                cells.append(f"{v:.0f}")
            elif key == "final_cv_seeds":
                cells.append(f"{v:.3f}")
            elif key == "auc_normalized":
                cells.append(f"{v:.3f}")
            else:
                cells.append(f"{v:,.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "**Reading the table.** *Final reward* is the headline performance "
        "number; the ± span is the spread across seeds (smaller = more "
        "stable). *Best reward* shows peak performance during training and "
        "is useful for spotting algorithms that learned but then collapsed. "
        "*AUC* rewards algorithms that learn fast and stay high. "
        "*Steps-to-X%* are the canonical sample-efficiency numbers — lower "
        "is better. *Across-seed CV* is the coefficient of variation of the "
        "final reward across seeds (lower = more reproducible). *Final ep "
        "length* should be 1000 (the HalfCheetah time limit) for a healthy "
        "policy; lower means the agent is somehow terminating early."
    )
    path.write_text("\n".join(lines))


def main() -> None:
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {algo: load_algo(algo) for algo in ALGOS}
    payloads = {a: p for a, p in payloads.items() if p is not None}
    if not payloads:
        raise SystemExit("No evaluation files found in results/. Train first.")

    global_best = max(
        p["rewards"][:, -max(1, int(p["rewards"].shape[1] * FINAL_WINDOW_FRAC)):]
        .mean(axis=1).mean()
        for p in payloads.values()
    )
    print(f"Global best final reward (used as 100% baseline): {global_best:,.1f}")

    rows = [per_algo_metrics(a, p, global_best) for a, p in payloads.items()]
    df = pd.DataFrame(rows).sort_values("final_mean_reward", ascending=False)

    seed_rows = []
    for algo, payload in payloads.items():
        seed_rows.extend(per_seed_rows(algo, payload))
    seed_df = pd.DataFrame(seed_rows).sort_values(["algorithm", "seed"])

    df.to_csv(out_dir / "metrics.csv", index=False)
    seed_df.to_csv(out_dir / "metrics_per_seed.csv", index=False)
    write_markdown_summary(df, out_dir / "metrics_summary.md")

    print("\n=== Aggregate metrics ===")
    print(df.to_string(index=False))
    print("\n=== Per-seed metrics ===")
    print(seed_df.to_string(index=False))
    print(f"\nWrote: {out_dir / 'metrics.csv'}")
    print(f"Wrote: {out_dir / 'metrics_per_seed.csv'}")
    print(f"Wrote: {out_dir / 'metrics_summary.md'}")


if __name__ == "__main__":
    main()
