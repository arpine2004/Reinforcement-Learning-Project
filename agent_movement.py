import gymnasium as gym
from stable_baselines3 import SAC, TD3, PPO, DDPG
import imageio
import numpy as np
from pathlib import Path

ALGORITHMS = {
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "ddpg": DDPG,
}

BASE = Path(__file__).resolve().parent
ROOT = BASE
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
VIDEOS = PLOTS / "videos"
VIDEOS.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]


def eval_seed(algo, seed, n_eval_episodes=5):
    AlgoClass = ALGORITHMS[algo]
    model_path = CHECKPOINTS / f"{algo}_seed{seed}" / "best_model.zip"
    if not model_path.exists():
        return None

    env = gym.make("HalfCheetah-v5")
    model = AlgoClass.load(model_path, env=env)

    episode_returns = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        episode_returns.append(total_reward)

    env.close()
    return float(np.mean(episode_returns))


def choose_best_seed(algo):
    scores = {}

    for seed in SEEDS:
        score = eval_seed(algo, seed)
        if score is not None:
            scores[seed] = score

    if not scores:
        raise FileNotFoundError(f"No best_model.zip files found for {algo}")

    best_seed = max(scores, key=scores.get)
    return best_seed, scores


def record_video(algo, seed, steps=1000):
    AlgoClass = ALGORITHMS[algo]
    model_path = CHECKPOINTS / f"{algo}_seed{seed}" / "best_model.zip"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[INFO] Loading {algo} seed {seed}: {model_path}")
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    model = AlgoClass.load(model_path, env=env)

    obs, _ = env.reset()
    frames = []

    for _ in range(steps):
        frames.append(env.render())

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    env.close()

    outpath = VIDEOS / f"{algo}_seed{seed}_best.gif"
    imageio.mimsave(outpath, frames, fps=30)
    print(f"[OK] Saved video -> {outpath}")


def main():
    summary = {}

    for algo in ["sac", "td3", "ppo", "ddpg"]:
        best_seed, scores = choose_best_seed(algo)
        summary[algo] = {"best_seed": best_seed, "scores": scores}
        print(f"[INFO] {algo}: scores={scores}, best_seed={best_seed}")
        record_video(algo, best_seed)

    lines = ["# Best Seed Selection", ""]
    for algo, info in summary.items():
        lines.append(
            f"- {algo}: best seed = {info['best_seed']}, scores = {info['scores']}"
        )

    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots" / "best_seed_selection.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()