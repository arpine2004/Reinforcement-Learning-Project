import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "HalfCheetah-v5"
ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGOS.keys()), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    model_path = Path(f"checkpoints/{args.algo}_seed{args.seed}/best_model.zip")
    if not model_path.exists():
        model_path = Path(f"checkpoints/{args.algo}_seed{args.seed}/final_model.zip")
    if not model_path.exists():
        raise FileNotFoundError(f"No model found for {args.algo}, seed {args.seed}")

    env = gym.make(ENV_ID, render_mode=None)
    env.reset(seed=args.seed + 100)
    model = ALGOS[args.algo].load(model_path)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=True,
        return_episode_rewards=False,
    )
    print({
        "algo": args.algo,
        "seed": args.seed,
        "episodes": args.episodes,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
    })
    env.close()


if __name__ == "__main__":
    main()