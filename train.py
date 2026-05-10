import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

ENV_ID = "HalfCheetah-v5"


def make_env(seed: int, monitor_dir: str | None = None):
    env = gym.make(ENV_ID)
    if monitor_dir is not None:
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, monitor_dir)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def build_model(algo: str, seed: int, tb_log: str):
    algo = algo.lower()

    if algo == "sac":
        env = make_env(seed)
        model = SAC(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            tensorboard_log=tb_log,
            learning_starts=10_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
        )
        eval_env = make_env(seed + 10_000, monitor_dir=f"results/{algo}_seed{seed}/eval_monitor")

    elif algo == "td3":
        env = make_env(seed)
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        model = TD3(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            tensorboard_log=tb_log,
            action_noise=action_noise,
            learning_starts=10_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
        )
        eval_env = make_env(seed + 10_000, monitor_dir=f"results/{algo}_seed{seed}/eval_monitor")

    elif algo == "ddpg":
        env = make_env(seed)
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        model = DDPG(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            tensorboard_log=tb_log,
            action_noise=action_noise,
            learning_starts=10_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
        )
        eval_env = make_env(seed + 10_000, monitor_dir=f"results/{algo}_seed{seed}/eval_monitor")

    elif algo == "ppo":
        env = make_vec_env(ENV_ID, n_envs=4, seed=seed)
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            tensorboard_log=tb_log,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.0,
        )
        eval_env = make_env(seed + 10_000, monitor_dir=f"results/{algo}_seed{seed}/eval_monitor")

    else:
        raise ValueError(f"Unsupported algo: {algo}")

    return model, eval_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sac", "td3", "ppo", "ddpg"], required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    for seed in args.seeds:
        tb_log = f"logs/{args.algo}"

        best_path = f"checkpoints/{args.algo}_seed{seed}"
        result_path = f"results/{args.algo}_seed{seed}"
        os.makedirs(best_path, exist_ok=True)
        os.makedirs(result_path, exist_ok=True)

        model, eval_env = build_model(args.algo, seed, tb_log)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_path,
            log_path=result_path,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )

        model.learn(
            total_timesteps=args.timesteps,
            callback=eval_callback,
            tb_log_name=f"{args.algo}_seed{seed}",
        )
        model.save(f"{best_path}/final_model")

        model.env.close()
        eval_env.close()


if __name__ == "__main__":
    main()


    