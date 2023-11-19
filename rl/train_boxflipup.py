"""
Train a policy for manipuolation.gym.envs.box_flipup
"""

import argparse
import os
import sys

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
)
import wandb
from wandb.integration.sb3 import WandbCallback

# `multiprocessing` also provides this method, but empirically `psutil`'s
# version seems more reliable.
from psutil import cpu_count

from pydrake.all import StartMeshcat

import manipulation.envs.box_flipup  # no-member


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train_single_env", action="store_true")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--log_path",
        help="path to the logs directory.",
        default="/tmp/BoxFlipUp/",
    )
    args = parser.parse_args()

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 5e5 if not args.test else 5,
        "env_name": "BoxFlipUp-v0",
        "env_time_limit": 10 if not args.test else 0.5,
        "local_log_dir": args.log_path,
        "observations": "state",
    }

    if args.wandb:
        run = wandb.init(
            project=config["env_name"],
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload videos
            save_code=True,
        )
    else:
        run = wandb.init(mode="disabled")

    zip = f"data/box_flipup_ppo_{config['observations']}.zip"

    num_cpu = int(cpu_count() / 2) if not args.test else 2
    if args.train_single_env:
        meshcat = StartMeshcat()
        env = gym.make(
            config["env_name"],
            meshcat=meshcat,
            observations=config["observations"],
            time_limit=config["env_time_limit"],
        )
        check_env(env)
        input("Open meshcat (optional). Press Enter to continue...")
    else:
        env = make_vec_env(
            config["env_name"],
            n_envs=num_cpu,
            seed=0,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={
                "observations": config["observations"],
                "time_limit": config["env_time_limit"],
            },
        )

    if args.test:
        model = PPO(
            config["policy_type"], env, n_steps=4, n_epochs=2, batch_size=8
        )
    elif os.path.exists(zip):
        model = PPO.load(zip, env, verbose=1, tensorboard_log=f"runs/{run.id}")
    else:
        model = PPO(
            config["policy_type"],
            env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
        )

    # Separate evaluation env.
    eval_env = gym.make(
        config["env_name"],
        observations=config["observations"],
        time_limit=config["env_time_limit"],
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    # Record a video every n evaluation rollouts.
    n = 1
    eval_env = VecVideoRecorder(
        eval_env,
        log_dir + f"videos/test",
        record_video_trigger=lambda x: x % n == 0,
        video_length=100,
    )
    # Use deterministic actions for evaluation.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + f"eval_logs/test",
        log_path=log_dir + f"eval_logs/test",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    new_log = True
    while True:
        model.learn(
            total_timesteps=100000 if not args.test else 4,
            reset_num_timesteps=new_log,
            callback=WandbCallback(),
        )
        if args.test:
            break
        model.save(zip)
        new_log = False


if __name__ == "__main__":
    sys.exit(main())
