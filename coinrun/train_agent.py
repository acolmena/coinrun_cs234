"""
Train an agent using a PPO2 based on OpenAI Baselines.
"""

import time
import os
import json
import sys
import importlib
from mpi4py import MPI
import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, ppo2
from coinrun.obs_wrappers import PhotometricJitter, ObsTransformWrapper
from coinrun.config import Config

def _import_wandb():
    wandb = importlib.import_module("wandb")
    if hasattr(wandb, "init"):
        return wandb

    # If ./wandb exists in the working directory, Python can import that folder
    # as a namespace package and shadow the real wandb library.
    local_wandb_dir = os.path.join(os.getcwd(), "wandb")
    wandb_path = getattr(wandb, "__path__", None)
    has_local_shadow = (
        wandb_path is not None
        and any(os.path.abspath(p) == os.path.abspath(local_wandb_dir) for p in wandb_path)
    )

    if has_local_shadow:
        original_sys_path = list(sys.path)
        try:
            sys.path = [p for p in sys.path if p not in ("", os.getcwd())]
            if "wandb" in sys.modules:
                del sys.modules["wandb"]
            wandb = importlib.import_module("wandb")
        finally:
            sys.path = original_sys_path

    if not hasattr(wandb, "init"):
        raise ImportError(
            "Imported module 'wandb' does not expose wandb.init. "
            "Ensure the wandb package is installed in the container and "
            "that a local ./wandb directory is not shadowing it."
        )

    return wandb

def main():
    args = setup_utils.setup_and_load()
    safe_args = dict(vars(args))
    safe_args.pop("wandb_api_key", None)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    wandb_enabled = bool(args.wandb and rank == 0)
    wandb_module = None

    logdir = os.path.join(Config.WORKDIR, args.run_id)
    if rank == 0:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "config.json"), "w") as f:
            json.dump(safe_args, f, indent=2)

    if wandb_enabled:
        wandb = _import_wandb()

        if args.wandb_api_key:
            # Works across old/new wandb versions; older py3.5 builds may lack wandb.login.
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
            if hasattr(wandb, "login"):
                try:
                    wandb.login(key=args.wandb_api_key)
                except TypeError:
                    wandb.login(key=args.wandb_api_key, relogin=True)

        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        run_name = args.wandb_name or args.run_id
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            tags=tags,
            mode=args.wandb_mode,
            config=safe_args,
        )
        wandb_module = wandb

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    nenvs = Config.NUM_ENVS
    total_timesteps = int(1e6)
    save_interval = args.save_interval

    env = utils.make_general_env(nenvs, seed=rank)

    with tf.Session(config=config):
        env = wrappers.add_final_wrappers(env)

        if args.jitter_p > 0.0:
            jitter = PhotometricJitter(
                p=args.jitter_p,
                brightness=args.jitter_brightness,
                contrast=args.jitter_contrast,
                seed=args.jitter_seed,
            )
            env = ObsTransformWrapper(env, jitter)

        obs = env.reset()
        print("OBS dtype", obs.dtype, "shape", obs.shape, "min", obs.min(), "max", obs.max())
        o0 = obs[0] if obs.ndim == 4 else obs
        print("OBS[0] dtype", o0.dtype, "min", o0.min(), "max", o0.max())

        policy = policies.get_policy()
        try:
            ppo2.learn(policy=policy,
                        env=env,
                        save_interval=save_interval,
                        nsteps=Config.NUM_STEPS,
                        nminibatches=Config.NUM_MINIBATCHES,
                        lam=0.95,
                        gamma=Config.GAMMA,
                        noptepochs=Config.PPO_EPOCHS,
                        log_interval=1,
                        ent_coef=Config.ENTROPY_COEFF,
                        lr=lambda f : f * Config.LEARNING_RATE,
                        cliprange=lambda f : f * 0.2,
                        total_timesteps=total_timesteps,
                        debug_save_frames=args.debug_save_frames,
                        debug_save_frames_n=args.debug_save_frames_n,
                        debug_save_frames_every=args.debug_save_frames_every,
                        debug_save_dir=os.path.join(logdir, "debug_frames"))
        finally:
            if wandb_enabled and wandb_module is not None:
                wandb_module.finish()

if __name__ == '__main__':
    main()

