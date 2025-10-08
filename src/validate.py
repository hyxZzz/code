"""Utility script to evaluate saved checkpoints over multiple validation episodes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch

from src.agents.controller import ControllerConfig, ControllerPolicy
from src.envs.three_d_pursuit import ThreeDPursuitEnv


DEFAULT_EPISODES = 1000
DEFAULT_SEED = 42
CHECKPOINT_NAMES = {"best", "latest"}


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict) -> ThreeDPursuitEnv:
    return ThreeDPursuitEnv(cfg)


def quick_eval(make_env_fn, policy: ControllerPolicy, device: torch.device, *, episodes: int, seed: int) -> float:
    """Evaluate a policy and return the success rate over ``episodes`` trials."""
    successes = 0
    for episode in range(episodes):
        env = make_env_fn()
        obs = env.reset(seed=seed + episode)
        done = False
        success = False
        last_reward = 0.0

        while not done:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs["low"]).float().to(device)
                act_t, _, _ = policy.act(obs_t, deterministic=True)
            obs, reward, done, info = env.step(act_t.cpu().numpy())
            last_reward = reward

            if info.get("success") or info.get("done_success"):
                success = True
                break
            if "attackers_alive" in info and int(info["attackers_alive"]) == 0:
                success = True
                break

        if not success and last_reward > 0:
            success = True

        successes += int(success)

    return successes / float(episodes)


def build_policy(cfg: dict, device: torch.device) -> ControllerPolicy:
    env = make_env(cfg)
    obs_dim = env.low_obs_dim
    policy_cfg = ControllerConfig(obs_dim=obs_dim, device=device)
    policy = ControllerPolicy(policy_cfg).to(device)
    return policy


def resolve_checkpoints(requested: Iterable[str]) -> List[str]:
    resolved = []
    for name in requested:
        if name not in CHECKPOINT_NAMES:
            raise ValueError(f"Unknown checkpoint '{name}'. Expected one of {sorted(CHECKPOINT_NAMES)}")
        if name not in resolved:
            resolved.append(name)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate saved policy checkpoints over multiple episodes.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/default.yaml",
        help="Path to the training configuration file used to create the checkpoints.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Number of validation episodes to run per checkpoint (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed for environment resets.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        choices=sorted(CHECKPOINT_NAMES),
        nargs="+",
        default=["best", "latest"],
        help="Checkpoint name(s) to evaluate. Defaults to evaluating both best and latest.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Optional override for the checkpoint directory. Defaults to '../ckpts' relative to this file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_device = cfg.get("device", "auto")
    if cfg_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg_device)

    policy = build_policy(cfg, device)

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(__file__).resolve().parent.parent / "ckpts"
    checkpoints = resolve_checkpoints(args.checkpoint)

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    make_env_fn = lambda: make_env(cfg)

    for name in checkpoints:
        ckpt_path = ckpt_dir / f"{name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(state_dict)
        policy.eval()

        success_rate = quick_eval(
            make_env_fn,
            policy,
            device,
            episodes=int(args.episodes),
            seed=int(args.seed),
        )
        print(f"{name} checkpoint success rate over {args.episodes} episodes: {success_rate:.4f}")


if __name__ == "__main__":
    main()