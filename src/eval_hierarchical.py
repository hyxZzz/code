"""Evaluate the full hierarchical stack (manager + controller) over many episodes."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import yaml

from src.agents.controller import ControllerConfig, ControllerPolicy
from src.agents.manager import ManagerConfig, ManagerPolicy
from src.envs.three_d_pursuit import ThreeDPursuitEnv


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict) -> ThreeDPursuitEnv:
    return ThreeDPursuitEnv(cfg)


def build_controller(cfg: dict, device: torch.device) -> ControllerPolicy:
    env = make_env(cfg)
    obs_dim = env.low_obs_dim
    policy = ControllerPolicy(ControllerConfig(obs_dim=obs_dim, device=str(device))).to(device)
    return policy


def build_manager(cfg: dict, device: torch.device) -> tuple[ManagerPolicy, int, int, int]:
    env = make_env(cfg)
    obs_dim = env.get_manager_observation().shape[0]
    manager_hidden = int(cfg.get("manager", {}).get("hidden", 256))
    manager_cfg = ManagerConfig(nD=env.nD, nP=env.nP, hidden=manager_hidden, device=str(device))
    policy = ManagerPolicy(manager_cfg, obs_dim=obs_dim).to(device)
    return policy, env.nD, env.nP, obs_dim


def rollout_episode(
    make_env_fn,
    controller: ControllerPolicy,
    manager: ManagerPolicy,
    device: torch.device,
    *,
    seed: int,
    deterministic: bool,
) -> bool:
    env: ThreeDPursuitEnv = make_env_fn()
    obs = env.reset(seed=seed)
    done = False
    success = False
    last_reward = 0.0

    while not done:
        need_update = ((env.t + 1) % max(1, env.manager_period)) == 0
        if env.manager_mode == "learned" and need_update:
            with torch.no_grad():
                mgr_obs = torch.from_numpy(env.get_manager_observation()).float().to(device)
                mgr_mask = torch.from_numpy(env.get_manager_action_mask()).float().to(device)
                mgr_action, _, _ = manager.act(mgr_obs, action_mask=mgr_mask, deterministic=deterministic)
            env.set_manager_action(mgr_action.squeeze(0).cpu().numpy())

        with torch.no_grad():
            low_obs = torch.from_numpy(obs["low"]).float().to(device)
            ctrl_action, _, _ = controller.act(low_obs, deterministic=deterministic)
        obs, reward, done, info = env.step(ctrl_action.cpu().numpy())
        last_reward = reward

        if info.get("success") or info.get("done_success"):
            success = True
            break
        if int(info.get("attackers_alive", 1)) == 0:
            success = True
            break

    if not success and last_reward > 0:
        success = True
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate controller + manager checkpoints together.")
    parser.add_argument("--config", type=str, default="src/configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--controller-ckpt",
        type=str,
        default="ckpts/best.pt",
        help="Path to the residual controller checkpoint (default: ckpts/best.pt).",
    )
    parser.add_argument(
        "--manager-ckpt",
        type=str,
        default="ckpts/manager_best.pt",
        help="Path to the high-level manager checkpoint (default: ckpts/manager_best.pt).",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Number of evaluation episodes (default: 200).")
    parser.add_argument("--seed", type=int, default=2025, help="Base seed for episode resets (default: 2025).")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample both controller and manager actions stochastically (default: deterministic).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval.setdefault("manager", {})["mode"] = "learned"

    cfg_device = cfg_eval.get("device", "auto")
    if cfg_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg_device)

    controller = build_controller(cfg_eval, device)
    controller_ckpt = Path(args.controller_ckpt)
    if not controller_ckpt.exists():
        raise FileNotFoundError(f"Controller checkpoint not found: {controller_ckpt}")
    controller.load_state_dict(torch.load(controller_ckpt, map_location=device))
    controller.eval()

    manager = build_manager(cfg_eval, device)[0]
    manager_ckpt = Path(args.manager_ckpt)
    if not manager_ckpt.exists():
        raise FileNotFoundError(f"Manager checkpoint not found: {manager_ckpt}")
    m_state = torch.load(manager_ckpt, map_location=device)
    missing, unexpected = manager.load_state_dict(m_state, strict=False)
    if missing or unexpected:
        print(
            "[eval] Warning: manager checkpoint loaded with partial match. "
            f"Missing: {missing} | Unexpected: {unexpected}"
        )
    manager.eval()

    def make_eval_env() -> ThreeDPursuitEnv:
        return make_env(cfg_eval)

    successes = 0
    for ep in range(args.episodes):
        seed = args.seed + ep
        success = rollout_episode(
            make_eval_env,
            controller,
            manager,
            device,
            seed=seed,
            deterministic=not args.stochastic,
        )
        successes += int(success)

    sr = successes / float(max(1, args.episodes))
    mode = "stochastic" if args.stochastic else "deterministic"
    print(
        f"Hierarchical stack success rate over {args.episodes} episodes ({mode} actions): {sr:.4f}"
    )


if __name__ == "__main__":
    main()

