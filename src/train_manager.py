"""Training entry for the high-level manager policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import Adam
import yaml

from src.agents.controller import ControllerConfig, ControllerPolicy
from src.agents.manager import ManagerConfig, ManagerPolicy
from src.algos.ppo_manager import (
    ManagerPPOConfig,
    ManagerRolloutBuffer,
    manager_ppo_update,
)
from src.envs.three_d_pursuit import ThreeDPursuitEnv
from src.utils.logger import Logger


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict) -> ThreeDPursuitEnv:
    return ThreeDPursuitEnv(cfg)


def evaluate_manager(
    make_env_fn: Callable[[], ThreeDPursuitEnv],
    policy: ManagerPolicy,
    device: torch.device,
    *,
    controller: Optional[ControllerPolicy] = None,
    controller_deterministic: bool = True,
    episodes: int = 50,
    seed: int = 0,
) -> float:
    success = 0
    policy.eval()
    with torch.no_grad():
        for ep in range(episodes):
            env = make_env_fn()
            obs_dict = env.reset(seed=seed + ep)
            done = False
            while not done:
                need_update = ((env.t + 1) % max(1, env.manager_period)) == 0
                if env.manager_mode == "learned" and need_update:
                    obs = torch.from_numpy(env.get_manager_observation()).float().to(device)
                    mask = torch.from_numpy(env.get_manager_action_mask()).float().to(device)
                    action, _, _ = policy.act(obs, action_mask=mask, deterministic=True)
                    env.set_manager_action(action.squeeze(0).cpu().numpy())
                if controller is not None:
                    low_obs = torch.from_numpy(obs_dict["low"]).float().to(device)
                    ctrl_action, _, _ = controller.act(
                        low_obs, deterministic=controller_deterministic
                    )
                    action_np = ctrl_action.cpu().numpy()
                else:
                    action_np = np.zeros((env.nD, 3), dtype=np.float32)

                obs_dict, _, done, info = env.step(action_np)
            if info.get("success") or info.get("done_success"):
                success += 1
    policy.train()
    return success / max(1, episodes)


def main(args):
    cfg = load_config(args.config)
    cfg.setdefault("manager", {})
    cfg["manager"]["mode"] = "learned"

    device_name = cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    out_dir = cfg.get("manager_out_dir", os.path.join(cfg.get("out_dir", "runs/default"), "manager"))
    os.makedirs("ckpts", exist_ok=True)
    logger = Logger(out_dir)

    env = make_env(cfg)
    obs_dim = env.get_manager_observation().shape[0]
    manager_hidden = int(cfg.get("manager", {}).get("hidden", 256))
    manager_cfg = ManagerConfig(nD=env.nD, nP=env.nP, hidden=manager_hidden, device=str(device))
    policy = ManagerPolicy(manager_cfg, obs_dim=obs_dim).to(device)

    train_cfg_dict = cfg.get("manager_train", {})

    controller_ckpt = train_cfg_dict.get("controller_ckpt") or train_cfg_dict.get(
        "controller_weights"
    )
    controller: ControllerPolicy | None = None
    if controller_ckpt is not None:
        controller_path = Path(controller_ckpt)
        if not controller_path.exists():
            raise FileNotFoundError(
                f"Controller checkpoint not found: {controller_path}. "
                "Train the residual controller first and set manager_train.controller_ckpt."
            )
        ctrl_cfg = ControllerConfig(obs_dim=env.low_obs_dim, device=str(device))
        controller = ControllerPolicy(ctrl_cfg).to(device)
        state_dict = torch.load(controller_path, map_location=device)
        controller.load_state_dict(state_dict)
        controller.eval()
    else:
        raise ValueError(
            "manager_train.controller_ckpt is required so the manager trains on top of the learned controller."
        )

    controller_deterministic = bool(train_cfg_dict.get("controller_deterministic", True))
    ppo_cfg = ManagerPPOConfig(
        gamma=float(train_cfg_dict.get("gamma", 0.99)),
        gae_lambda=float(train_cfg_dict.get("gae_lambda", 0.95)),
        clip_eps=float(train_cfg_dict.get("clip_eps", 0.2)),
        lr=float(train_cfg_dict.get("lr", 3e-4)),
        epochs=int(train_cfg_dict.get("epochs", 4)),
        batch_size=int(train_cfg_dict.get("batch_size", 256)),
        value_coef=float(train_cfg_dict.get("value_coef", 0.5)),
        entropy_coef=float(train_cfg_dict.get("entropy_coef", 0.005)),
        max_grad_norm=float(train_cfg_dict.get("max_grad_norm", 0.5)),
        target_kl=float(train_cfg_dict.get("target_kl", 0.1)),
        adv_norm_eps=float(train_cfg_dict.get("adv_norm_eps", 1e-8)),
        device=str(device),
        teacher_coef=float(train_cfg_dict.get("teacher_coef", 0.4)),
    )
    optimizer = Adam(policy.parameters(), lr=ppo_cfg.lr)

    updates = int(train_cfg_dict.get("updates", 300))
    horizon = int(train_cfg_dict.get("horizon", 256))
    teacher_mix_start = float(train_cfg_dict.get("teacher_mixing_start", 1.0))
    teacher_mix_end = float(train_cfg_dict.get("teacher_mixing_end", 0.2))
    teacher_mix_decay = int(train_cfg_dict.get("teacher_mixing_decay", updates))

    def teacher_mix_schedule(step: int) -> float:
        if teacher_mix_decay <= 0:
            return float(teacher_mix_end)
        frac = min(1.0, max(0.0, step / float(teacher_mix_decay)))
        return float(teacher_mix_start + (teacher_mix_end - teacher_mix_start) * frac)

    rng = np.random.default_rng(cfg.get("seed", 0) + 2025)

    best_sr = 0.0
    for upd in range(1, updates + 1):
        buf = ManagerRolloutBuffer()
        obs = env.reset(seed=cfg.get("seed", 0) + upd)
        done = False
        manager_steps = 0
        teacher_used = 0
        ep_reward = 0.0

        mix_prob = float(teacher_mix_schedule(upd - 1))

        while not done and manager_steps < horizon:
            need_update = ((env.t + 1) % max(1, env.manager_period)) == 0
            if env.manager_mode == "learned" and need_update:
                obs_np = env.get_manager_observation()
                mask_np = env.get_manager_action_mask()
                teacher_action = env.compute_rule_manager_action()
                obs_t = torch.from_numpy(obs_np).float().to(device)
                mask_t = torch.from_numpy(mask_np).float().to(device)
                action_t, logp_t, value_t = policy.act(
                    obs_t, action_mask=mask_t, deterministic=False
                )
                action_np = action_t.squeeze(0).cpu().numpy()

                use_teacher = bool(rng.random() < mix_prob)
                if use_teacher:
                    teacher_tensor = torch.from_numpy(teacher_action).long().unsqueeze(0).to(device)
                    exec_logp, _, _, _ = policy.evaluate_actions(
                        obs_t.unsqueeze(0), teacher_tensor, action_mask=mask_t.unsqueeze(0)
                    )
                    logp_val = exec_logp.squeeze(0)
                    action_exec = teacher_action.astype(np.int32)
                    teacher_used += 1
                else:
                    logp_val = logp_t.squeeze(0)
                    action_exec = action_np.astype(np.int32)

                env.set_manager_action(action_exec)
                buf.add(
                    obs_np,
                    action_exec,
                    float(logp_val.item()),
                    value_t.item(),
                    mask_np,
                    teacher_action.astype(np.int32),
                )
                manager_steps += 1

            if controller is not None:
                with torch.no_grad():
                    low_obs = torch.from_numpy(obs["low"]).float().to(device)
                    ctrl_action, _, _ = controller.act(
                        low_obs, deterministic=controller_deterministic
                    )
                action_residual = ctrl_action.cpu().numpy()
            else:
                action_residual = np.zeros((env.nD, 3), dtype=np.float32)

            obs, reward, done, _ = env.step(action_residual)
            buf.add_reward(reward, done)
            ep_reward += reward

        if buf.obs:
            if done:
                buf.set_last_value(0.0)
            else:
                obs_np = env.get_manager_observation()
                mask_np = env.get_manager_action_mask()
                obs_t = torch.from_numpy(obs_np).float().to(device)
                mask_t = torch.from_numpy(mask_np).float().to(device)
                with torch.no_grad():
                    _, _, last_value_t = policy.act(
                        obs_t, action_mask=mask_t, deterministic=True
                    )
                buf.set_last_value(last_value_t.item())

        if not buf.obs:
            continue

        data = buf.cat()
        stats = manager_ppo_update(policy, optimizer, data, ppo_cfg)

        logger.log_scalar("train/ep_return", float(ep_reward), upd)
        logger.log_scalar("loss/policy", stats["policy_loss"], upd)
        logger.log_scalar("loss/value", stats["value_loss"], upd)
        logger.log_scalar("loss/entropy", stats["entropy"], upd)
        logger.log_scalar("loss/imitation", stats["imitation_loss"], upd)
        logger.log_scalar("debug/approx_kl", stats["approx_kl"], upd)
        logger.log_scalar("debug/approx_kl_last", stats["approx_kl_last"], upd)
        logger.log_scalar("debug/kl_stop", 1.0 if stats["kl_stop"] else 0.0, upd)
        logger.log_scalar("train/teacher_mix_prob", mix_prob, upd)
        ratio = teacher_used / max(1, manager_steps)
        logger.log_scalar("train/teacher_mix_ratio", float(ratio), upd)

        if upd % int(train_cfg_dict.get("eval_every", 20)) == 0:
            sr = evaluate_manager(
                lambda: make_env(cfg),
                policy,
                device,
                controller=controller,
                controller_deterministic=controller_deterministic,
                episodes=int(train_cfg_dict.get("eval_episodes", 100)),
                seed=123,
            )
            logger.log_scalar("eval/success", float(sr), upd)
            if sr >= best_sr:
                best_sr = sr
                torch.save(policy.state_dict(), os.path.join("ckpts", "manager_best.pt"))

        logger.flush()
        torch.save(policy.state_dict(), os.path.join("ckpts", "manager_latest.pt"))

    final_sr = evaluate_manager(
        lambda: make_env(cfg),
        policy,
        device,
        controller=controller,
        controller_deterministic=controller_deterministic,
        episodes=int(train_cfg_dict.get("final_eval_episodes", 300)),
        seed=2025,
    )
    print(f"Final manager eval over {int(train_cfg_dict.get('final_eval_episodes', 300))} eps: {final_sr:.3f} | Best: {best_sr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train high-level manager policy")
    parser.add_argument("--config", type=str, default="src/configs/default.yaml")
    args = parser.parse_args()
    main(args)
