# src/train.py
# ---------------------------------------
from __future__ import annotations
import os, argparse, yaml
import numpy as np
import torch
from torch.optim import Adam

from src.envs.three_d_pursuit import ThreeDPursuitEnv
from src.agents.controller import ControllerPolicy, ControllerConfig
from src.algos.ppo import PPOConfig, RolloutBuffer, ppo_update
from src.utils.logger import Logger


def quick_eval(make_env_fn, policy, device, episodes=100, seed=0):
    """
    成功判定优先级：
    1) info["success"] / info["done_success"] 为 True；
    2) info["attackers_alive"] == 0；
    3) 兜底：回合结束时最后一步奖励 > 0。
    """
    sr = 0
    for ep in range(episodes):
        env = make_env_fn()
        obs = env.reset(seed=seed + ep)
        done = False
        success_flag = False
        last_rew = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs["low"]).float().to(next(policy.parameters()).device)
                act_t, _, _ = policy.act(obs_t, deterministic=True)
            obs, rew, done, info = env.step(act_t.cpu().numpy())
            last_rew = rew

            if info.get("success") or info.get("done_success"):
                success_flag = True
                break
            if "attackers_alive" in info and int(info["attackers_alive"]) == 0:
                success_flag = True
                break

        if not success_flag and last_rew > 0:
            success_flag = True
        sr += 1 if success_flag else 0
    return sr / episodes


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg):
    return ThreeDPursuitEnv(cfg)


def main(args):
    cfg = load_config(args.config)

    # device 选择（修复 "auto" 逻辑）
    cfg_device = cfg.get("device", "auto")
    if cfg_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg_device

    out_dir = cfg.get("out_dir", "runs/default")
    os.makedirs("ckpts", exist_ok=True)
    logger = Logger(out_dir)

    # 环境与策略
    env = make_env(cfg)
    obs_dim = env.low_obs_dim
    ctrl_cfg = ControllerConfig(obs_dim=obs_dim, device=device)
    policy = ControllerPolicy(ctrl_cfg).to(device)

    # PPO 配置
    ppo_cfg = PPOConfig(
        gamma=cfg["train"].get("gamma", 0.99),
        gae_lambda=cfg["train"].get("gae_lambda", 0.95),
        clip_eps=cfg["train"].get("clip_eps", 0.2),
        lr=cfg["train"].get("lr", 3e-4),
        epochs=cfg["train"].get("epochs", 4),
        batch_size=cfg["train"].get("batch_size", 8192),
        value_coef=cfg["train"].get("value_coef", 0.5),
        entropy_coef=cfg["train"].get("entropy_coef", 0.01),
        imitation_coef=cfg["train"].get("imitation_w_start", 1.0),
        imitation_eps=cfg["control"].get("imitation_min_budget", 0.1),
        max_grad_norm=cfg["train"].get("max_grad_norm", 0.5),
        device=device,
        target_kl=cfg["train"].get("target_kl", 0.15),
        value_clip_eps=cfg["train"].get("value_clip_eps", 0.2),
        adv_norm_eps=cfg["train"].get("adv_norm_eps", 1e-8),
    )
    optimizer = Adam(policy.parameters(), lr=ppo_cfg.lr)
    base_lr = ppo_cfg.lr
    lr_anneal = bool(cfg["train"].get("lr_anneal", True))
    lr_min_factor = float(cfg["train"].get("lr_min_factor", 0.05))

    total_updates = cfg["train"].get("updates", 2000)
    horizon = cfg["train"].get("horizon", 256)

    # 残差增益（如需热身冻结，可在此处按 upd 动态改）
    residual_gain = float(cfg["control"].get("residual_gain", 1.0))

    # imitation 权重调度
    imitation_w_start = float(cfg["train"].get("imitation_w_start", 1.0))
    imitation_w_end   = float(cfg["train"].get("imitation_w_end", 0.1))
    imitation_decay_updates = int(cfg["train"].get("imitation_decay_updates", total_updates))

    best_sr = 0.0

    for upd in range(1, total_updates + 1):
        # 线性衰减 imitation 权重
        alpha = min(1.0, upd / max(1, imitation_decay_updates))
        ppo_cfg.imitation_coef = float(imitation_w_start + (imitation_w_end - imitation_w_start) * alpha)

        buf = RolloutBuffer()
        obs = env.reset(seed=cfg.get("seed", 0) + upd)

        ep_rewards = []
        done = False
        t = 0
        last_info = {}
        while not done and t < horizon:
            low_obs = torch.from_numpy(obs["low"]).float().to(device)  # (nD, obs_dim)
            act, logp, val = policy.act(low_obs, deterministic=False)
            act_np = act.detach().cpu().numpy()       # (nD,3)
            logp_np = logp.detach().cpu().numpy()     # (nD,)
            val_np = val.detach().cpu().numpy()       # (nD,)

            obs2, rew, done, info = env.step(act_np)
            last_info = info

            buf.add(
                obs=obs["low"], action=act_np, logp=logp_np, value=val_np,
                reward=rew, done=float(done),
                teacher_res=info["pn_action"], pn_valid=info["pn_valid"], res_budget=info["res_budget"]
            )
            ep_rewards.append(rew)
            obs = obs2
            t += 1

        # 用最后一个观测计算 bootstrap value（处理 horizon 截断）
        with torch.no_grad():
            last_value = policy.critic(torch.from_numpy(obs["low"]).float().to(device)).squeeze(-1).cpu().numpy()
        buf.set_last_value(last_value)

        data = buf.cat()
        stats = ppo_update(policy, optimizer, data, ppo_cfg, residual_gain=residual_gain)

        if lr_anneal:
            frac = 1.0 - (upd / total_updates)
            frac = min(max(frac, lr_min_factor), 1.0)
            current_lr = base_lr * frac
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # 训练段统计（只记录，不用于刷新 best）
        train_sr = float(last_info.get("success", False)) if last_info else 0.0
        logger.log_scalar("train/ep_return", float(np.sum(ep_rewards)), upd)
        logger.log_scalar("train/success", train_sr, upd)
        logger.log_scalar("loss/policy", stats["policy_loss"], upd)
        logger.log_scalar("loss/value", stats["value_loss"], upd)
        logger.log_scalar("loss/imitation", stats["im_loss"], upd)
        logger.log_scalar("loss/entropy", stats["entropy"], upd)
        logger.log_scalar("debug/approx_kl", stats["approx_kl"], upd)
        logger.log_scalar("debug/approx_kl_last", stats["approx_kl_last"], upd)
        logger.log_scalar("debug/kl_stop", 1.0 if stats["kl_stop"] else 0.0, upd)
        logger.log_scalar("train/lr", float(current_lr), upd)

        # 定期完整评测，并据此刷新 best
        EVAL_EVERY = 20
        if (upd % EVAL_EVERY) == 0:
            sr_eval = quick_eval(lambda: make_env(cfg), policy, device, episodes=100, seed=123)
            logger.log_scalar("eval/success", sr_eval, upd)
            if sr_eval >= best_sr:
                best_sr = sr_eval
                torch.save(policy.state_dict(), os.path.join("ckpts", "best.pt"))

        logger.flush()
        torch.save(policy.state_dict(), os.path.join("ckpts", "latest.pt"))

    # 收官大评测（避免只看历史最好值）
    final_sr = quick_eval(lambda: make_env(cfg), policy, device, episodes=300, seed=2025)
    print(f"Final eval over 300 eps: {final_sr:.3f} | Best during training: {best_sr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/default.yaml")
    args = parser.parse_args()
    main(args)
