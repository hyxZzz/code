"""PPO implementation tailored for the high-level manager policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .ppo import compute_gae


@dataclass
class ManagerPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    batch_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    adv_norm_eps: float = 1e-8
    device: str = "cpu"
    teacher_coef: float = 0.1


class ManagerRolloutBuffer:
    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.logps: list[float] = []
        self.values: list[float] = []
        self.masks: list[np.ndarray] = []
        self.teacher_actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.last_value: float | None = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        logp: float,
        value: float,
        mask: np.ndarray,
        teacher_action: np.ndarray,
    ):
        self.obs.append(obs.astype(np.float32))
        self.actions.append(action.astype(np.int32))
        self.logps.append(float(logp))
        self.values.append(float(value))
        self.masks.append(mask.astype(np.float32))
        self.teacher_actions.append(teacher_action.astype(np.int32))
        self.rewards.append(0.0)
        self.dones.append(0.0)

    def add_reward(self, reward: float, done: bool):
        if not self.rewards:
            return
        self.rewards[-1] += float(reward)
        self.dones[-1] = float(done)

    def set_last_value(self, value: float):
        self.last_value = float(value)

    def cat(self):
        if not self.obs:
            raise RuntimeError("rollout buffer is empty")
        data = {
            "obs": np.stack(self.obs, axis=0).astype(np.float32),
            "actions": np.stack(self.actions, axis=0).astype(np.int64),
            "logps": np.array(self.logps, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
            "masks": np.stack(self.masks, axis=0).astype(np.float32),
            "teacher_actions": np.stack(self.teacher_actions, axis=0).astype(np.int64),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
        }

        if self.last_value is not None:
            data["last_value"] = np.array(self.last_value, dtype=np.float32)

        return data


def manager_ppo_update(policy, optimizer, data, cfg: ManagerPPOConfig):
    device = torch.device(cfg.device)
    obs = torch.from_numpy(data["obs"]).to(device)
    actions = torch.from_numpy(data["actions"]).to(device)
    old_logps = torch.from_numpy(data["logps"]).to(device)
    old_values = torch.from_numpy(data["values"]).to(device)
    action_masks = torch.from_numpy(data["masks"]).to(device)
    teacher_actions = torch.from_numpy(data["teacher_actions"]).long().to(device)

    rewards = data["rewards"]
    dones = data["dones"]
    last_value = float(data.get("last_value", 0.0))
    adv, returns = compute_gae(
        rewards, data["values"], dones, cfg.gamma, cfg.gae_lambda, last_value
    )
    adv_t = torch.from_numpy(adv).float().to(device)
    returns_t = torch.from_numpy(returns).float().to(device)

    if adv_t.numel() > 1:
        adv_mean = adv_t.mean()
        adv_std = adv_t.std()
        if torch.isfinite(adv_std) and adv_std > cfg.adv_norm_eps:
            adv_t = (adv_t - adv_mean) / (adv_std + cfg.adv_norm_eps)

    B = obs.shape[0]
    inds = np.arange(B)
    n_minibatches = max(1, B // max(1, cfg.batch_size))

    policy_loss = torch.tensor(0.0, device=device)
    value_loss = torch.tensor(0.0, device=device)
    entropy_term = torch.tensor(0.0, device=device)
    imitation_loss = torch.tensor(0.0, device=device)
    last_kl = 0.0
    kl_stop = False

    for epoch in range(cfg.epochs):
        np.random.shuffle(inds)
        for mb_inds in np.array_split(inds, n_minibatches):
            mb = torch.from_numpy(mb_inds).long().to(device)
            mb_obs = obs[mb]
            mb_actions = actions[mb]
            mb_old_logp = old_logps[mb]
            mb_old_val = old_values[mb]
            mb_adv = adv_t[mb]
            mb_ret = returns_t[mb]
            mb_mask = action_masks[mb]

            new_logp, v_pred, entropy, logits = policy.evaluate_actions(
                mb_obs, mb_actions, action_mask=mb_mask
            )
            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * F.mse_loss(v_pred, mb_ret)
            entropy_loss = -entropy.mean() * cfg.entropy_coef
            entropy_term = entropy.mean()

            mb_teacher = teacher_actions[mb]
            logits_flat = logits.reshape(-1, policy.cfg.nP + 1)
            targets_flat = mb_teacher.reshape(-1)
            mask_flat = mb_mask.reshape(-1, policy.cfg.nP + 1)
            valid = mask_flat.sum(dim=-1) > 0.5
            if valid.any():
                imitation_loss = F.cross_entropy(
                    logits_flat[valid], targets_flat[valid], reduction="mean"
                )
            else:
                imitation_loss = torch.tensor(0.0, device=device)

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                + entropy_loss
                + cfg.teacher_coef * imitation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            approx_kl_mb = torch.mean(mb_old_logp - new_logp).detach()
            last_kl = float(approx_kl_mb.item())
            if cfg.target_kl > 0 and last_kl > cfg.target_kl:
                kl_stop = True
                break
        if kl_stop:
            break

    with torch.no_grad():
        new_logp_all, _, _, _ = policy.evaluate_actions(obs, actions, action_mask=action_masks)
        approx_kl = (old_logps - new_logp_all).mean().item()

    return dict(
        policy_loss=float(policy_loss.item()),
        value_loss=float(value_loss.item()),
        entropy=float(entropy_term.item()),
        approx_kl=float(approx_kl),
        approx_kl_last=float(last_kl),
        kl_stop=bool(kl_stop),
        imitation_loss=float(imitation_loss.item()),
    )
