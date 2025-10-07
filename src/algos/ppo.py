# src/algos/ppo.py
# ---------------------------------------
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    batch_size: int = 4096
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    imitation_coef: float = 0.5  # 会按配置衰减
    imitation_eps: float = 0.1   # res_budget 下限
    max_grad_norm: float = 0.5
    device: str = "cpu"
    target_kl: float = 0.15
    value_clip_eps: float = 0.2
    adv_norm_eps: float = 1e-8

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logps = []
        self.values = []
        self.rewards = []
        self.dones = []
        # imitation
        self.teacher_res = []
        self.pn_valid = []
        self.res_budget = []

    def add(self, obs, action, logp, value, reward, done, teacher_res, pn_valid, res_budget):
        self.obs.append(obs.copy())
        self.actions.append(action.copy())
        self.logps.append(logp.copy())
        self.values.append(value.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.teacher_res.append(teacher_res.copy())
        self.pn_valid.append(pn_valid.copy())
        self.res_budget.append(res_budget.copy())

    def cat(self):
        data = {
            "obs": np.concatenate(self.obs, axis=0).astype(np.float32),           # (T*nD, obs_dim)
            "actions": np.concatenate(self.actions, axis=0).astype(np.float32),   # (T*nD, 3)
            "logps": np.concatenate(self.logps, axis=0).astype(np.float32),       # (T*nD,)
            "values": np.concatenate(self.values, axis=0).astype(np.float32),     # (T*nD,)
            "rewards": np.array(self.rewards, dtype=np.float32),                  # (T,)
            "dones": np.array(self.dones, dtype=np.float32),                      # (T,)
            "teacher_res": np.concatenate(self.teacher_res, axis=0).astype(np.float32), # (T*nD,3)
            "pn_valid": np.concatenate(self.pn_valid, axis=0).astype(np.float32), # (T*nD,)
            "res_budget": np.concatenate(self.res_budget, axis=0).astype(np.float32),   # (T*nD,)
        }
        return data

def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
        next_value = values[t]
    returns = adv + values
    return adv, returns

def ppo_update(policy, optimizer, data, cfg: PPOConfig, residual_gain: float):
    device = cfg.device
    obs = torch.from_numpy(data["obs"]).to(device)
    actions = torch.from_numpy(data["actions"]).to(device)
    old_logps = torch.from_numpy(data["logps"]).to(device)
    values = torch.from_numpy(data["values"]).to(device)

    rewards = data["rewards"]
    dones = data["dones"]
    # 为了价值计算，需要把每步 team reward 展开到 nD 个样本的 value 上
    # 这里采用“时间主导 GAE”：对每时间步的价值取平均（或复制），简化处理
    # 我们做复制：每步的 value 是 (nD) 均值的代表
    nD = actions.shape[0] // len(rewards)

    # 取每步 value 的均值（按 defender 维度）
    values_time = values.view(len(rewards), nD).mean(-1).cpu().numpy()
    adv, rets = compute_gae(rewards, values_time, dones, cfg.gamma, cfg.gae_lambda)
    adv_tiled = torch.from_numpy(np.repeat(adv[:, None], nD, axis=1).reshape(-1)).float().to(device)
    rets_tiled = torch.from_numpy(np.repeat(rets[:, None], nD, axis=1).reshape(-1)).float().to(device)

    # advantage 标准化可大幅缓解梯度爆炸 / 占优任务梯度不平衡
    adv_mean = adv_tiled.mean()
    adv_std = adv_tiled.std()
    if torch.isfinite(adv_std) and adv_std > cfg.adv_norm_eps:
        adv_tiled = (adv_tiled - adv_mean) / (adv_std + cfg.adv_norm_eps)

    # imitation 目标（以“绝对残差指令”为目标，减小预算尺度带来的不适定）
    teacher_res = torch.from_numpy(data["teacher_res"]).to(device)          # (B,3)
    pn_valid = torch.from_numpy(data["pn_valid"]).to(device)                # (B,)
    res_budget = torch.from_numpy(data["res_budget"]).to(device)            # (B,)

    B = actions.shape[0]
    inds = np.arange(B)
    n_minib = max(1, B // max(1, cfg.batch_size))

    policy_loss = torch.tensor(0.0, device=device)
    value_loss = torch.tensor(0.0, device=device)
    im_loss = torch.tensor(0.0, device=device)
    entropy_term = torch.tensor(0.0, device=device)
    kl_stop = False
    last_kl = 0.0

    for epoch in range(cfg.epochs):
        np.random.shuffle(inds)
        for mb in np.array_split(inds, n_minib):
            mb = torch.from_numpy(mb).long().to(device)
            mb_obs = obs[mb]
            mb_act = actions[mb]
            mb_old_logp = old_logps[mb]
            mb_adv = adv_tiled[mb]
            mb_ret = rets_tiled[mb]
            mb_old_val = values[mb]

            new_logp, v_pred, ent = policy.evaluate_actions(mb_obs, mb_act)
            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred_clipped = mb_old_val + (v_pred - mb_old_val).clamp(-cfg.value_clip_eps, cfg.value_clip_eps)
            value_losses = (v_pred - mb_ret) ** 2
            value_losses_clipped = (value_pred_clipped - mb_ret) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            entropy_loss = -ent.mean() * cfg.entropy_coef
            entropy_term = ent.mean()

            # imitation：只在 (pn_valid==1 & res_budget>eps) 上计算
            mb_teacher = teacher_res[mb]
            mb_valid = (pn_valid[mb] > 0.5) & (res_budget[mb] > cfg.imitation_eps)
            # 用当前策略对观测重新前向，得到可回传梯度的残差预测
            mb_act_pred = torch.tanh(policy.actor(mb_obs))
            # 预测的绝对残差（把动作从[-1,1]映射为：residual_gain * res_budget * action）
            mb_res_pred = residual_gain * res_budget[mb].unsqueeze(-1) * mb_act_pred
            if mb_valid.any():
                im_loss = F.mse_loss(mb_res_pred[mb_valid], mb_teacher[mb_valid])
            else:
                im_loss = torch.tensor(0.0, device=device)

            loss = policy_loss + cfg.value_coef * value_loss + entropy_loss + cfg.imitation_coef * im_loss

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

    # 返回最后一轮的统计
    with torch.no_grad():
        approx_kl = (old_logps - policy.evaluate_actions(obs, actions)[0]).mean().item()
    return dict(policy_loss=float(policy_loss.item()),
                value_loss=float(value_loss.item()),
                im_loss=float(im_loss.item()),
                entropy=float(entropy_term.item()),
                approx_kl=float(approx_kl),
                approx_kl_last=float(last_kl),
                kl_stop=kl_stop)
