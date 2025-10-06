# src/agents/controller.py
# ---------------------------------------
# 低层连续控制策略（3D 残差向量 in [-1,1]^3），Actor-Critic + Tanh 高斯
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ControllerConfig:
    obs_dim: int
    action_dim: int = 3
    hidden: int = 256
    logstd_init: float = -0.5
    device: str = "cpu"

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class ControllerPolicy(nn.Module):
    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg
        self.actor = MLP(cfg.obs_dim, cfg.action_dim, cfg.hidden)
        self.critic = MLP(cfg.obs_dim, 1, cfg.hidden)
        self.log_std = nn.Parameter(torch.ones(cfg.action_dim) * cfg.logstd_init)

    def forward(self, obs):
        raise NotImplementedError

    def act(self, obs: torch.Tensor, deterministic: bool=False):
        """
        obs: (B, obs_dim)
        return action in [-1,1], logp, value
        """
        mean = torch.tanh(self.actor(obs))
        std = torch.exp(self.log_std)
        if deterministic:
            a_tanh = mean
            logp = None
        else:
            # 采样高斯 → tanh squash（对应的 logp 需做tanh修正）
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            a_tanh = torch.tanh(z)

            # tanh log-prob 修正：logp(z) - log(1 - tanh(z)^2)
            logp = normal.log_prob(z) - torch.log(1 - a_tanh.pow(2) + 1e-6)
            logp = logp.sum(-1)

        value = self.critic(obs).squeeze(-1)
        return a_tanh, logp, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        obs: (B, obs_dim), actions ∈ [-1,1]
        返回：logp(actions), value, entropy
        """
        mean = torch.tanh(self.actor(obs))
        std = torch.exp(self.log_std)
        # 反解 pre-tanh value
        atanh = torch.atanh(actions.clamp(-0.999, 0.999))
        normal = torch.distributions.Normal(mean, std)
        logp = normal.log_prob(atanh) - torch.log(1 - actions.pow(2) + 1e-6)
        logp = logp.sum(-1)
        entropy = normal.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return logp, value, entropy
