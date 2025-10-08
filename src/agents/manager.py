"""High-level manager policy for hierarchical reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ManagerConfig:
    """Configuration for the manager policy network."""

    nD: int
    nP: int
    hidden: int = 256
    device: str = "cpu"


class ManagerPolicy(nn.Module):
    """Actor-Critic policy that outputs per-defender target preferences."""

    def __init__(self, cfg: ManagerConfig, obs_dim: int):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        action_dim = cfg.nP + 1  # include "no preference" slot

        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(cfg.hidden, cfg.nD * action_dim)
        self.critic = nn.Linear(cfg.hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _masked_logits(self, logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return torch.nan_to_num(logits)
        mask = action_mask.to(logits.device)
        # Clamp to avoid log(0) during categorical sampling
        masked = logits.masked_fill(mask <= 0, -1e9)
        return torch.nan_to_num(masked)

    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or greedily select) defender target indices."""

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.encoder(obs)
        logits = self.actor(features).view(obs.size(0), self.cfg.nD, self.cfg.nP + 1)
        logits = self._masked_logits(logits, action_mask)

        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
            logp = dist.log_prob(actions)
        else:
            actions = dist.sample()
            logp = dist.log_prob(actions)
        logp = logp.sum(dim=-1)
        value = self.critic(features).squeeze(-1)
        return actions, logp, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        features = self.encoder(obs)
        logits = self.actor(features).view(obs.size(0), self.cfg.nD, self.cfg.nP + 1)
        logits = self._masked_logits(logits, action_mask)

        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)
        return logp, value, entropy, logits
