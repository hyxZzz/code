# src/agents/manager.py
# ---------------------------------------
# （保留）学习型 Manager 的占位实现（本阶段不训练）
# 说明：第一阶段我们使用规则分配（matcher.assign_targets）。此文件保留以便后续第二阶段启用。
from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ManagerConfig:
    nD: int
    nP: int
    hidden: int = 256
    device: str = "cpu"

class ManagerPolicy(nn.Module):
    def __init__(self, cfg: ManagerConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.nD * cfg.nP  # 展平成本矩阵
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden), nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.nD * cfg.nP)
        )

    def forward(self, costs, mask):
        """
        costs: (nD, nP), mask: (nD, nP) in {0,1}
        return logits masked (nD, nP)
        """
        x = costs.view(1, -1)  # 简化
        logits = self.net(x).view(1, self.cfg.nD, self.cfg.nP)
        # 将无效位置置为 -inf
        logits = logits + (mask.view(1, self.cfg.nD, self.cfg.nP) - 1.0) * 1e9
        return logits
