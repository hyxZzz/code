# src/envs/dynamics.py
# ---------------------------------------
# 简单质点动力学 + 工具函数
from __future__ import annotations
import numpy as np

def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    if max_norm <= 0:
        return np.zeros_like(v)
    scale = np.minimum(1.0, max_norm / n)
    return v * scale

def integrate_pm(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float,
                 v_max: float) -> tuple[np.ndarray, np.ndarray]:
    """点质量积分（欧拉），速度限幅"""
    vel = vel + acc * dt
    vel = clamp_norm(vel, v_max)
    pos = pos + vel * dt
    return pos, vel

def within_radius(p1: np.ndarray, p2: np.ndarray, r: float) -> bool:
    return np.linalg.norm(p1 - p2) <= r + 1e-9

def soft_clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))

def safe_norm(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(x) + eps)

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n
