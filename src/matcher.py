# src/matcher.py
# ---------------------------------------
# 目标分配：匈牙利（首选）+ 防抖（锁定&换目标代价），无 SciPy 时回退贪心
from __future__ import annotations
import numpy as np

def _hungarian(costs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
        # 将无效位置设成大数
        big = np.where(np.isfinite(costs), 0.0, 1.0).max(initial=0.0) + 1.0
        cost = np.where(mask > 0.5, costs, np.max(costs[np.isfinite(costs)]) + 1e6)
        r, c = linear_sum_assignment(cost)
        assign = -np.ones(cost.shape[0], dtype=np.int32)
        for rr, cc in zip(r, c):
            if mask[rr, cc] > 0.5 and np.isfinite(costs[rr, cc]):
                assign[rr] = int(cc)
        return assign
    except Exception:
        return _greedy(costs, mask)

def _greedy(costs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    nD, nP = costs.shape
    assign = -np.ones(nD, dtype=np.int32)
    used = set()
    # 依次挑选最小可用成本
    pairs = [(costs[i,j], i, j) for i in range(nD) for j in range(nP) if mask[i,j] > 0.5 and np.isfinite(costs[i,j])]
    pairs.sort(key=lambda x: x[0])
    for _, i, j in pairs:
        if assign[i] == -1 and j not in used:
            assign[i] = j
            used.add(j)
    return assign

def assign_targets(costs: np.ndarray, mask: np.ndarray, prev_assign: np.ndarray,
                   switch_penalty: float = 10.0, lock_steps: int = 6, algo: str = "hungarian") -> np.ndarray:
    nD, nP = costs.shape
    # 应用“换目标代价”
    adj_costs = costs.copy()
    for i in range(nD):
        for j in range(nP):
            if prev_assign[i] == j:
                continue
            if mask[i, j] > 0.5 and np.isfinite(adj_costs[i, j]):
                adj_costs[i, j] += switch_penalty

    if algo == "hungarian":
        base_assign = _hungarian(adj_costs, mask)
    else:
        base_assign = _greedy(adj_costs, mask)

    # 锁定：保留与之前相同的分配（若目标仍有效）
    final = -np.ones(nD, dtype=np.int32)
    for i in range(nD):
        j_prev = prev_assign[i]
        if 0 <= j_prev < nP and mask[i, j_prev] > 0.5:
            final[i] = j_prev
        else:
            final[i] = base_assign[i]
    return final
