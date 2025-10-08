#!/usr/bin/env python3
"""
viz_gif.py — 可视化 3D 追逃/护航过程，输出为 GIF。

用法示例：
  python -m viz_gif \
    --config src/configs/default.yaml \
    --weights ckpts/best.pt \
    --frames 600 --fps 15 --out viz.gif --det

    python -m src.viz_gif --config src/configs/default.yaml \
  --weights ckpts/best.pt \
  --out runs/viz_best.gif --seed 123


依赖：matplotlib, imageio, pyyaml, torch, numpy
"""
from __future__ import annotations
import os
import argparse
import yaml
import numpy as np
import torch

# 兼容无显示环境
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
import imageio.v2 as imageio

from src.envs.three_d_pursuit import ThreeDPursuitEnv
from src.agents.controller import ControllerPolicy, ControllerConfig

# ----------------------------
# 工具函数
# ----------------------------

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_device(cfg_device: str = "auto") -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def set_aspect_equal_3d(ax, X, Y, Z):
    """使 3D 轴的缩放一致（正方体），避免形变。"""
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    z_min, z_max = np.min(Z), np.max(Z)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range == 0:
        max_range = 1.0
    x_mid = (x_max + x_min) * 0.5
    y_mid = (y_max + y_min) * 0.5
    z_mid = (z_max + z_min) * 0.5
    r = max_range * 0.5
    ax.set_xlim(x_mid - r, x_mid + r)
    ax.set_ylim(y_mid - r, y_mid + r)
    ax.set_zlim(z_mid - r, z_mid + r)


def fig_to_ndarray(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        # 新版本推荐：RGBA 缓冲区
        buf = np.asarray(fig.canvas.buffer_rgba())
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]  # 丢弃 alpha
    except Exception:
        # 兼容旧版本
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        arr = buf.reshape(h, w, 3)
    return arr



# ----------------------------
# 核心渲染
# ----------------------------

def render_episode(cfg: dict,
                   weights: str | None = None,
                   seed: int = 0,
                   frames: int = 600,
                   fps: int = 15,
                   out: str = "viz.gif",
                   deterministic: bool = True,
                   draw_assign_lines: bool = True,
                   figsize=(8, 6),
                   dpi: int = 110):
    device = choose_device(cfg.get("device", "auto"))

    # 环境与策略
    env = ThreeDPursuitEnv(cfg)
    obs_dim = env.low_obs_dim
    policy = None
    if weights is not None and os.path.isfile(weights):
        ctrl_cfg = ControllerConfig(obs_dim=obs_dim, device=device)
        policy = ControllerPolicy(ctrl_cfg).to(device)
        sd = torch.load(weights, map_location=device)
        policy.load_state_dict(sd)
        policy.eval()
        print(f"[viz] Loaded weights: {weights}")
    else:
        print("[viz] No weights provided — 使用零残差作为对照。")

    # 轨迹缓存（用于画 trail）
    T_hist = []
    D_hist = [ [] for _ in range(env.nD) ]
    P_hist = [ [] for _ in range(env.nP) ]

    # 初始化环境
    obs = env.reset(seed=seed)

    # 画布
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    xlim = cfg["env"].get("world_bounds", [[-1000, 1000], [-300, 300], [-150, 150]])[0]
    ylim = cfg["env"].get("world_bounds", [[-1000, 1000], [-300, 300], [-150, 150]])[1]
    zlim = cfg["env"].get("world_bounds", [[-1000, 1000], [-300, 300], [-150, 150]])[2]

    writer = imageio.get_writer(out, mode='I', duration=1.0/max(1, fps))

    # 渲染循环
    done = False
    t = 0
    try:
        while (not done) and (t < frames):
            # --- 记录轨迹 ---
            T_hist.append(env.T.pos.copy())
            for i, d in enumerate(env.D):
                D_hist[i].append(d.pos.copy())
            for j, p in enumerate(env.P):
                P_hist[j].append(p.pos.copy())

            # --- 选择动作 ---
            if policy is None:
                act = np.zeros((env.nD, 3), dtype=np.float32)
            else:
                with torch.no_grad():
                    low_obs = torch.from_numpy(obs["low"]).float().to(device)
                    act_t, _, _ = policy.act(low_obs, deterministic=deterministic)
                    act = act_t.cpu().numpy()

            # --- 推进一步 ---
            obs, rew, done, info = env.step(act)

            # --- 绘制 ---
            ax.cla()
            ax.set_title(f"3D Pursuit — step {t} | rew {rew:.3f}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # 世界边界盒
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_zlim(zlim[0], zlim[1])

            # 画轨迹
            if len(T_hist) > 1:
                Th = np.array(T_hist)
                ax.plot(Th[:,0], Th[:,1], Th[:,2], lw=1.5, alpha=0.9, label='T path', color='tab:green')
            for i in range(env.nD):
                if len(D_hist[i]) > 1:
                    Dh = np.array(D_hist[i])
                    ax.plot(Dh[:,0], Dh[:,1], Dh[:,2], lw=1.0, alpha=0.8, color='tab:blue')
            for j in range(env.nP):
                if len(P_hist[j]) > 1:
                    Ph = np.array(P_hist[j])
                    ax.plot(Ph[:,0], Ph[:,1], Ph[:,2], lw=1.0, alpha=0.6, color='tab:red')

            # 画当前点
            # T（绿色星形）
            ax.scatter([env.T.pos[0]],[env.T.pos[1]],[env.T.pos[2]], s=60, marker='*', color='tab:green', label='T')
            # D（蓝色圆点）
            Dpos = np.array([d.pos for d in env.D])
            ax.scatter(Dpos[:,0], Dpos[:,1], Dpos[:,2], s=30, marker='o', color='tab:blue', label='D')
            # P（按存活与否染色）
            Ppos = np.array([p.pos for p in env.P])
            Palive = np.array([int(p.alive) for p in env.P])
            if len(Ppos) > 0:
                # 活着的红色叉，死亡的灰色叉
                alive_idx = np.where(Palive == 1)[0]
                dead_idx  = np.where(Palive == 0)[0]
                if alive_idx.size:
                    ax.scatter(Ppos[alive_idx,0], Ppos[alive_idx,1], Ppos[alive_idx,2], s=40, marker='x', color='tab:red', label='P (alive)')
                if dead_idx.size:
                    ax.scatter(Ppos[dead_idx,0], Ppos[dead_idx,1], Ppos[dead_idx,2], s=40, marker='x', color='0.5', label='P (down)')

            # 画 D→P 分配连线（如果需要）
            if draw_assign_lines:
                assign = info.get("assign", getattr(env, 'curr_assign', None))
                if assign is None:
                    assign = getattr(env, 'curr_assign', None)
                if assign is not None:
                    for i, j in enumerate(assign):
                        if j is None or j < 0:
                            continue
                        if j >= len(env.P) or not env.P[j].alive:
                            continue
                        p = env.P[j]
                        d = env.D[i]
                        xs = [d.pos[0], p.pos[0]]
                        ys = [d.pos[1], p.pos[1]]
                        zs = [d.pos[2], p.pos[2]]
                        ax.plot(xs, ys, zs, ls='--', lw=0.8, color='0.3')

            # 文本信息
            attackers_alive = info.get("attackers_alive")
            if attackers_alive is None and "alive_P" in info:
                attackers_alive = int(np.sum(info["alive_P"]))
            succ = bool(info.get("success", False))
            fail = bool(info.get("done_failure", False))
            status = "RUN" if not (succ or fail) else ("SUCCESS" if succ else "FAIL")
            ax.text2D(0.02, 0.98, f"status: {status}\nP alive: {attackers_alive if attackers_alive is not None else '?'}",
                      transform=ax.transAxes, ha='left', va='top', fontsize=10, bbox=dict(boxstyle='round', fc='w', ec='0.8', alpha=0.7))

            # 统一比例（避免变形）。以当前点集为依据
            allX = np.hstack([ [xlim[0], xlim[1]], Dpos[:,0], Ppos[:,0], [env.T.pos[0]] ])
            allY = np.hstack([ [ylim[0], ylim[1]], Dpos[:,1], Ppos[:,1], [env.T.pos[1]] ])
            allZ = np.hstack([ [zlim[0], zlim[1]], Dpos[:,2], Ppos[:,2], [env.T.pos[2]] ])
            set_aspect_equal_3d(ax, allX, allY, allZ)

            if t == 0:
                ax.legend(loc='upper right', fontsize=8)

            # 写帧
            frame = fig_to_ndarray(fig)
            writer.append_data(frame)

            t += 1

    finally:
        writer.close()
        plt.close(fig)
        print(f"[viz] Saved GIF: {out}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='src/configs/default.yaml', help='YAML 配置路径')
    p.add_argument('--weights', type=str, default=None, help='策略权重路径（可选）')
    p.add_argument('--out', type=str, default='viz.gif', help='输出 GIF 路径')
    p.add_argument('--frames', type=int, default=600, help='最多渲染的步数')
    p.add_argument('--fps', type=int, default=15, help='GIF 帧率')
    p.add_argument('--seed', type=int, default=0, help='环境随机种子')
    p.add_argument('--det', action='store_true', help='策略使用确定性动作')
    p.add_argument('--no-assign-lines', action='store_true', help='不绘制 D→P 分配连线')
    p.add_argument('--figw', type=float, default=8.0, help='图宽 (inch)')
    p.add_argument('--figh', type=float, default=6.0, help='图高 (inch)')
    p.add_argument('--dpi', type=int, default=110, help='DPI')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    render_episode(
        cfg=cfg,
        weights=args.weights,
        seed=args.seed,
        frames=args.frames,
        fps=args.fps,
        out=args.out,
        deterministic=args.det,
        draw_assign_lines=(not args.no_assign_lines),
        figsize=(args.figw, args.figh),
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
