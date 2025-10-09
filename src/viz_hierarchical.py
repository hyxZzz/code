#!/usr/bin/env python3
"""Visualize the hierarchical (manager + controller) policy stack."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from src.agents.controller import ControllerConfig, ControllerPolicy
from src.agents.manager import ManagerConfig, ManagerPolicy
from src.envs.three_d_pursuit import ThreeDPursuitEnv
from src.viz_gif import fig_to_ndarray, load_config, choose_device, set_aspect_equal_3d


def _load_controller(cfg: dict, device: torch.device, ckpt: Path | None) -> ControllerPolicy | None:
    env = ThreeDPursuitEnv(cfg)
    obs_dim = env.low_obs_dim
    policy = ControllerPolicy(ControllerConfig(obs_dim=obs_dim, device=str(device))).to(device)
    if ckpt is None:
        print("[viz/hier] No controller checkpoint provided — using zero-residual baseline.")
        return None
    if not ckpt.exists():
        raise FileNotFoundError(f"Controller checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    print(f"[viz/hier] Loaded controller weights from {ckpt}")
    return policy


def _load_manager(cfg: dict, device: torch.device, ckpt: Path | None) -> ManagerPolicy | None:
    if ckpt is None:
        print("[viz/hier] No manager checkpoint provided — using configuration default assignments.")
        return None
    env = ThreeDPursuitEnv(cfg)
    obs_dim = env.get_manager_observation().shape[0]
    hidden = int(cfg.get("manager", {}).get("hidden", 256))
    manager_cfg = ManagerConfig(nD=env.nD, nP=env.nP, hidden=hidden, device=str(device))
    policy = ManagerPolicy(manager_cfg, obs_dim=obs_dim).to(device)
    if not ckpt.exists():
        raise FileNotFoundError(f"Manager checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            "[viz/hier] Warning: manager checkpoint loaded with partial match. "
            f"Missing: {missing} | Unexpected: {unexpected}"
        )
    policy.eval()
    print(f"[viz/hier] Loaded manager weights from {ckpt}")
    return policy


def _format_assignment(assign: Iterable[int], null_idx: int) -> str:
    labels = []
    for idx, target in enumerate(assign):
        try:
            tgt = int(target)
        except (TypeError, ValueError):
            labels.append(f"D{idx}: ?")
            continue
        if tgt < 0 or tgt == null_idx:
            labels.append(f"D{idx}: -")
        else:
            labels.append(f"D{idx}: P{tgt}")
    return " | ".join(labels)


def render_hierarchical_episode(
    cfg: dict,
    controller_ckpt: Path | None,
    manager_ckpt: Path | None,
    *,
    seed: int = 0,
    frames: int = 600,
    fps: int = 15,
    out: Path = Path("viz_hier.gif"),
    deterministic_controller: bool = True,
    deterministic_manager: bool = True,
    draw_assign_lines: bool = True,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 110,
) -> None:
    cfg = dict(cfg)
    cfg.setdefault("manager", {})
    if manager_ckpt is not None:
        cfg["manager"]["mode"] = "learned"
    device = torch.device(choose_device(cfg.get("device", "auto")))

    controller = _load_controller(cfg, device, controller_ckpt)
    manager = _load_manager(cfg, device, manager_ckpt)

    env = ThreeDPursuitEnv(cfg)
    null_idx = env.manager_null_action

    T_hist: list[np.ndarray] = []
    D_hist: list[list[np.ndarray]] = [[] for _ in range(env.nD)]
    P_hist: list[list[np.ndarray]] = [[] for _ in range(env.nP)]

    obs = env.reset(seed=seed)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    world = cfg["env"].get("world_bounds", [[-1000, 1000], [-300, 300], [-150, 150]])
    xlim, ylim, zlim = world

    writer = imageio.get_writer(out, mode="I", duration=1.0 / max(1, fps))

    try:
        done = False
        step = 0
        info = {}
        while not done and step < frames:
            T_hist.append(env.T.pos.copy())
            for i, d in enumerate(env.D):
                D_hist[i].append(d.pos.copy())
            for j, p in enumerate(env.P):
                P_hist[j].append(p.pos.copy())

            need_manager = (
                manager is not None
                and env.manager_mode == "learned"
                and ((env.t + 1) % max(1, env.manager_period) == 0)
            )
            if need_manager:
                with torch.no_grad():
                    mgr_obs = torch.from_numpy(env.get_manager_observation()).float().to(device)
                    mgr_mask = torch.from_numpy(env.get_manager_action_mask()).float().to(device)
                    mgr_action, _, _ = manager.act(
                        mgr_obs,
                        action_mask=mgr_mask,
                        deterministic=deterministic_manager,
                    )
                env.set_manager_action(mgr_action.squeeze(0).cpu().numpy())

            if controller is None:
                ctrl_action = np.zeros((env.nD, 3), dtype=np.float32)
            else:
                with torch.no_grad():
                    low_obs = torch.from_numpy(obs["low"]).float().to(device)
                    ctrl_action, _, _ = controller.act(
                        low_obs,
                        deterministic=deterministic_controller,
                    )
                ctrl_action = ctrl_action.cpu().numpy()

            obs, reward, done, info = env.step(ctrl_action)

            ax.cla()
            ax.set_title(f"Hierarchical 3D Pursuit — step {step} | reward {reward:.3f}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_zlim(zlim[0], zlim[1])

            if len(T_hist) > 1:
                Th = np.array(T_hist)
                ax.plot(Th[:, 0], Th[:, 1], Th[:, 2], lw=1.5, alpha=0.9, color="tab:green", label="Target path")
            for i in range(env.nD):
                if len(D_hist[i]) > 1:
                    Dh = np.array(D_hist[i])
                    ax.plot(Dh[:, 0], Dh[:, 1], Dh[:, 2], lw=1.0, alpha=0.8, color="tab:blue")
            for j in range(env.nP):
                if len(P_hist[j]) > 1:
                    Ph = np.array(P_hist[j])
                    ax.plot(Ph[:, 0], Ph[:, 1], Ph[:, 2], lw=1.0, alpha=0.6, color="tab:red")

            ax.scatter(
                [env.T.pos[0]],
                [env.T.pos[1]],
                [env.T.pos[2]],
                s=60,
                marker="*",
                color="tab:green",
                label="Target",
            )
            Dpos = np.array([d.pos for d in env.D])
            ax.scatter(Dpos[:, 0], Dpos[:, 1], Dpos[:, 2], s=30, marker="o", color="tab:blue", label="Defenders")
            Ppos = np.array([p.pos for p in env.P])
            Palive = np.array([int(p.alive) for p in env.P])
            if len(Ppos):
                alive_idx = np.where(Palive == 1)[0]
                dead_idx = np.where(Palive == 0)[0]
                if alive_idx.size:
                    ax.scatter(
                        Ppos[alive_idx, 0],
                        Ppos[alive_idx, 1],
                        Ppos[alive_idx, 2],
                        s=40,
                        marker="x",
                        color="tab:red",
                        label="Attackers (alive)",
                    )
                if dead_idx.size:
                    ax.scatter(
                        Ppos[dead_idx, 0],
                        Ppos[dead_idx, 1],
                        Ppos[dead_idx, 2],
                        s=40,
                        marker="x",
                        color="0.5",
                        label="Attackers (down)",
                    )

            if draw_assign_lines:
                accepted = info.get("assign")
                proposed = info.get("manager_proposed_action")
                if accepted is not None:
                    for i, j in enumerate(accepted):
                        if j is None or j < 0 or j >= len(env.P) or not env.P[j].alive:
                            continue
                        xs = [env.D[i].pos[0], env.P[j].pos[0]]
                        ys = [env.D[i].pos[1], env.P[j].pos[1]]
                        zs = [env.D[i].pos[2], env.P[j].pos[2]]
                        ax.plot(xs, ys, zs, ls="-", lw=1.2, color="tab:purple")
                if proposed is not None:
                    proposed = np.asarray(proposed)
                    for i, j in enumerate(proposed):
                        if j is None or j < 0 or j >= len(env.P) or not env.P[j].alive:
                            continue
                        xs = [env.D[i].pos[0], env.P[j].pos[0]]
                        ys = [env.D[i].pos[1], env.P[j].pos[1]]
                        zs = [env.D[i].pos[2], env.P[j].pos[2]]
                        ax.plot(xs, ys, zs, ls="--", lw=0.9, color="tab:orange", alpha=0.7)

            attackers_alive = info.get("attackers_alive", np.count_nonzero(Palive))
            status = "RUN"
            if info.get("success") or info.get("done_success"):
                status = "SUCCESS"
            elif info.get("done_failure"):
                status = "FAIL"

            manager_lines = []
            if manager is not None:
                manager_lines.append(f"mode: {info.get('manager_mode', 'learned')}")
                manager_lines.append(
                    f"bonus: {info.get('manager_bonus', 0.0):+.3f} | proxy: {info.get('manager_reward_proxy', 0.0):+.3f}"
                )
                manager_lines.append(
                    f"costΔ: {info.get('manager_cost_delta', 0.0):+.2f} | threat: {info.get('manager_threat_score', 0.0):.2f}"
                )
                manager_lines.append(
                    "accepted: "
                    f"{bool(info.get('manager_assignment_accepted', True))}"
                )
                current_assign = info.get("manager_action")
                if current_assign is not None:
                    manager_lines.append(
                        "curr: " + _format_assignment(current_assign, null_idx)
                    )
                if info.get("manager_proposed_action") is not None:
                    manager_lines.append(
                        "prop: " + _format_assignment(info["manager_proposed_action"], null_idx)
                    )

            overlay = (
                f"status: {status}\n"
                f"P alive: {attackers_alive}\n"
                + ("\n".join(manager_lines) if manager_lines else "")
            ).strip()
            ax.text2D(
                0.02,
                0.98,
                overlay,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", fc="w", ec="0.8", alpha=0.75),
            )

            allX = np.hstack([[xlim[0], xlim[1]], Dpos[:, 0], Ppos[:, 0], [env.T.pos[0]]])
            allY = np.hstack([[ylim[0], ylim[1]], Dpos[:, 1], Ppos[:, 1], [env.T.pos[1]]])
            allZ = np.hstack([[zlim[0], zlim[1]], Dpos[:, 2], Ppos[:, 2], [env.T.pos[2]]])
            set_aspect_equal_3d(ax, allX, allY, allZ)

            if step == 0:
                ax.legend(loc="upper right", fontsize=8)

            frame = fig_to_ndarray(fig)
            writer.append_data(frame)
            step += 1

    finally:
        writer.close()
        plt.close(fig)
        print(f"[viz/hier] Saved GIF to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a GIF for the hierarchical policy stack.")
    parser.add_argument("--config", type=str, default="src/configs/default.yaml", help="Path to the YAML config.")
    parser.add_argument(
        "--controller-ckpt",
        type=str,
        default="ckpts/best.pt",
        help="Path to the residual controller checkpoint.",
    )
    parser.add_argument(
        "--manager-ckpt",
        type=str,
        default="ckpts/manager_best.pt",
        help="Path to the high-level manager checkpoint.",
    )
    parser.add_argument("--out", type=str, default="viz_hier.gif", help="Output GIF path.")
    parser.add_argument("--frames", type=int, default=600, help="Maximum number of environment steps to render.")
    parser.add_argument("--fps", type=int, default=15, help="GIF framerate.")
    parser.add_argument("--seed", type=int, default=0, help="Environment random seed.")
    parser.add_argument(
        "--stochastic-controller",
        action="store_true",
        help="Sample controller actions instead of taking the greedy action.",
    )
    parser.add_argument(
        "--stochastic-manager",
        action="store_true",
        help="Sample manager assignments instead of taking the greedy action.",
    )
    parser.add_argument("--no-assign-lines", action="store_true", help="Disable drawing defender→attacker lines.")
    parser.add_argument("--figw", type=float, default=8.0, help="Figure width in inches.")
    parser.add_argument("--figh", type=float, default=6.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=110, help="Figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    render_hierarchical_episode(
        cfg=cfg,
        controller_ckpt=Path(args.controller_ckpt) if args.controller_ckpt else None,
        manager_ckpt=Path(args.manager_ckpt) if args.manager_ckpt else None,
        seed=args.seed,
        frames=args.frames,
        fps=args.fps,
        out=Path(args.out),
        deterministic_controller=not args.stochastic_controller,
        deterministic_manager=not args.stochastic_manager,
        draw_assign_lines=not args.no_assign_lines,
        figsize=(args.figw, args.figh),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
