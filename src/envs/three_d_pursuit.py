# src/envs/three_d_pursuit.py
# ------------------------------------------------------
# 3D target (T) - attacker (P) - defender (D) escort and intercept environment
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .dynamics import clamp_norm, within_radius, unit
from ..matcher import assign_targets


@dataclass
class Entity:
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)
    alive: bool = True


class ThreeDPursuitEnv:
    def __init__(self, cfg: Dict):
        e = cfg["env"]
        c = cfg["control"]

        # --- environment basics ---
        self.nD = int(e.get("n_defenders", 5))
        self.nP = int(e.get("n_attackers", 5))
        self.dt = float(e.get("dt", 0.1))
        self.max_steps = int(e.get("max_steps", 600))
        self.world_bounds = np.array(
            e.get("world_bounds", [[-1000, 1000], [-300, 300], [-150, 150]]),
            dtype=np.float32
        )

        # constant velocity for the target
        self.T_vel = np.array(e.get("t_velocity", [8.0, 0.0, 0.0]), dtype=np.float32)
        self.T_speed = float(np.linalg.norm(self.T_vel))
        self.T_vel_unit = (self.T_vel / (self.T_speed + 1e-9)).astype(np.float32)

        # kill radii
        self.t_attack_radius = float(e.get("t_attack_radius", 10.0))
        self.d_attack_radius = float(e.get("d_attack_radius", 15.0))
        self.t_threat_radius = float(
            e.get("t_threat_radius", self.t_attack_radius + self.d_attack_radius)
        )
        # defenders now rely on固定截击半径，不再因为威胁态而放宽
        self.d_threat_radius = float(
            e.get("d_threat_radius", self.d_attack_radius)
        )

        # speed and acceleration limits
        self.p_a_max = float(e.get("p_accel_max", 1.8))
        self.p_v_max = float(e.get("p_speed_max", 22.0))
        self.d_a_max = float(e.get("d_accel_max", 6.0))
        self.d_v_max = float(e.get("d_speed_max", 30.0))

        # spawn configuration
        s = e.get("spawn", {})
        self.spawn_t_x_frac = float(s.get("t_x_frac", 0.5))
        self.d_guard_radius = float(s.get("d_guard_radius", 40.0))
        self.front_hit_time = float(s.get("front_hit_time", 20.0))
        self.lateral_spread_y = float(s.get("lateral_spread_y", 60.0))
        self.lateral_spread_z = float(s.get("lateral_spread_z", 40.0))
        # minimum spacing and re-sampling budget
        self.min_pp_dist = float(s.get("min_pp_dist", 30.0))
        self.min_tp_dist = float(s.get("min_tp_dist", 80.0))
        self.spawn_attempts = int(s.get("spawn_attempts", 50))

        # reward shaping
        self.approach_reward_scale = float(e.get("approach_reward_scale", 0.002))
        self.closing_reward_scale = float(e.get("closing_reward_scale", 0.0))
        self.time_penalty = float(e.get("time_penalty", 0.001))
        self.kill_reward = float(e.get("kill_reward", 0.25))
        self.failure_penalty = float(e.get("failure_penalty", 1.0))
        self.success_bonus = float(e.get("success_bonus", 1.0))
        self.defender_loss_penalty = float(e.get("defender_loss_penalty", 0.0))

        # --- control defaults ---
        self.base_type = str(c.get("base_type", "attack")).lower()
        self.base_kp = float(c.get("base_kp", 0.15))
        self.base_kd = float(c.get("base_kd", 0.50))
        self.attack_kp = float(c.get("attack_kp", self.base_kp * 1.8))
        self.attack_kd = float(c.get("attack_kd", self.base_kd * 1.2))
        self.attack_lead_time = float(c.get("attack_lead_time", 3.0))
        self.attack_bias = float(c.get("attack_bias", 0.7))
        self.pn_nav_gain = float(c.get("pn_nav_gain", 3.0))
        self.base_alpha = float(c.get("base_alpha", 0.6))
        self.residual_gain = float(c.get("residual_gain", 1.0))
        self.manager_period = int(c.get("manager_period", 2))
        self.imitation_eps = float(c.get("imitation_min_budget", 0.1))

        # assignment stabilisation
        m = cfg.get("matcher", {})
        self.switch_penalty = float(m.get("switch_penalty", 12.0))
        self.assign_lock_steps = int(m.get("assign_lock_steps", 8))
        self.matcher_algo = str(m.get("algo", "hungarian"))

        # state
        self.rng = np.random.default_rng(seed=0)
        self.reset()

    # ----------------- observations -----------------
    @property
    def low_obs_dim(self) -> int:
        # [r(3), v_rel(3), d_vel(3), T_vel(3), has_target(1)] = 13
        return 13

    def _get_obs_low(self):
        low = np.zeros((self.nD, 13), dtype=np.float32)
        for i, d in enumerate(self.D):
            if not d.alive:
                continue
            j = self.curr_assign[i]
            has = 1.0 if (j >= 0 and self.P[j].alive) else 0.0
            if has > 0.5:
                p = self.P[j]
                r = p.pos - d.pos
                v_rel = p.vel - d.vel
            else:
                r = np.zeros(3, dtype=np.float32)
                v_rel = np.zeros(3, dtype=np.float32)
            low[i, :3] = r
            low[i, 3:6] = v_rel
            low[i, 6:9] = d.vel
            low[i, 9:12] = self.T_vel
            low[i, 12] = has
        return low

    def _get_obs_high(self):
        # cost matrix and mask used by assignment, approximated via distance / relative speed
        costs = np.full((self.nD, self.nP), 1e6, dtype=np.float32)
        mask = np.zeros((self.nD, self.nP), dtype=np.float32)
        for i, d in enumerate(self.D):
            if not d.alive:
                continue
            for j, p in enumerate(self.P):
                if not p.alive:
                    continue
                diff = p.pos - d.pos
                dist = np.linalg.norm(diff) + 1e-6
                rel_speed = np.linalg.norm(p.vel - d.vel) + 1e-6
                costs[i, j] = dist / rel_speed
                mask[i, j] = 1.0
        return {"costs": costs, "mask": mask}

    def _get_obs(self):
        return {"low": self._get_obs_low(), "high": self._get_obs_high()}

    # ----------------- reset -----------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        x_lo, x_hi = self.world_bounds[0]
        y_lo, y_hi = self.world_bounds[1]
        z_lo, z_hi = self.world_bounds[2]
        t_x = (x_lo + x_hi) * self.spawn_t_x_frac
        self.T = Entity(
            pos=np.array([t_x, 0.0, 0.0], dtype=np.float32),
            vel=self.T_vel.copy(),
            alive=True
        )

        # attackers: sampled distance based on expected closing time with lateral spread
        self.P = []
        for _ in range(self.nP):
            placed = False
            for _try in range(self.spawn_attempts):
                speed = self.rng.uniform(0.6 * self.p_v_max, 1.0 * self.p_v_max)
                closing_speed = speed + self.T_speed
                axial = closing_speed * self.front_hit_time
                anchor = self.T.pos + self.T_vel_unit * axial
                dy = self.rng.uniform(-self.lateral_spread_y, self.lateral_spread_y)
                dz = self.rng.uniform(-self.lateral_spread_z, self.lateral_spread_z)
                cand = anchor + np.array([0.0, dy, dz], dtype=np.float32)
                cand = np.clip(cand, self.world_bounds[:, 0], self.world_bounds[:, 1])
                if np.linalg.norm(cand - self.T.pos) < self.min_tp_dist:
                    continue
                too_close = False
                for pj in self.P:
                    if np.linalg.norm(cand - pj.pos) < self.min_pp_dist:
                        too_close = True
                        break
                if too_close:
                    continue
                dir_to_T = unit(self.T.pos - cand)
                p_vel = dir_to_T * speed
                self.P.append(Entity(pos=cand.astype(np.float32),
                                     vel=p_vel.astype(np.float32),
                                     alive=True))
                placed = True
                break

            if not placed:
                fallback_axial = (self.p_v_max + self.T_speed) * self.front_hit_time * 1.25
                cand = self.T.pos + self.T_vel_unit * fallback_axial
                cand = np.clip(cand, self.world_bounds[:, 0], self.world_bounds[:, 1])
                dir_to_T = unit(self.T.pos - cand)
                p_vel = dir_to_T * self.p_v_max
                self.P.append(Entity(pos=cand.astype(np.float32),
                                     vel=p_vel.astype(np.float32),
                                     alive=True))

        # defenders: escort ring around the target
        self.D = []
        for _ in range(self.nD):
            ang_y = self.rng.uniform(0, 2 * np.pi)
            ang_z = self.rng.uniform(0, 2 * np.pi)
            offset = np.array([
                -self.rng.uniform(15.0, self.d_guard_radius),
                np.cos(ang_y) * self.d_guard_radius * 0.2,
                np.sin(ang_z) * self.d_guard_radius * 0.2
            ], dtype=np.float32)
            d_pos = np.clip(self.T.pos + offset, self.world_bounds[:, 0], self.world_bounds[:, 1])
            d_vel = np.array([self.T_vel[0], 0.0, 0.0], dtype=np.float32)  # initial speed along +x
            self.D.append(Entity(pos=d_pos, vel=d_vel, alive=True))

        self.guard_offsets = np.array([d.pos - self.T.pos for d in self.D], dtype=np.float32)

        # manager state and initial assignment
        self.prev_assign = np.full(self.nD, -1, dtype=np.int32)
        self.curr_assign = np.full(self.nD, -1, dtype=np.int32)
        self.assign_lock = np.zeros(self.nD, dtype=np.int32)
        self.set_assignments_rule()
        return self._get_obs()

    def _assignment_distances(self) -> np.ndarray:
        dists = np.zeros(self.nD, dtype=np.float32)
        for i, d in enumerate(self.D):
            if not d.alive:
                continue
            j = self.curr_assign[i]
            if j >= 0 and self.P[j].alive:
                dists[i] = float(np.linalg.norm(self.P[j].pos - d.pos))
        return dists

    # ----------------- teacher / baseline -----------------
    def _attack_intercept_accel(self, d: Entity, p: Entity, limit: float) -> np.ndarray:
        """Compute an aggressive intercept acceleration using PN + predictive PD."""
        rel = p.pos - d.pos
        dist = np.linalg.norm(rel) + 1e-6
        los = rel / dist
        rel_vel = p.vel - d.vel
        closing = -np.dot(los, rel_vel)

        # Proportional navigation term encourages快速转向目标
        pn_term = np.cross(rel_vel, los)
        pn_term = np.cross(pn_term, los)
        a_pn = self.pn_nav_gain * closing * pn_term

        lead_time = max(0.5, self.attack_lead_time)
        intercept = p.pos + p.vel * lead_time
        desired_pos = intercept - d.pos
        desired_vel = p.vel
        a_pd = self.attack_kp * desired_pos + self.attack_kd * (desired_vel - d.vel)

        return clamp_norm(a_pn + a_pd, limit)

    def _base_accel(self, idx: int, d: Entity, p: Optional[Entity]) -> np.ndarray:
        """Baseline acceleration without residual learning."""
        mode = self.base_type
        if mode == "pd":
            if p is None:
                return np.zeros(3, dtype=np.float32)
            r = p.pos - d.pos
            v = p.vel - d.vel
            a_pd = self.base_kp * r + self.base_kd * v
            return clamp_norm(a_pd, self.base_alpha * self.d_a_max)

        if mode == "attack" and p is not None:
            return self._attack_intercept_accel(d, p, self.base_alpha * self.d_a_max)

        guard = self.guard_offsets[idx] if idx < len(self.guard_offsets) else np.zeros(3, dtype=np.float32)
        guard_target_pos = self.T.pos + guard
        guard_target_vel = self.T.vel

        if p is not None:
            # Escort/混合模式在保持护航的同时偏向指派攻击者
            lead_time = max(0.5, self.attack_lead_time)
            intercept = p.pos + p.vel * lead_time
            target_pos = (1.0 - self.attack_bias) * guard_target_pos + self.attack_bias * intercept
            target_vel = (1.0 - self.attack_bias) * guard_target_vel + self.attack_bias * p.vel
            kp = self.attack_kp
            kd = self.attack_kd
        else:
            target_pos = guard_target_pos
            target_vel = guard_target_vel
            kp = self.base_kp
            kd = self.base_kd

        pos_err = target_pos - d.pos
        vel_err = target_vel - d.vel
        a_cmd = kp * pos_err + kd * vel_err
        return clamp_norm(a_cmd, self.base_alpha * self.d_a_max)

    def _full_teacher(self, d: Entity, p: Optional[Entity]) -> np.ndarray:
        if p is None:
            return np.zeros(3, dtype=np.float32)
        if self.base_type == "attack":
            return self._attack_intercept_accel(d, p, self.d_a_max)
        r = p.pos - d.pos
        v = p.vel - d.vel
        a_full = self.base_kp * r + self.base_kd * v
        a_full = clamp_norm(a_full, self.d_a_max)
        return a_full

    # ----------------- assignment API -----------------
    def set_assignments_rule(self):
        """Stage one: rule-based assignment with lockout and switch penalty."""
        high = self._get_obs_high()
        costs, mask = high["costs"], high["mask"]
        new_assign = assign_targets(
            costs=costs,
            mask=mask,
            prev_assign=self.prev_assign,
            switch_penalty=self.switch_penalty,
            lock_steps=self.assign_lock_steps,
            algo=self.matcher_algo
        )
        for i, d in enumerate(self.D):
            if not d.alive:
                new_assign[i] = -1
        self.curr_assign = new_assign.copy()
        self.prev_assign = new_assign.copy()

    # ----------------- step -----------------
    def step(self, action_residual: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Args:
            action_residual: (nD, 3) values in [-1, 1] from the learned residual controller.
        """
        assert action_residual.shape == (self.nD, 3)
        self.t += 1
        done = False
        reward = 0.0

        prev_assign = self.curr_assign.copy()
        prev_dists = self._assignment_distances()

        # update assignments periodically
        if self.t % max(1, self.manager_period) == 0:
            self.set_assignments_rule()

        pn_actions = np.zeros((self.nD, 3), dtype=np.float32)
        pn_valid = np.zeros((self.nD,), dtype=np.float32)
        res_budget = np.zeros((self.nD,), dtype=np.float32)
        closing_rates = np.zeros((self.nD,), dtype=np.float32)

        # defender propagation
        for i, d in enumerate(self.D):
            if not d.alive:
                pn_actions[i] = np.zeros(3, dtype=np.float32)
                pn_valid[i] = 0.0
                res_budget[i] = 0.0
                continue
            j = self.curr_assign[i]
            p = self.P[j] if (j >= 0 and self.P[j].alive) else None

            a_base = self._base_accel(i, d, p)
            a_full = self._full_teacher(d, p)

            budget = max(0.0, self.d_a_max - np.linalg.norm(a_base) + 1e-6)
            res_budget[i] = budget
            max_residual = budget * self.residual_gain

            a_res = np.clip(action_residual[i], -1.0, 1.0) * max_residual
            a_cmd = clamp_norm(a_base + a_res, self.d_a_max)

            if p is not None and budget > self.imitation_eps and max_residual > 1e-6:
                teacher_residual = clamp_norm(a_full - a_base, max_residual)
                pn_actions[i] = teacher_residual
                pn_valid[i] = 1.0
            else:
                pn_actions[i] = np.zeros(3, dtype=np.float32)
                pn_valid[i] = 0.0

            d.vel = clamp_norm(d.vel + a_cmd * self.dt, self.d_v_max)
            d.pos = d.pos + d.vel * self.dt

            if p is not None:
                rel = p.pos - d.pos
                dist = np.linalg.norm(rel)
                if dist > 1e-6:
                    los = rel / dist
                    rel_vel = d.vel - p.vel
                    closing = -float(np.dot(los, rel_vel))
                    if closing > 0.0:
                        closing_rates[i] = closing

        new_dists = self._assignment_distances()
        approach_delta = np.zeros_like(prev_dists)
        if self.approach_reward_scale > 0.0:
            for i in range(self.nD):
                if self.curr_assign[i] == prev_assign[i] and self.curr_assign[i] >= 0:
                    j = self.curr_assign[i]
                    if self.P[j].alive:
                        approach_delta[i] = prev_dists[i] - new_dists[i]
            reward += float(np.sum(approach_delta) * self.approach_reward_scale)

        if self.closing_reward_scale > 0.0:
            reward += float(np.sum(closing_rates) * self.dt * self.closing_reward_scale)

        # attacker propagation (toward the target)
        for p in self.P:
            if not p.alive:
                continue
            to_T = unit(self.T.pos - p.pos)
            p.vel = clamp_norm(p.vel + to_T * self.p_a_max * self.dt, self.p_v_max)
            p.pos = p.pos + p.vel * self.dt

        if self.T.alive:
            self.T.pos = self.T.pos + self.T_vel * self.dt

        # contact checks
        # attacker reaches the target -> failure
        for p in self.P:
            if not p.alive:
                continue
            if within_radius(p.pos, self.T.pos, self.t_attack_radius):
                done = True
                reward -= self.failure_penalty
                self.T.alive = False
                break

        # defender kills attacker
        if self.T.alive:
            for p in self.P:
                if not p.alive:
                    continue

                for i, d in enumerate(self.D):
                    if not d.alive:
                        continue

                    if within_radius(d.pos, p.pos, self.d_attack_radius):
                        p.alive = False
                        d.alive = False
                        self.curr_assign[i] = -1
                        self.prev_assign[i] = -1
                        reward += self.kill_reward
                        if self.defender_loss_penalty > 0.0:
                            reward -= self.defender_loss_penalty
                        break

        # success if all attackers are destroyed
        if not any(p.alive for p in self.P):
            done = True
            reward += self.success_bonus

        # small time penalty
        reward -= self.time_penalty

        # constrain to world bounds
        for ent in [self.T] + self.P + self.D:
            ent.pos = np.clip(ent.pos, self.world_bounds[:, 0], self.world_bounds[:, 1])

        obs = self._get_obs()
        info = {
            "pn_action": pn_actions,      # (nD, 3) teacher residual capped by available budget
            "pn_valid": pn_valid,         # (nD,)
            "res_budget": res_budget,     # (nD,)
            "assign": self.curr_assign.copy(),
            "alive_P": np.array([p.alive for p in self.P], dtype=np.int32),
            "alive_D": np.array([d.alive for d in self.D], dtype=np.int32),
            "approach_delta": approach_delta,
            "closing_rate": closing_rates,
        }

        # evaluation helpers
        attackers_alive = int(np.count_nonzero(info["alive_P"]))
        info["attackers_alive"] = attackers_alive
        info["success"] = (attackers_alive == 0)
        info["done_success"] = bool(info["success"])
        info["done_failure"] = bool(done and not info["success"])

        if self.t >= self.max_steps:
            done = True

        return obs, float(reward), bool(done), info



















