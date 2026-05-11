"""FormationHallwayEnv — dynamic-cluster-size formation control with teleop.

4 robots in a long hallway. Target formation switches with the active
(non-teleop'd) cluster size: 4=square, 3=triangle, 2=line, 1=solo.

Built on the same vector_reset/vector_step pattern as env_line.PassageEnv
so train.py / eval.py infrastructure transfers with minimal changes.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pygame
import torch
from scipy.optimize import linear_sum_assignment

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contract import (
    AGENT_RADIUS,
    DEFAULT_MAX_TIME_STEPS,
    DEFAULT_RENDER_PX_PER_M,
    DT,
    FORMATION_SCALE,
    GOAL_Y,
    MAX_A,
    MAX_AGENTS,
    MAX_V,
    MIN_A,
    REWARD_COEFFS,
    SPAWN_Y,
    WORLD_H,
    WORLD_W,
)

X = 0
Y = 1


def target_formation_positions(n: int, scale: float = FORMATION_SCALE) -> torch.Tensor:
    """Return canonical formation slot positions, centred at origin.

    n=4: square (vertices at +/- s/2)
    n=3: equilateral triangle, one vertex pointing +y
    n=2: horizontal line
    n=1: single point at origin
    """
    s = scale
    if n == 4:
        return torch.tensor(
            [[-s / 2, -s / 2], [s / 2, -s / 2], [s / 2, s / 2], [-s / 2, s / 2]],
            dtype=torch.float32,
        )
    if n == 3:
        # equilateral triangle, side s, centroid at origin, one vertex up
        h = s / (2 * 3 ** 0.5)
        H = s / (3 ** 0.5)
        return torch.tensor(
            [[-s / 2, -h], [s / 2, -h], [0.0, H]],
            dtype=torch.float32,
        )
    if n == 2:
        return torch.tensor([[-s / 2, 0.0], [s / 2, 0.0]], dtype=torch.float32)
    if n == 1:
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    raise ValueError(f"unsupported active_count={n}")


class FormationHallwayEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[dict] = None):
        cfg = dict(config or {})
        cfg.setdefault("n_agents", MAX_AGENTS)
        cfg.setdefault("num_envs", 1)
        cfg.setdefault("dt", DT)
        cfg.setdefault("device", "cpu")
        cfg.setdefault("world_dim", (WORLD_W, WORLD_H))
        cfg.setdefault("max_v", MAX_V)
        cfg.setdefault("max_a", MAX_A)
        cfg.setdefault("min_a", MIN_A)
        cfg.setdefault("agent_radius", AGENT_RADIUS)
        cfg.setdefault("agent_radius_start", cfg["agent_radius"])
        cfg.setdefault("agent_radius_end", cfg["agent_radius"])
        cfg.setdefault("agent_radius_curriculum_steps", 1)
        cfg.setdefault("max_time_steps", DEFAULT_MAX_TIME_STEPS)
        cfg.setdefault("pos_noise_std", 0.0)
        cfg.setdefault("formation_scale", FORMATION_SCALE)
        cfg.setdefault("render_px_per_m", DEFAULT_RENDER_PX_PER_M)
        cfg.setdefault("spawn_y", SPAWN_Y)
        cfg.setdefault("goal_y", GOAL_Y)
        cfg.setdefault("reward_coeffs", dict(REWARD_COEFFS))
        cfg.setdefault("render", False)
        self.cfg = cfg

        n = self.cfg["n_agents"]
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=-cfg["max_v"], high=cfg["max_v"], shape=(2,), dtype=float),) * n
        )
        max_t = cfg["max_time_steps"] * cfg["dt"]
        self.observation_space = gym.spaces.Dict(
            {
                "pos": gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
                "vel": gym.spaces.Box(-1e5, 1e5, shape=(n, 2), dtype=float),
                "goal": gym.spaces.Box(-WORLD_H, WORLD_H, shape=(n, 2), dtype=float),
                "teleop_mask": gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=float),
                "present_mask": gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=float),
                "time": gym.spaces.Box(0.0, max_t, shape=(n, 1), dtype=float),
            }
        )

        self.device = torch.device(cfg["device"])
        self._curriculum_step = 0
        self.vec_p_shape = (cfg["num_envs"], n, 2)

        # stall detection: ring buffer of cluster centroids per env
        self._stall_window = int(cfg["reward_coeffs"]["stall_window"])
        self._centroid_history: List[deque] = [
            deque(maxlen=self._stall_window) for _ in range(cfg["num_envs"])
        ]

        self.vector_reset()

        self.display = None
        if cfg.get("render", False):
            pygame.init()
            size = (
                int(cfg["world_dim"][0] * cfg["render_px_per_m"]),
                int(cfg["world_dim"][1] * cfg["render_px_per_m"]),
            )
            self.display = pygame.display.set_mode(size)

    # --- utilities -------------------------------------------------------
    def create_state_tensor(self) -> torch.Tensor:
        return torch.zeros(self.vec_p_shape, dtype=torch.float32, device=self.device)

    def sample_pos_noise(self) -> torch.Tensor:
        std = self.cfg["pos_noise_std"]
        if std > 0.0:
            return torch.normal(0.0, std, self.vec_p_shape, device=self.device)
        return self.create_state_tensor()

    def compute_agent_dists(self, ps: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(ps, ps)
        n = ps.shape[1]
        diag = torch.eye(n, device=ps.device).bool().unsqueeze(0).expand(ps.shape[0], -1, -1)
        d = d.masked_fill(diag, float("inf"))
        return d

    def rand(self, size, a: float, b: float) -> torch.Tensor:
        return (b - a) * torch.rand(size, device=self.device) + a

    def set_curriculum_step(self, step: int):
        self._curriculum_step = max(0, int(step))

    def current_agent_radius(self) -> float:
        start = float(self.cfg["agent_radius_start"])
        end = float(self.cfg["agent_radius_end"])
        total = max(1, int(self.cfg["agent_radius_curriculum_steps"]))
        alpha = min(1.0, self._curriculum_step / total)
        return start + alpha * (end - start)

    # --- formation helper -----------------------------------------------
    def target_formation_positions(self, n: int) -> torch.Tensor:
        return target_formation_positions(n, self.cfg["formation_scale"])

    # --- spawn / reset ---------------------------------------------------
    def get_starts_and_goals(self, num_envs: int):
        n = self.cfg["n_agents"]
        # spawn near (0, SPAWN_Y) in a small jittered square
        base = self.target_formation_positions(n)
        starts = base.unsqueeze(0).repeat(num_envs, 1, 1).to(self.device)
        starts[:, :, X] += self.rand((num_envs, n), -0.05, 0.05)
        starts[:, :, Y] += self.rand((num_envs, n), -0.05, 0.05) + self.cfg["spawn_y"]
        goals = torch.zeros(num_envs, n, 2, device=self.device)
        goals[:, :, Y] = self.cfg["goal_y"]
        return starts, goals

    def vector_reset(self):
        starts, goals = self.get_starts_and_goals(self.cfg["num_envs"])
        self.ps = starts
        self.goal_ps = goals
        self.measured_vs = self.create_state_tensor()
        self.teleop_mask = torch.zeros(
            self.cfg["num_envs"], self.cfg["n_agents"], dtype=torch.float32, device=self.device
        )
        self.present_mask = torch.ones_like(self.teleop_mask)
        self.teleop_vels = self.create_state_tensor()
        self.timesteps = torch.zeros(self.cfg["num_envs"], dtype=torch.int32, device=self.device)
        self.goal_reached = torch.zeros(self.cfg["num_envs"], dtype=torch.bool, device=self.device)
        for h in self._centroid_history:
            h.clear()
        return [self.get_obs(i) for i in range(self.cfg["num_envs"])]

    def reset_at(self, index: int):
        start, goal = self.get_starts_and_goals(1)
        self.ps[index] = start[0]
        self.goal_ps[index] = goal[0]
        self.measured_vs[index] = 0.0
        self.teleop_mask[index] = 0.0
        self.present_mask[index] = 1.0
        self.teleop_vels[index] = 0.0
        self.timesteps[index] = 0
        self.goal_reached[index] = False
        self._centroid_history[index].clear()
        return self.get_obs(index)

    # --- teleop interface (used by training disturbance + keyboard) -----
    def set_teleop(self, env_idx: int, robot_idx: int, active: bool):
        self.teleop_mask[env_idx, robot_idx] = 1.0 if active else 0.0
        if not active:
            self.teleop_vels[env_idx, robot_idx] = 0.0

    def set_teleop_action(self, env_idx: int, robot_idx: int, vel):
        v = torch.as_tensor(vel, dtype=torch.float32, device=self.device)
        v = torch.clamp(v, -self.cfg["max_v"], self.cfg["max_v"])
        self.teleop_vels[env_idx, robot_idx] = v

    # --- obs -------------------------------------------------------------
    def get_obs(self, index: int):
        n = self.cfg["n_agents"]
        t = (self.timesteps[index] * self.cfg["dt"]).item()
        return {
            "pos": self.ps[index].tolist(),
            "vel": self.measured_vs[index].tolist(),
            "goal": self.goal_ps[index].tolist(),
            "teleop_mask": self.teleop_mask[index].tolist(),
            "present_mask": self.present_mask[index].tolist(),
            "time": [[t]] * n,
        }

    # --- reward components ----------------------------------------------
    def _formation_reward(self, env_idx: int, ps: torch.Tensor):
        """Per-robot formation penalty for env_idx.

        Hungarian-assigns active (non-teleop'd) robots to slots of the target
        formation centred at the active-robot centroid. Returns a tuple
        (per_robot_penalty (n_agents,), mean_slot_distance or None).
        """
        n = self.cfg["n_agents"]
        out = torch.zeros(n, device=ps.device)
        active_idx = (self.teleop_mask[env_idx] < 0.5).nonzero(as_tuple=True)[0]
        k = active_idx.numel()
        if k < 2:
            return out, None
        active_ps = ps[active_idx]  # (k, 2)
        centroid = active_ps.mean(dim=0)
        slots = self.target_formation_positions(k).to(ps.device) + centroid  # (k, 2)
        cost = torch.cdist(active_ps, slots).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        coeff = float(self.cfg["reward_coeffs"]["k_form"]) * (k / max(1, n))
        dists = []
        for ri, ci in zip(row, col):
            robot = int(active_idx[ri].item())
            d = float(cost[ri, ci])
            dists.append(d)
            out[robot] = -coeff * d
        return out, sum(dists) / len(dists)

    def _stall_penalty(self, env_idx: int, centroid: torch.Tensor) -> float:
        h = self._centroid_history[env_idx]
        h.append(centroid.detach().clone())
        if len(h) < self._stall_window:
            return 0.0
        moved = float(torch.linalg.norm(h[-1] - h[0]).item())
        if moved < self.cfg["reward_coeffs"]["stall_eps"]:
            return -float(self.cfg["reward_coeffs"]["k_stall"])
        return 0.0

    # --- step -----------------------------------------------------------
    def vector_step(self, actions):
        cfg = self.cfg
        n = cfg["n_agents"]
        nE = cfg["num_envs"]
        coeffs = cfg["reward_coeffs"]

        actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        # Override teleop slots with stored teleop velocities
        mask3 = self.teleop_mask.unsqueeze(-1)  # (nE, n, 1)
        actions_t = actions_t * (1.0 - mask3) + self.teleop_vels * mask3

        desired_vs = torch.clip(actions_t, -cfg["max_v"], cfg["max_v"])
        desired_as = (desired_vs - self.measured_vs) / cfg["dt"]
        possible_as = torch.clip(desired_as, cfg["min_a"], cfg["max_a"])
        possible_vs = self.measured_vs + possible_as * cfg["dt"]

        previous_ps = self.ps.clone()
        rewards = torch.zeros(nE, n, device=self.device)

        # Per-agent collision check — update step-by-step so blamed agent eats penalty
        next_ps = self.ps.clone()
        for i in range(n):
            trial = next_ps.clone()
            trial[:, i] += possible_vs[:, i] * cfg["dt"]
            d = self.compute_agent_dists(trial)[:, i]  # (nE, n) infs on diag
            collide = torch.min(d, dim=1)[0] <= 2 * self.current_agent_radius()
            next_ps[~collide, i] = trial[~collide, i]
            rewards[collide, i] -= coeffs["k_coll"]

        # Wall containment in x
        half_w = cfg["world_dim"][0] / 2.0 - self.current_agent_radius()
        overshoot_x = (next_ps[:, :, X].abs() - half_w).clamp(min=0.0)
        rewards -= coeffs["k_wall"] * overshoot_x
        next_ps[:, :, X] = torch.clip(next_ps[:, :, X], -half_w, half_w)

        # Y soft bounds — clip to world height
        half_h = cfg["world_dim"][1] / 2.0
        next_ps[:, :, Y] = torch.clip(next_ps[:, :, Y], -half_h, half_h)

        next_ps += self.sample_pos_noise()
        self.ps = next_ps
        self.measured_vs = (self.ps - previous_ps) / cfg["dt"]

        # Forward progress (credited to active robots; later modulated by formation error)
        dy = (self.ps[:, :, Y] - previous_ps[:, :, Y])
        active = 1.0 - self.teleop_mask
        fwd_reward = coeffs["k_fwd"] * dy * active

        # Formation + stall + goal — per env
        formation_errs: List[Optional[float]] = [None] * nE
        stalled_flags = [False] * nE
        for e in range(nE):
            penalty, err = self._formation_reward(e, self.ps[e])
            # If active robots are far from desired formation, downweight forward reward.
            if err is not None:
                beta = float(coeffs.get("fwd_form_beta", 0.0))
                floor = float(coeffs.get("fwd_scale_min", 0.0))
                scale = max(floor, float(np.exp(-beta * float(err))))
            else:
                scale = 1.0
            rewards[e] += fwd_reward[e] * scale
            rewards[e] += penalty * active[e]
            formation_errs[e] = err

            active_e = active[e].bool()
            if active_e.any():
                centroid = self.ps[e][active_e].mean(dim=0)
                stall_pen = self._stall_penalty(e, centroid)
                if stall_pen != 0.0:
                    rewards[e] += stall_pen * active[e]
                    stalled_flags[e] = True

                # Goal bonus: cluster centroid past GOAL_Y (one-shot)
                if not bool(self.goal_reached[e]) and float(centroid[Y].item()) >= cfg["goal_y"]:
                    rewards[e] += coeffs["k_goal"] * active[e]
                    self.goal_reached[e] = True

        # Zero reward for teleop'd robots (will be masked in trainer too)
        rewards = rewards * active

        self.timesteps += 1
        timeout = self.timesteps >= cfg["max_time_steps"]
        dones = (timeout | self.goal_reached).tolist()
        obs = [self.get_obs(i) for i in range(nE)]
        infos = []
        for e in range(nE):
            active_e = active[e].bool()
            mean_dy = (
                float((dy[e][active_e]).mean().item()) / cfg["dt"] if active_e.any() else 0.0
            )
            wall_hit = bool((overshoot_x[e] > 0).any().item())
            collided = bool((rewards[e] <= -coeffs["k_coll"] + 1e-6).any().item())
            infos.append(
                {
                    "rewards": {k: float(rewards[e, k].item()) for k in range(n)},
                    "active_count": int(active[e].sum().item()),
                    "goal_reached": bool(self.goal_reached[e].item()),
                    "formation_error": formation_errs[e],
                    "fwd_velocity": mean_dy,
                    "stalled": stalled_flags[e],
                    "wall_hit": wall_hit,
                    "collided": collided,
                }
            )
        return obs, torch.sum(rewards, dim=1).tolist(), dones, infos

    def get_unwrapped(self):
        return []

    def close(self):
        if self.display is not None:
            pygame.display.quit()
            pygame.quit()
            self.display = None


# Lightweight single-env render wrapper — mirrors PassageEnvRender from env_line
class FormationHallwayEnvRender(FormationHallwayEnv):
    def __init__(self, config: Optional[dict] = None):
        config = dict(config or {})
        config["num_envs"] = 1
        config.setdefault("render", True)
        super().__init__(config)

    def reset(self):
        return self.reset_at(0)

    def step(self, actions):
        a = np.zeros((1, self.cfg["n_agents"], 2), dtype=np.float32)
        a[0] = np.asarray(actions, dtype=np.float32)
        obs, r, done, info = self.vector_step(a)
        return obs[0], r[0], done[0], info[0]


def _smoke_test(steps: int = 200, seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = FormationHallwayEnv({"num_envs": 4})
    env.vector_reset()
    rng = np.random.default_rng(seed)
    total_r = np.zeros(env.cfg["num_envs"])
    for t in range(steps):
        a = rng.uniform(-MAX_V, MAX_V, size=(env.cfg["num_envs"], env.cfg["n_agents"], 2))
        obs, r, done, info = env.vector_step(a)
        total_r += np.array(r)
        if t == 50:
            env.set_teleop(0, 1, True)
            env.set_teleop_action(0, 1, np.array([0.5, 0.0]))
        if any(done):
            for i, d in enumerate(done):
                if d:
                    env.reset_at(i)
    print(f"smoke OK: steps={steps}, mean total reward={total_r.mean():.3f}")
    print(f"final teleop_mask[0]={env.teleop_mask[0].tolist()}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--random", action="store_true")
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    if args.random:
        _smoke_test(args.steps)
    else:
        _smoke_test(args.steps)
