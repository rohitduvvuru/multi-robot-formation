"""Teleop drivers for FormationHallwayEnv.

RandomTeleop  — synthetic disturbance for training. Drives all four
mechanisms the policy needs to be robust to: grab a robot, release it,
spawn a new robot, delete one. Initial active count is sampled per episode
so the policy sees the full n_present in [1..10] regime spectrum.

KeyboardTeleop — pygame key handler for the eval/demo binary.
  1-9       toggle teleop on robot 1-9
  0         toggle teleop on robot 10
  W A S D   drive the most-recently-selected teleop robot
  Z / X     decrease / increase the teleop drive speed
  =  / +    spawn a new robot (no-op if already at MAX_AGENTS=10)
  -  / _    delete the selected robot (no-op if at MIN_AGENTS=1)
  R         release all teleop'd robots

Both call env.set_teleop / env.set_teleop_action / env.spawn / env.delete —
see code/contract.py.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contract import MAX_AGENTS, MAX_V, MIN_AGENTS


@dataclass
class _Grab:
    robot: int
    age: int
    duration: int
    drift_dir: float   # +1 / -1 lateral push (0 -> hold still)
    base_vy: float
    drift_speed: float


# Default mass on each n_present in {1..MAX_AGENTS} at episode start.
# Keep this uniform so PPO sees every regime equally often by default.
def _default_init_n_present_dist() -> List[float]:
    return [1.0 / MAX_AGENTS] * MAX_AGENTS


DEFAULT_INIT_N_PRESENT_DIST = _default_init_n_present_dist()


class RandomTeleop:
    """Per-env synthetic disturbance for training.

    Per step, four independent Bernoulli events:
      * grab a present, non-teleop'd robot for a random duration
      * release a teleop'd robot early
      * spawn a new robot (if n_present < MAX_AGENTS)
      * delete a present, non-teleop'd robot (if n_present > MIN_AGENTS)

    On reset_env, sample the initial n_present from `init_n_present_dist`
    over [1..MAX_AGENTS] and pre-populate present_mask accordingly.
    """

    def __init__(
        self,
        env,
        p_grab: float = 0.005,
        p_release: float = 0.01,
        p_spawn: float = 0.002,
        p_delete: float = 0.002,
        drift_speed: float = 0.6,
        min_duration: int = 40,
        max_duration: int = 160,
        max_concurrent_grabs: int = 3,
        init_n_present_dist: Optional[List[float]] = None,
        init_hold_min: int = 30,
        init_hold_max: int = 80,
        seed: int = 0,
    ):
        self.env = env
        self.num_envs = env.cfg["num_envs"]
        self.n_agents = env.cfg["n_agents"]
        if self.n_agents != MAX_AGENTS:
            raise ValueError(
                f"RandomTeleop assumes env.n_agents==MAX_AGENTS={MAX_AGENTS}; "
                f"got {self.n_agents}"
            )
        self.p_grab = p_grab
        self.p_release = p_release
        self.p_spawn = p_spawn
        self.p_delete = p_delete
        self.drift_speed = drift_speed
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_concurrent_grabs = max(0, min(max_concurrent_grabs, MAX_AGENTS - 1))
        if init_n_present_dist is None:
            init_n_present_dist = list(DEFAULT_INIT_N_PRESENT_DIST)
        if len(init_n_present_dist) != MAX_AGENTS:
            raise ValueError(
                f"init_n_present_dist must have length MAX_AGENTS={MAX_AGENTS}; "
                f"got {len(init_n_present_dist)}"
            )
        s = float(sum(init_n_present_dist))
        if s <= 0:
            raise ValueError("init_n_present_dist must sum to a positive value")
        self.init_n_present_dist = [p / s for p in init_n_present_dist]
        self.init_hold_min = init_hold_min
        self.init_hold_max = init_hold_max
        self.rng = np.random.default_rng(seed)
        # per-env dict of {robot_idx -> _Grab}; robot is teleop'd iff present here
        self.grabs: List[dict] = [{} for _ in range(self.num_envs)]
        self.steps = np.zeros(self.num_envs, dtype=np.int64)

    # ---- helpers --------------------------------------------------------
    def _sample_initial_n_present(self) -> int:
        """Sample n_present in {1..MAX_AGENTS} from init_n_present_dist."""
        return int(self.rng.choice(MAX_AGENTS, p=self.init_n_present_dist)) + 1

    def _present_robots(self, env_idx: int) -> List[int]:
        return [
            i for i in range(self.n_agents)
            if float(self.env.present_mask[env_idx, i]) > 0.5
        ]

    def _start_grab(self, env_idx: int, robot: int):
        duration = int(self.rng.integers(self.min_duration, self.max_duration))
        drift_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        base_vy = float(self.rng.uniform(0.0, 0.5))
        grab = _Grab(
            robot=robot, age=0, duration=duration,
            drift_dir=drift_dir, base_vy=base_vy, drift_speed=self.drift_speed,
        )
        self.grabs[env_idx][robot] = grab
        self.env.set_teleop(env_idx, robot, True)

    def _release_grab(self, env_idx: int, robot: int):
        self.env.set_teleop(env_idx, robot, False)
        self.grabs[env_idx].pop(robot, None)

    # ---- public API -----------------------------------------------------
    def reset_env(self, env_idx: int):
        # release any holdovers from the previous episode
        for r in list(self.grabs[env_idx].keys()):
            self.env.set_teleop(env_idx, r, False)
        self.grabs[env_idx] = {}
        self.steps[env_idx] = 0

        # The env's reset_at already populated present_mask with INITIAL_AGENTS
        # robots in slots [0, INITIAL_AGENTS). Reshape that to match the sampled
        # initial n_present by spawn'ing or delete'ing as needed.
        target_n = self._sample_initial_n_present()
        cur = self.env.n_present(env_idx)
        while cur < target_n:
            if self.env.spawn(env_idx) is None:
                break
            cur += 1
        while cur > target_n:
            present = self._present_robots(env_idx)
            if len(present) <= MIN_AGENTS:
                break
            r = int(self.rng.choice(present))
            if not self.env.delete(env_idx, r):
                break
            cur -= 1

    def step(self):
        """Call once per env timestep, BEFORE env.vector_step.

        Toggles teleop_mask / present_mask via env API in place.
        """
        for e in range(self.num_envs):
            self.steps[e] += 1
            n_pres = self.env.n_present(e)

            # spawn / delete events first (they can change which robots are grabbable)
            if n_pres < MAX_AGENTS and self.rng.random() < self.p_spawn:
                self.env.spawn(e)
                n_pres = self.env.n_present(e)

            if n_pres > MIN_AGENTS and self.rng.random() < self.p_delete:
                # only delete present, NON-teleop'd robots so we don't yank a grab
                cands = [
                    i for i in self._present_robots(e)
                    if i not in self.grabs[e]
                ]
                if cands:
                    r = int(self.rng.choice(cands))
                    if self.env.delete(e, r):
                        n_pres -= 1

            # try to start new grabs on currently-untouched, present robots
            current = set(self.grabs[e].keys())
            slots_free = self.max_concurrent_grabs - len(current)
            # cap: never grab so many that n_active drops to 0
            max_grab_now = max(0, n_pres - 1)  # leave at least one active
            slots_free = min(slots_free, max_grab_now - len(current))
            if slots_free > 0:
                cands = [
                    i for i in self._present_robots(e)
                    if i not in current
                ]
                self.rng.shuffle(cands)
                for r in cands:
                    if slots_free <= 0:
                        break
                    if self.rng.random() < self.p_grab:
                        self._start_grab(e, r)
                        current.add(r)
                        slots_free -= 1

            # advance / release existing grabs and push their teleop velocities
            for r in list(self.grabs[e].keys()):
                # If env deleted this robot for us, drop the grab
                if float(self.env.present_mask[e, r]) < 0.5:
                    self.grabs[e].pop(r, None)
                    continue
                grab = self.grabs[e][r]
                grab.age += 1
                if grab.age >= grab.duration or self.rng.random() < self.p_release:
                    self._release_grab(e, r)
                    continue
                phase = grab.age * 0.15
                vx = grab.drift_speed * grab.drift_dir * float(np.cos(phase))
                vy = grab.base_vy
                self.env.set_teleop_action(e, r, np.array([vx, vy], dtype=np.float32))


class KeyboardTeleop:
    """Pygame-key-driven teleop for the demo/eval window.

    Keys:
      1-9        toggle teleop on robot idx 0-8
      0          toggle teleop on robot idx 9
      W A S D    drive the most-recently-selected teleop robot
      Z / X      decrease / increase teleop drive speed (0.25 m/s steps)
      =  / +     spawn a new robot
      -  / _     delete the selected (or highest-index) robot
      R          release all teleop'd robots
    """

    DRIVE_KEYS = {
        ord("w"): np.array([0.0, 1.0]),
        ord("s"): np.array([0.0, -1.0]),
        ord("a"): np.array([-1.0, 0.0]),
        ord("d"): np.array([1.0, 0.0]),
    }
    SPEED_STEP = 0.25
    SPEED_MIN = 0.25
    SPEED_MAX = 2.5

    def __init__(self, env, env_idx: int = 0, drive_speed: float = MAX_V):
        self.env = env
        self.env_idx = env_idx
        self.n_agents = env.cfg["n_agents"]
        self.drive_speed = float(drive_speed)
        self.selected: Optional[int] = None
        self.pressed: set = set()

    # ---- helpers --------------------------------------------------------
    def _key_to_robot_idx(self, ch: str) -> Optional[int]:
        if ch in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            return int(ch) - 1
        if ch == "0":
            return 9
        return None

    def _toggle(self, robot: int):
        if not (0 <= robot < self.n_agents):
            return
        if float(self.env.present_mask[self.env_idx, robot]) < 0.5:
            return  # can't teleop a non-present robot
        active_now = float(self.env.teleop_mask[self.env_idx, robot]) > 0.5
        self.env.set_teleop(self.env_idx, robot, not active_now)
        if not active_now:
            self.selected = robot
        elif self.selected == robot:
            self.selected = None

    def _release_all(self):
        for r in range(self.n_agents):
            self.env.set_teleop(self.env_idx, r, False)
        self.selected = None

    def _spawn(self):
        self.env.spawn(self.env_idx)

    def _delete(self):
        # prefer the currently-selected (teleop'd) robot
        target = self.selected
        if target is None or float(self.env.present_mask[self.env_idx, target]) < 0.5:
            present = [
                i for i in range(self.n_agents)
                if float(self.env.present_mask[self.env_idx, i]) > 0.5
            ]
            if not present:
                return
            target = present[-1]
        if self.env.delete(self.env_idx, target):
            if self.selected == target:
                self.selected = None

    # ---- pygame plumbing ------------------------------------------------
    def handle_event(self, event):
        import pygame

        if event.type == pygame.KEYDOWN:
            ch = event.unicode.lower() if event.unicode else ""
            robot = self._key_to_robot_idx(ch) if ch else None
            if robot is not None:
                self._toggle(robot)
            elif ch == "r":
                self._release_all()
            elif ch in ("=", "+"):
                self._spawn()
            elif ch in ("-", "_"):
                self._delete()
            elif ch == "z":
                self.drive_speed = max(self.SPEED_MIN, self.drive_speed - self.SPEED_STEP)
            elif ch == "x":
                self.drive_speed = min(self.SPEED_MAX, self.drive_speed + self.SPEED_STEP)
            elif event.key in self.DRIVE_KEYS:
                self.pressed.add(event.key)
        elif event.type == pygame.KEYUP and event.key in self.DRIVE_KEYS:
            self.pressed.discard(event.key)

    def apply(self):
        """Compute desired velocity for the selected teleop robot from the
        current set of pressed keys and push it via set_teleop_action.

        Call once per frame, after pumping pygame events.
        """
        v = np.zeros(2, dtype=np.float32)
        if self.selected is not None and self.pressed:
            for k in self.pressed:
                v += self.DRIVE_KEYS[k]
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm * self.drive_speed
        for r in range(self.n_agents):
            if float(self.env.teleop_mask[self.env_idx, r]) > 0.5:
                if r == self.selected:
                    self.env.set_teleop_action(self.env_idx, r, v)
                else:
                    # held in place
                    self.env.set_teleop_action(self.env_idx, r, np.zeros(2, dtype=np.float32))


def _demo():
    """Run RandomTeleop against the real env for a few hundred steps and
    report the n_present trajectory so we can confirm coverage.
    """
    if __package__ in (None, ""):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env_hallway import FormationHallwayEnv

    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    rt = RandomTeleop(env, p_grab=0.05, p_release=0.05, p_spawn=0.05, p_delete=0.05, seed=1)
    rt.reset_env(0)
    seen_counts = set()
    for t in range(800):
        rt.step()
        env.vector_step(np.zeros((env.cfg["num_envs"], env.cfg["n_agents"], 2)))
        seen_counts.add(env.n_present(0))
    print(f"demo OK: 800 steps, n_present values seen={sorted(seen_counts)}")
    print(f"final present_mask=\n{env.present_mask}")
    print(f"final teleop_mask=\n{env.teleop_mask}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    _demo()