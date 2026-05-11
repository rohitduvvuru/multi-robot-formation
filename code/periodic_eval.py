"""Periodic eval during training + best.pt symlink management.

Used by train_hallway.py at every checkpoint. Runs a quick fixed-n-present
eval over a configurable list of cluster sizes, scores the result with a
worst-regime-first rule, and (if it improves) points weights/best.pt at
the just-saved checkpoint.

Self-contained — does not import anything from eval_hallway.py. The CLI
eval tool stays separate.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional

import numpy as np
import torch

from contract import MAX_AGENTS, MIN_AGENTS
from env_hallway import FormationHallwayEnv
from metrics import EpisodeAccumulator, RunLogger
from teleop import RandomTeleop


def run_episodes(
    *,
    agent,
    device,
    n_present: int,
    episodes: int,
    max_steps: int,
    seed: int = 0,
) -> dict:
    """Run `episodes` eval episodes with `n_present` robots fixed.

    RandomTeleop runs grab/release at default rates (so the policy sees the
    same disturbance pattern training uses) but spawn/delete are zeroed and
    the initial-regime distribution is one-hot at `n_present`.
    """
    if not (MIN_AGENTS <= n_present <= MAX_AGENTS):
        raise ValueError(
            f"n_present must be in [{MIN_AGENTS}, {MAX_AGENTS}]; got {n_present}"
        )
    env = FormationHallwayEnv({
        "num_envs": 1,
        "max_time_steps": max_steps,
        "device": "cpu",
        "render": False,
        "initial_agents": int(n_present),
    })
    dist = [0.0] * MAX_AGENTS
    dist[n_present - 1] = 1.0
    teleop = RandomTeleop(
        env, seed=seed, p_spawn=0.0, p_delete=0.0, init_n_present_dist=dist,
    )

    obs = env.vector_reset()
    teleop.reset_env(0)
    acc = EpisodeAccumulator(env.cfg["n_agents"])

    records: List[dict] = []
    while len(records) < episodes:
        with torch.no_grad():
            x = agent.format_input(obs, device)
            action, _, _, _ = agent.get_action_and_value(x)
        teleop.step()
        obs, _r, done, infos = env.vector_step(action.cpu().numpy())
        acc.update(
            per_agent_rewards=[infos[0]["rewards"][k] for k in range(MAX_AGENTS)],
            active_count=int(infos[0]["active_count"]),
            teleop_mask=obs[0]["teleop_mask"],
            n_present=int(infos[0]["n_present"]),
            formation_err=infos[0]["formation_error"],
            circle_radius=float(infos[0].get("circle_radius", 0.0)),
            fwd_velocity=float(infos[0]["fwd_velocity"]),
            stalled=bool(infos[0]["stalled"]),
            had_collision=bool(infos[0]["collided"]),
            had_wall_hit=bool(infos[0]["wall_hit"]),
        )
        if done[0]:
            rec = acc.emit(
                iteration=0, env_id=0,
                reached_goal=bool(infos[0]["goal_reached"]),
            )
            records.append(rec)
            acc.reset()
            env.reset_at(0)
            teleop.reset_env(0)

    return {
        "n_present": int(n_present),
        "n_episodes": len(records),
        "success_rate": float(np.mean([r["reached_goal"] for r in records])),
        "mean_total_reward": float(np.mean([r["total_reward"] for r in records])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
        "mean_v_y": float(np.mean([r["forward_velocity_mean"] for r in records])),
        "mean_form_err": float(np.mean([r["formation_error_mean"] for r in records])),
        "mean_circle_radius": float(np.mean([r["circle_radius_mean"] for r in records])),
        "mean_collisions": float(np.mean([r["num_collisions"] for r in records])),
    }


def score(per_n: dict) -> float:
    """Worst-regime success first, then mean success, then v_y, then -form_err.

    Hitting 0% on any regime nets `1000 * 0` from the dominant term, so a
    policy that only succeeds at the easiest cluster size cannot win.
    """
    rs = list(per_n.values())
    if not rs:
        return float("-inf")
    succs = [r["success_rate"] for r in rs]
    v_ys = [r["mean_v_y"] for r in rs]
    forms = [r["mean_form_err"] for r in rs]
    return (
        1000.0 * min(succs)
        + 250.0 * (sum(succs) / len(succs))
        + 25.0 * (sum(v_ys) / len(v_ys))
        - 50.0 * max(forms)
    )


def evaluate_and_maybe_save_best(
    *,
    agent,
    device,
    logger: RunLogger,
    ckpt_path: str,
    n_present_list: Iterable[int],
    episodes: int,
    max_steps: int,
    current_best_score: Optional[float],
    seed: int = 0,
):
    """Run eval, compute score, update best.pt + best_eval.json on improvement.

    Returns (score, per_n: dict[int, dict], is_new_best: bool).
    Caller is responsible for tracking `current_best_score` across iterations.
    """
    was_training = agent.training
    agent.eval()
    try:
        per_n = {}
        for n in n_present_list:
            per_n[int(n)] = run_episodes(
                agent=agent, device=device, n_present=int(n),
                episodes=episodes, max_steps=max_steps, seed=seed,
            )
    finally:
        agent.train(was_training)

    s = score(per_n)
    is_new_best = (current_best_score is None) or (s > current_best_score)
    if is_new_best:
        logger.update_named_symlink("best.pt", ckpt_path)
        with open(os.path.join(logger.weights_dir, "best_eval.json"), "w") as f:
            json.dump(
                {
                    "ckpt": os.path.relpath(ckpt_path, logger.weights_dir),
                    "score": s,
                    "per_n": {str(k): v for k, v in per_n.items()},
                },
                f, indent=2, default=str,
            )
    return s, per_n, is_new_best