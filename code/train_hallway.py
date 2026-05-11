"""PPO trainer for FormationHallwayEnv with random-teleop disturbance.

Trains a single shared GNN policy that controls a dynamic-size cluster
(1..MAX_AGENTS robots) in an 8x8 m square arena. The cluster always
targets a circle whose radius scales with the active count.

Loss is masked by `present_mask * (1 - teleop_mask)` so gradients flow
only through robots that are (a) actually in the world and (b) under
policy control.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint import load_checkpoint, save_checkpoint
from contract import INITIAL_AGENTS, MAX_AGENTS, REWARD_COEFFS
from env_hallway import FormationHallwayEnv
from metrics import EpisodeAccumulator, RunLogger
from model import Agent
from periodic_eval import evaluate_and_maybe_save_best
from teleop import RandomTeleop


def make_config(num_envs: int, max_time_steps: int) -> dict:
    return {
        "seed": 0,
        "clip_param": 0.2,
        "entropy_coeff": 0.01,
        "vf_clip_param": 1.0,
        "vf_loss_coeff": 0.5,
        "max_grad_norm": 1.0,
        "norm_adv": True,
        "clip_vloss": True,
        "num_sgd_iter": 8,
        "lr": 1e-4,
        "gamma": 0.995,
        "lambda": 0.95,
        "model": {
            "custom_model_config": {
                "activation": "relu",
                "msg_features": 32,
                "comm_range": 4.0,
                "use_masks": True,
            },
        },
        "env_config": {
            "num_envs": num_envs,
            "device": "cpu",
            "max_time_steps": max_time_steps,
            "render": False,
        },
        "teleop": {
            "p_grab": 0.005,
            "p_release": 0.01,
            "p_spawn": 0.002,
            "p_delete": 0.002,
            "drift_speed": 0.6,
            "init_n_present_dist": None,  # None -> teleop.DEFAULT_INIT_N_PRESENT_DIST
        },
    }


def _parse_dist(s: str | None):
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != MAX_AGENTS:
        raise ValueError(
            f"--init-n-present-dist must be {MAX_AGENTS} comma-separated weights; got {len(parts)}"
        )
    return [float(p) for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--tag", type=str, default="circle-v1")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument(
        "--no-teleop",
        action="store_true",
        help="disable random-teleop disturbance (debug only)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to a .pt checkpoint to resume from "
        "(continues iteration numbering, restores optimizer state)",
    )
    ap.add_argument("--lr", type=float, default=None, help="override LR")
    ap.add_argument("--entropy-coeff", type=float, default=None, help="override entropy coeff")
    ap.add_argument("--vf-loss-coeff", type=float, default=None, help="override value-fn loss weight")
    ap.add_argument("--max-grad-norm", type=float, default=None, help="override grad clip")
    ap.add_argument("--clip-param", type=float, default=None, help="override PPO clip")
    ap.add_argument("--num-sgd-iter", type=int, default=None, help="override PPO epochs / iter")
    ap.add_argument("--gamma", type=float, default=None, help="override discount")
    ap.add_argument("--gae-lambda", type=float, default=None, help="override GAE lambda")
    ap.add_argument("--comm-range", type=float, default=None,
                    help="override GNN comm radius (m)")
    ap.add_argument("--p-grab", type=float, default=None)
    ap.add_argument("--p-release", type=float, default=None)
    ap.add_argument("--p-spawn", type=float, default=None)
    ap.add_argument("--p-delete", type=float, default=None)
    ap.add_argument(
        "--curriculum-teleop",
        action="store_true",
        help="linearly ramp teleop disturbance probs from --curriculum-start-scale to 1.0",
    )
    ap.add_argument(
        "--curriculum-start-scale",
        type=float,
        default=0.25,
        help="starting multiplier for p_grab/p_release/p_spawn/p_delete when curriculum is enabled",
    )
    ap.add_argument(
        "--init-n-present-dist",
        type=str,
        default=None,
        help=f"comma-separated {MAX_AGENTS} weights for sampling initial n_present at reset",
    )
    ap.add_argument(
        "--initial-agents",
        type=int,
        default=4,
        help="default n_present after env.reset_at; RandomTeleop may resample if active",
    )
    # Periodic eval — runs at every checkpoint, scores per regime, updates best.pt
    ap.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="run periodic eval every N iters (defaults to --checkpoint-every; "
             "set 0 to disable)",
    )
    ap.add_argument("--eval-episodes", type=int, default=4,
                    help="episodes per regime per eval pass")
    ap.add_argument(
        "--anneal-lr",
        action="store_true",
        help="linearly anneal learning rate from initial value to 0 over training iterations",
    )
    ap.add_argument("--eval-max-steps", type=int, default=None,
                    help="max env steps per eval episode (defaults to --max-steps)")
    ap.add_argument(
        "--eval-n-present-counts",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="comma-separated list of n_present values to evaluate at "
             "(each gets --eval-episodes episodes)",
    )
    args = ap.parse_args()

    eval_every = args.eval_every if args.eval_every is not None else args.checkpoint_every
    eval_max_steps = args.eval_max_steps or args.max_steps
    eval_n_present = [int(x) for x in args.eval_n_present_counts.split(",") if x.strip()]
    if eval_every and not eval_n_present:
        raise SystemExit("--eval-n-present-counts must be non-empty if eval is enabled")

    config = make_config(num_envs=args.num_envs, max_time_steps=args.max_steps)
    config["seed"] = args.seed
    if args.lr is not None:
        config["lr"] = args.lr
    if args.entropy_coeff is not None:
        config["entropy_coeff"] = args.entropy_coeff
    if args.vf_loss_coeff is not None:
        config["vf_loss_coeff"] = args.vf_loss_coeff
    if args.max_grad_norm is not None:
        config["max_grad_norm"] = args.max_grad_norm
    if args.clip_param is not None:
        config["clip_param"] = args.clip_param
    if args.num_sgd_iter is not None:
        config["num_sgd_iter"] = args.num_sgd_iter
    if args.gamma is not None:
        config["gamma"] = args.gamma
    if args.gae_lambda is not None:
        config["lambda"] = args.gae_lambda
    if args.comm_range is not None:
        config["model"]["custom_model_config"]["comm_range"] = args.comm_range
    for k, v in (
        ("p_grab", args.p_grab),
        ("p_release", args.p_release),
        ("p_spawn", args.p_spawn),
        ("p_delete", args.p_delete),
    ):
        if v is not None:
            config["teleop"][k] = v
    init_dist = _parse_dist(args.init_n_present_dist)
    if init_dist is not None:
        config["teleop"]["init_n_present_dist"] = init_dist
    config["env_config"]["initial_agents"] = args.initial_agents

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    device = torch.device("cpu")

    env = FormationHallwayEnv(config["env_config"])
    agent = Agent(env, config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config["lr"], eps=1e-5)

    start_iteration = 0
    resume_meta = None
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, device)
        agent.load_state_dict(ckpt["agent"])
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = int(ckpt["iteration"])
        resume_meta = {
            "from": os.path.abspath(args.resume),
            "from_iteration": start_iteration,
            "had_optimizer_state": ckpt["optimizer"] is not None,
        }
        print(
            f"[resume] from {resume_meta['from']} at iter={start_iteration}"
            + (
                ""
                if resume_meta["had_optimizer_state"]
                else "  (NO optimizer state — Adam moments restart)"
            )
        )

    teleop = (
        None
        if args.no_teleop
        else RandomTeleop(
            env,
            p_grab=config["teleop"]["p_grab"],
            p_release=config["teleop"]["p_release"],
            p_spawn=config["teleop"]["p_spawn"],
            p_delete=config["teleop"]["p_delete"],
            drift_speed=config["teleop"]["drift_speed"],
            init_n_present_dist=config["teleop"]["init_n_present_dist"],
            seed=args.seed,
        )
    )

    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
    runs_dir = os.path.abspath(runs_dir)
    os.makedirs(runs_dir, exist_ok=True)
    logger = RunLogger(runs_dir, tag=args.tag)
    logger.write_config(
        {
            "ppo": {
                k: v
                for k, v in config.items()
                if k not in ("env_config", "model", "teleop")
            },
            "env": config["env_config"],
            "model": config["model"],
            "teleop": config["teleop"] if teleop is not None else None,
            "reward_coeffs": REWARD_COEFFS,
            "args": vars(args),
            "resume": resume_meta,
            "start_iteration": start_iteration,
        }
    )
    print(f"[run] {logger.run_id} -> {logger.dir}")

    # --- buffer allocation ---------------------------------------------
    nE = env.cfg["num_envs"]
    nA = env.cfg["n_agents"]   # always MAX_AGENTS
    T = env.cfg["max_time_steps"]
    actions_buf = torch.zeros((T, nE, nA, 2), device=device)
    logprobs_buf = torch.zeros((T, nE, nA), device=device)
    rewards_buf = torch.zeros((T, nE, nA), device=device)
    dones_buf = torch.zeros((T, nE), device=device)
    values_buf = torch.zeros((T, nE, nA), device=device)
    teleop_buf = torch.zeros((T, nE, nA), device=device)
    present_buf = torch.zeros((T, nE, nA), device=device)

    next_obs = env.vector_reset()
    if teleop is not None:
        for e in range(nE):
            teleop.reset_env(e)
    next_done = torch.zeros(nE, device=device)
    accs = [EpisodeAccumulator(nA) for _ in range(nE)]

    global_step = 0
    start_time = time.time()
    obs_per_step: list = []

    iter_lo = start_iteration + 1
    iter_hi = start_iteration + args.iterations + 1
    best_eval_score: float | None = None
    best_eval_iter: int | None = None
    best_eval_per_n: dict | None = None
    for iteration in range(iter_lo, iter_hi):
        obs_per_step = []
        ep_rewards: list = []
        ep_lengths: list = []
        ep_reached_goal: list = []
        ep_n_present: list = []        # mean n_present per finished episode
        ep_form_err: list = []         # mean formation_err per finished episode
        ep_circle_radius: list = []    # mean circle_radius per finished episode

        # Optional LR annealing: improves late-stage stability at almost no cost.
        if args.anneal_lr:
            frac = 1.0 - ((iteration - iter_lo) / max((args.iterations - 1), 1))
            optimizer.param_groups[0]["lr"] = frac * config["lr"]

        for step in range(T):
            global_step += nE
            x = agent.format_input(next_obs, device)
            obs_per_step.append(x)
            dones_buf[step] = next_done
            teleop_buf[step] = torch.as_tensor(
                [o["teleop_mask"] for o in next_obs], dtype=torch.float32, device=device
            )
            present_buf[step] = torch.as_tensor(
                [o["present_mask"] for o in next_obs], dtype=torch.float32, device=device
            )

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(x)
                values_buf[step] = value
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            if teleop is not None:
                if args.curriculum_teleop:
                    frac = (iteration - iter_lo) / max((args.iterations - 1), 1)
                    scale = args.curriculum_start_scale + (1.0 - args.curriculum_start_scale) * frac
                    teleop.p_grab = config["teleop"]["p_grab"] * scale
                    teleop.p_release = config["teleop"]["p_release"] * scale
                    teleop.p_spawn = config["teleop"]["p_spawn"] * scale
                    teleop.p_delete = config["teleop"]["p_delete"] * scale
                teleop.step()

            next_obs, _r_summed, done, infos = env.vector_step(action.cpu().numpy())

            per_agent = torch.zeros(nE, nA, device=device)
            for e in range(nE):
                for k, rv in infos[e]["rewards"].items():
                    per_agent[e, k] = float(rv)
                accs[e].update(
                    per_agent_rewards=per_agent[e].tolist(),
                    active_count=int(infos[e]["active_count"]),
                    teleop_mask=next_obs[e]["teleop_mask"],
                    n_present=int(infos[e]["n_present"]),
                    formation_err=infos[e]["formation_error"],
                    circle_radius=float(infos[e].get("circle_radius", 0.0)),
                    fwd_velocity=float(infos[e]["fwd_velocity"]),
                    stalled=bool(infos[e]["stalled"]),
                    had_collision=bool(infos[e]["collided"]),
                    had_wall_hit=bool(infos[e]["wall_hit"]),
                )
            rewards_buf[step] = per_agent
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

            for e, d in enumerate(done):
                if d:
                    ep_rewards.append(accs[e].total_reward)
                    ep_lengths.append(accs[e].length)
                    ep_reached_goal.append(bool(infos[e]["goal_reached"]))
                    rec = accs[e].emit(
                        iteration=iteration,
                        env_id=e,
                        reached_goal=bool(infos[e]["goal_reached"]),
                    )
                    ep_n_present.append(rec["mean_n_present"])
                    ep_form_err.append(rec["formation_error_mean"])
                    if rec["circle_radius_mean"] > 0:
                        ep_circle_radius.append(rec["circle_radius_mean"])
                    logger.log_episode(rec)
                    accs[e].reset()
                    env.reset_at(e)
                    if teleop is not None:
                        teleop.reset_env(e)

        # --- advantages ------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(agent.format_input(next_obs, device)).to(
                device
            )
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                nextnonterminal = nextnonterminal.unsqueeze(-1)
                delta = (
                    rewards_buf[t]
                    + config["gamma"] * nextvalues * nextnonterminal
                    - values_buf[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + config["gamma"] * config["lambda"] * nextnonterminal * lastgaelam
                )
            returns_buf = advantages + values_buf

        # --- PPO update over the time-axis ----------------------------
        b_inds = np.arange(T)
        last_pg = last_vl = last_ent = last_kl = last_clip = last_grad = 0.0
        for epoch in tqdm(range(config["num_sgd_iter"]), leave=False):
            np.random.shuffle(b_inds)
            for mb_t in b_inds:
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_per_step[mb_t], actions_buf[mb_t]
                )
                logratio = newlogprob - logprobs_buf[mb_t]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clip_frac = (
                        ((ratio - 1.0).abs() > config["clip_param"])
                        .float()
                        .mean()
                        .item()
                    )

                mb_adv = advantages[mb_t]
                if config["norm_adv"]:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Active mask: only present, policy-controlled robots contribute
                active = present_buf[mb_t] * (1.0 - teleop_buf[mb_t])
                norm = active.sum().clamp(min=1.0)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - config["clip_param"], 1 + config["clip_param"]
                )
                pg_loss = (torch.max(pg_loss1, pg_loss2) * active).sum() / norm

                if config["clip_vloss"]:
                    v_unclipped = (newvalue - returns_buf[mb_t]) ** 2
                    v_clipped = values_buf[mb_t] + torch.clamp(
                        newvalue - values_buf[mb_t],
                        -config["vf_clip_param"],
                        config["vf_clip_param"],
                    )
                    v_clipped_loss = (v_clipped - returns_buf[mb_t]) ** 2
                    v_max = torch.max(v_unclipped, v_clipped_loss)
                    v_loss = 0.5 * (v_max * active).sum() / norm
                else:
                    v_loss = (
                        0.5
                        * (((newvalue - returns_buf[mb_t]) ** 2) * active).sum()
                        / norm
                    )

                ent_loss = (entropy * active).sum() / norm
                loss = (
                    pg_loss
                    - config["entropy_coeff"] * ent_loss
                    + v_loss * config["vf_loss_coeff"]
                )

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), config["max_grad_norm"]
                )
                optimizer.step()

                last_pg = pg_loss.item()
                last_vl = v_loss.item()
                last_ent = ent_loss.item()
                last_kl = approx_kl
                last_clip = clip_frac
                last_grad = float(
                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                )

        # Mean per-env-step reward (sum across robots) — divide by THIS iter's
        # rollout size, not the cumulative env-step counter. Bug pre-fix the
        # divisor was global_step which grows monotonically, making the metric
        # silently shrink across iters even when the policy held steady.
        mean_reward_per_step = float(rewards_buf.sum().item()) / max(T * nE, 1)
        mean_ep_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
        success_rate = (
            float(np.mean(ep_reached_goal)) if ep_reached_goal else float("nan")
        )

        def _mean_or_nan(xs):
            return float(np.mean(xs)) if xs else float("nan")

        mean_episode_reward = _mean_or_nan(ep_rewards)
        mean_n_present = _mean_or_nan(ep_n_present)
        mean_form_err = _mean_or_nan(ep_form_err)
        mean_circle_radius = _mean_or_nan(ep_circle_radius)

        wall = time.time() - start_time
        logger.log_iter(
            iter=iteration,
            wall_time_s=round(wall, 2),
            env_steps=global_step,
            policy_loss=last_pg,
            value_loss=last_vl,
            entropy=last_ent,
            total_loss=last_pg
            + config["vf_loss_coeff"] * last_vl
            - config["entropy_coeff"] * last_ent,
            approx_kl=last_kl,
            clip_frac=last_clip,
            grad_norm=last_grad,
            mean_reward=mean_reward_per_step,
            mean_episode_length=mean_ep_len,
            lr=config["lr"],
            success_rate=success_rate,
            mean_n_present=mean_n_present,
            mean_formation_error=mean_form_err,
            mean_circle_radius=mean_circle_radius,
            n_episodes=len(ep_rewards),
        )
        print(
            f"iter {iteration:4d}  rew/step {mean_reward_per_step:+.4f}  "
            f"rew/ep {mean_episode_reward:+8.2f}  "
            f"succ {success_rate*100:5.1f}%  "
            f"pg {last_pg:+.4f}  v {last_vl:.4f}  ent {last_ent:+.3f}  "
            f"kl {last_kl:+.4f}  ep_len {mean_ep_len:.1f}  "
            f"n_pres {mean_n_present:.2f}  form {mean_form_err:.3f}  "
            f"r_circ {mean_circle_radius:.2f}"
        )

        is_checkpoint = (
            iteration % args.checkpoint_every == 0 or iteration == iter_hi - 1
        )
        if is_checkpoint:
            ckpt = logger.checkpoint_path(iteration)
            save_checkpoint(ckpt, agent, optimizer, iteration)
            logger.update_latest_symlink(ckpt)

            if eval_every and (
                iteration % eval_every == 0 or iteration == iter_hi - 1
            ):
                s, per_n, is_new_best = evaluate_and_maybe_save_best(
                    agent=agent,
                    device=device,
                    logger=logger,
                    ckpt_path=ckpt,
                    n_present_list=eval_n_present,
                    episodes=args.eval_episodes,
                    max_steps=eval_max_steps,
                    current_best_score=best_eval_score,
                    seed=args.seed + iteration,  # vary seed across passes
                )
                if is_new_best:
                    best_eval_score = s
                    best_eval_iter = iteration
                    best_eval_per_n = per_n
                regime_str = "  ".join(
                    f"n={n}:{r['success_rate']*100:.0f}%/{r['mean_v_y']:+.2f}"
                    for n, r in per_n.items()
                )
                print(
                    f"  [eval iter {iteration:4d}] score {s:+8.2f}  "
                    f"{regime_str}  "
                    f"{'(NEW BEST)' if is_new_best else f'(best={best_eval_score:+.2f}@{best_eval_iter})'}"
                )

    logger.close()
    print(f"[done] last ckpt: {logger.checkpoint_path(iter_hi - 1)}")
    if best_eval_iter is not None:
        print(
            f"[done] best ckpt: weights/best.pt -> weights_epoch{best_eval_iter}.pt  "
            f"score={best_eval_score:+.2f}"
        )
        for n, r in (best_eval_per_n or {}).items():
            print(
                f"         n_present={n:2d}: succ={r['success_rate']*100:5.1f}%  "
                f"v_y={r['mean_v_y']:+.3f}  form={r['mean_form_err']:.3f}  "
                f"R={r['mean_total_reward']:+.2f}"
            )


if __name__ == "__main__":
    main()