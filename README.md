# AFOR — Dynamic-Formation PPO with Teleop-in-the-Loop

Final Demo:

<p align="center">
  <a href="https://www.youtube.com/watch?v=QdPGuu86Pl0">
    <img src="https://youtube.com" alt="Watch the video" width="70%" style="border-radius: 8px;">
    <br>
    <sub><b>▶ Watch on YouTube</b></sub>
  </a>
</p>

This repo extends the original [AFOR paper](https://arxiv.org/pdf/2404.01618)
("Coordinated Multi-Robot Navigation with Formation Adaptation") with a
single PPO policy that drives a 4-robot cluster down a long obstacle-free
hallway while **dynamically switching the target formation based on the
number of policy-controlled robots**, and gracefully handling **mid-episode
human teleoperation** of any subset of the cluster.

| Active cluster size | Target shape |
|---|---|
| 4 | square |
| 3 | equilateral triangle |
| 2 | horizontal line |
| 1 | solo (no formation term) |

The original paper's baselines (`env_line`, `env_pentagon`, `env_wedge` —
each pinned to 5 robots and a single formation) remain unchanged in
`code/`; everything new lives in additional files alongside them.

> Detailed design rationale and the 4-person work-split lives in
> [`plan.md`](./plan.md). This README is the operational manual.

---

## Table of contents

- [What changed vs. baseline](#what-changed-vs-baseline)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [The hallway environment](#the-hallway-environment)
- [Reward design](#reward-design)
- [Teleop](#teleop)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Interactive demo](#interactive-demo)
- [Persistent metrics](#persistent-metrics)
- [Comparing runs](#comparing-runs)
- [Tests](#tests)
- [Reproducing the baseline](#reproducing-the-baseline)
- [Limitations / out of scope](#limitations--out-of-scope)
- [License](#license)

---

## What changed vs. baseline

| Concern | Baseline (`env_line.py`, `train.py`) | This work (`env_hallway.py`, `train_hallway.py`) |
|---|---|---|
| Number of robots | hardcoded 5 | up to `MAX_AGENTS = 4`, dynamically active |
| Target formation | hardcoded per env file | switches at runtime by active count |
| Map | obstacle gauntlet | empty 2 × 12 m hallway |
| Teleop | not supported | per-robot mask + override velocity |
| Training disturbance | none | `RandomTeleop` grabs/releases robots stochastically |
| Loss | uniform over robots | masked: only policy-controlled robots contribute gradient |
| Metrics | one `print` per iteration | `config.json` + `iterations.csv` + `episodes.jsonl` per run |
| Checkpoints | overwritten in `weights/real-line2/` | `runs/<ts>_<tag>/weights/` + `latest.pt` symlink |
| Policy network | `n_agents = 5` hardcoded, 6-dim per-robot input | parametric `n_agents`, 8-dim input (adds `teleop_mask` + `present_mask`) — back-compatible with the baseline |

The model file (`code/model.py`) was patched in place but is fully
backward-compatible — running the original `code/train.py` against
`env_line` still works exactly as before.

---

## Repository layout

```
afor/
├── plan.md                       # full design + 4-person split
├── README.md                     # you are here
├── pyproject.toml                # uv project file
├── code/
│   ├── contract.py               # NEW · shared interface constants
│   ├── env_hallway.py            # NEW · dynamic-formation env
│   ├── teleop.py                 # NEW · RandomTeleop + KeyboardTeleop
│   ├── metrics.py                # NEW · RunLogger + EpisodeAccumulator
│   ├── checkpoint.py             # NEW · save/load helpers, resume support
│   ├── train_hallway.py          # NEW · PPO trainer with masked loss + metrics
│   ├── eval_hallway.py           # NEW · headless / rendered eval -> eval.json
│   ├── render_hallway.py         # NEW · pygame renderer with formation overlay
│   ├── run_demo.py               # NEW · interactive demo binary
│   ├── scripts/
│   │   └── compare_runs.py       # NEW · cross-run comparison + plot
│   ├── model.py                  # PATCHED · parametric n_agents + use_masks knob
│   ├── env_line.py               # baseline (unchanged)
│   ├── env_pentagon.py           # baseline (unchanged)
│   ├── env_wedge.py              # baseline (unchanged)
│   ├── train.py                  # baseline trainer (unchanged)
│   ├── eval.py                   # baseline eval (unchanged)
│   └── README.md                 # baseline-specific instructions
├── tests/
│   ├── conftest.py               # adds code/ to sys.path
│   ├── fake_env.py               # contract-shaped stub env
│   └── test_formation.py         # formation + env smoke tests
└── runs/                         # gitignored; created on first training run
    └── <timestamp>_<tag>/
        ├── config.json
        ├── iterations.csv
        ├── episodes.jsonl
        ├── eval.json             # appears after running eval_hallway.py
        └── weights/
            ├── weights_epoch{N}.pt
            └── latest.pt          # symlink -> most recent
```

---

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/) but plain `pip` works
too. From the repo root:

```bash
uv sync                  # creates .venv from pyproject.toml + uv.lock
```

or, if you prefer pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install gymnasium numpy pygame scipy tqdm pytest
pip install torch torch_geometric torch_cluster   # match your CUDA build
```

`torch_cluster` must match your installed `torch` version — see the
[PyG install docs](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for the right wheel. Verified versions in `.venv`: torch 2.11.0, PyG 2.7.0,
torch_cluster 1.6.3, gymnasium 1.3.0, pygame 2.6.1.

Optional: `matplotlib` for the run-comparison plot (degrades to a
print-only summary if missing).

---

## Quickstart

```bash
# 1) sanity-check everything imports + env basics work
.venv/bin/python -m pytest tests/ -q

# 2) short training run (a few iterations, fast, just to verify the loop)
.venv/bin/python code/train_hallway.py --iterations 10 --num-envs 4 --max-steps 200 --tag smoke

# 3) headless evaluation of the smoke checkpoint
RUN=$(ls runs | tail -1)
.venv/bin/python code/eval_hallway.py --weights runs/$RUN/weights/latest.pt --episodes 5

# 4) cross-run comparison table
.venv/bin/python code/scripts/compare_runs.py runs/$RUN --no-plot

# 5) interactive demo (opens a pygame window)
.venv/bin/python code/run_demo.py --weights runs/$RUN/weights/latest.pt --reset-on-done
```

For a real training run, scale up:

```bash
.venv/bin/python code/train_hallway.py \
    --iterations 5000 --num-envs 16 --max-steps 400 \
    --checkpoint-every 50 --tag hallway-v1
```

This typically takes 1–2 hrs on a single CPU at the default settings.
Move to GPU by changing `device="cuda"` in `make_config` in
`code/train_hallway.py` — the GNN and rollouts both transfer.

---

## The hallway environment

`code/env_hallway.py` exports `FormationHallwayEnv`, a `gym.Env` that
mirrors the existing `env_line.PassageEnv` interface (`vector_reset`,
`vector_step(actions) -> (obs, rewards_summed, dones, infos)`,
`reset_at(idx)`) so the PPO scaffold drops in unchanged.

**World.** 2.0 m × 12.0 m, no obstacles. Robots spawn near `y = -5`,
goal line at `y = +5` (along the long axis).

**Action.** `Tuple(Box(low=-MAX_V, high=MAX_V, shape=(2,))) * MAX_AGENTS`
— a 2D desired-velocity per robot.

**Observation (`Dict`):**

| Key | Shape | Meaning |
|---|---|---|
| `pos` | `(MAX_AGENTS, 2)` | robot positions |
| `vel` | `(MAX_AGENTS, 2)` | measured velocities |
| `goal` | `(MAX_AGENTS, 2)` | broadcast goal — same for every agent |
| `teleop_mask` | `(MAX_AGENTS,)` | 1.0 = under teleop control, 0.0 = policy |
| `present_mask` | `(MAX_AGENTS,)` | reserved (always 1.0 in v1) |
| `time` | `(MAX_AGENTS, 1)` | broadcast episode time |

**Active cluster** at any step is `sum(present_mask * (1 - teleop_mask))`.
The target formation slot positions are returned by the pure function
`target_formation_positions(n)` (also exported at module level for tests
and the renderer).

**Step infos** include the per-agent reward dict plus
`active_count`, `goal_reached`, `formation_error`, `fwd_velocity`,
`stalled`, `wall_hit`, `collided` — all consumed by
`EpisodeAccumulator` for the `episodes.jsonl` log.

---

## Reward design

All coefficients live in `code/contract.py::REWARD_COEFFS` and can be
overridden via the env config. Per-robot reward for active (non-teleop'd)
robot `i`:

| Term | Formula | Default coeff |
|---|---|---|
| Forward progress | `+ k_fwd * dy_i` | `5.0` |
| Stall penalty | `- k_stall` if cluster centroid moved < `eps` over last `K` steps | `0.5`, `K = 20`, `eps = 0.02` |
| Formation error | `- k_form * dist_to_assigned_slot` (Hungarian-matched per step) | `2.0` |
| Inter-robot collision | `- k_coll` if any pairwise dist < `2 * AGENT_RADIUS` | `5.0` |
| Wall overshoot | `- k_wall * |overshoot_x|` | `1.0` |
| Goal bonus | `+ k_goal` once cluster centroid passes `GOAL_Y` (one-shot) | `20.0` |

Teleop'd robots get a per-step reward of 0; the trainer additionally
masks their gradient contribution to zero.

---

## Teleop

`code/teleop.py` exports two interchangeable drivers that both speak the
env's `set_teleop(env_idx, robot_idx, active)` and
`set_teleop_action(env_idx, robot_idx, vel)` interface.

**`RandomTeleop`** — runs during training. Per env at each step:
- with prob `p_grab` (default 0.005): pick a random free robot, mark it
  teleop'd, sample a duration in `[40, 160]` steps, choose a left/right
  drift direction.
- while held: drive on a sinusoidal lateral push at `drift_speed`
  (default 0.6), with a small forward base velocity.
- with prob `p_release` per step (default 0.01) or after the duration:
  release the robot back to policy control.

This produces the grab/release pattern a human will create at inference
time, so the policy learns to re-form around a missing teammate and
re-absorb returning ones.

**`KeyboardTeleop`** — used by `run_demo.py`:

| Key | Action |
|---|---|
| `1` / `2` / `3` / `4` | toggle teleop on robot 1..4 |
| `W` / `A` / `S` / `D` | drive the most-recently selected robot |
| `0` | release all teleop'd robots |
| `ESC` | quit the demo |

The selected robot moves at `drive_speed = MAX_V` in the keyed direction.

---

## Model

`code/model.py` was patched (back-compatibly) so `n_agents` is read from
the observation space and a `use_masks` config knob grows the per-robot
input from 6 to 8 features by appending `teleop_mask` and `present_mask`.

```python
cfg["model"]["custom_model_config"]["use_masks"] = True
```

Architecture is otherwise unchanged from baseline AFOR:

- Permutation-invariant `GNNBranch` (encoder → message-passing → post)
  with shared parameters across robots, radius-graph at
  `comm_range = 2.0 m`.
- Beta-distribution policy head (alpha/beta per action dim, squashed to
  `[-MAX_V, MAX_V]`).
- Per-robot value head.

Backward compatibility verified: instantiating the patched `Agent`
against `env_line.PassageEnv` (5 robots, 6-dim input) loads + steps
exactly as before.

---

## Training

```
python code/train_hallway.py [flags]
```

| Flag | Default | Purpose |
|---|---|---|
| `--iterations` | 50 | total PPO iterations |
| `--tag` | `hallway-v0` | suffix on the run directory |
| `--num-envs` | 8 | parallel rollout envs |
| `--max-steps` | 400 | timesteps per rollout |
| `--checkpoint-every` | 20 | save checkpoint every N iters (last iter always saved) |
| `--no-teleop` | off | disable `RandomTeleop` (debug only) |
| `--seed` | 0 | seeds random / numpy / torch |
| `--resume` | none | path to a `.pt` checkpoint to resume from (continues iteration numbering, restores Adam moments) |

PPO hyperparameters (gamma 0.995, lambda 0.95, clip 0.2, lr 5e-5,
4 SGD epochs, value-clip 1.0, max grad norm 0.5, entropy coeff 0.001)
live in `make_config()` at the top of `train_hallway.py`.

Per-iteration print:
```
iter   10  rew -0.2087  pg +0.0366  v 60.5662  ent -0.204  kl +0.0026  ep_len 200.0
```
The persisted CSV/JSONL/JSON files (see [Persistent metrics](#persistent-metrics))
are the source of truth for analysis — the print is just a convenience.

**Loss masking.** The PPO objective multiplies per-robot pg, value, and
entropy losses by `(1 - teleop_mask)` and renormalises by the sum of the
mask, so gradients flow only through policy-controlled robots.

### Resuming a run

```
python code/train_hallway.py --resume runs/<ts>_<tag>/weights/latest.pt --iterations 200 --tag hallway-v1-cont
```

`--iterations N` is **additional** iterations to run, not a target total.
A resumed run:

- Restores the agent state dict **and** the Adam optimizer state, so the
  running moments do not reset (small but real impact on early stability).
- Continues iteration numbering from the checkpoint. If you stopped at
  `iter=500` and resume for `--iterations 200`, the new
  `iterations.csv` is numbered `501..700` — concatenate the two CSVs to
  get the full curve.
- Lands in its own `runs/<new-ts>_<new-tag>/` directory; `config.json`
  records the lineage in a `resume` block:
  ```json
  "resume": {
    "from": "/abs/path/to/runs/.../weights/latest.pt",
    "from_iteration": 500,
    "had_optimizer_state": true
  }
  ```
  `had_optimizer_state: false` means you resumed from a legacy
  bare-state_dict checkpoint and Adam started fresh.

Checkpoints are written as a dict `{agent, optimizer, iteration}` (see
`code/checkpoint.py`). `eval_hallway.py` and `run_demo.py` route
through the same `load_checkpoint` helper, which also accepts older
bare-state_dict files for backward compatibility.

---

## Evaluation

```
python code/eval_hallway.py --weights runs/<ts>/weights/latest.pt [flags]
```

| Flag | Default | Purpose |
|---|---|---|
| `--weights` | required | path to a `.pt` checkpoint |
| `--episodes` | 20 | how many episodes to roll out |
| `--max-steps` | 600 | cap per episode |
| `--num-envs` | 1 | parallel envs (1 is enough headless) |
| `--render` | off | open a pygame window and step at real-time |
| `--no-render` | on | force headless |
| `--teleop` | off | apply `RandomTeleop` during eval (robustness test) |
| `--out` | `<run>/eval.json` | output path |

Writes `<run>/eval.json` with a top-level summary plus a `records[]`
list, one entry per episode:

```json
{
  "weights": "...",
  "wall_time_s": 0.23,
  "episodes": 5,
  "success_rate": 0.0,
  "mean_total_reward": -136.75,
  "mean_episode_length": 80.0,
  "mean_forward_velocity": -0.043,
  "mean_formation_error": 0.157,
  "records": [ { ... per-episode record ... }, ... ]
}
```

---

## Interactive demo

```
python code/run_demo.py --weights runs/<ts>/weights/latest.pt [--reset-on-done]
```

Opens a pygame window showing the hallway, 4 colored robots, the active
target-formation outline (Hungarian-matched faded rings), and a HUD with
the active count, episode step, and accumulated reward. See
[Teleop](#teleop) for the keybindings.

`--reset-on-done` keeps the demo running across episodes so you can stay
in the window between rollouts.

---

## Persistent metrics

Every training run writes a self-contained directory under `runs/`
(gitignored). Format is intentionally diff-friendly so runs can be
compared with `pandas`, `jq`, `awk`, or eyeballs.

```
runs/20260503_161721_e2e/
├── config.json        # one-shot snapshot
├── iterations.csv     # one row per PPO iteration  <- primary comparison artefact
├── episodes.jsonl     # one JSON object per finished episode
└── weights/
    ├── weights_epoch1.pt
    ├── ...
    └── latest.pt      # symlink
```

**`config.json`** captures everything needed to reproduce the run:
PPO hyperparameters, reward coefficients, env constants, teleop
parameters, full argparse `--`flags, plus a `meta` block with timestamp,
hostname, platform, git SHA, torch + CUDA versions.

**`iterations.csv`** columns:
`iter, wall_time_s, env_steps, policy_loss, value_loss, entropy,
total_loss, approx_kl, clip_frac, grad_norm, mean_reward,
mean_episode_length, lr`.

**`episodes.jsonl`** records (one per `done` event):
`iter, env_id, episode_length, total_reward, reached_goal,
num_collisions, num_wall_hits, num_teleop_grabs, max_active_count,
min_active_count, formation_error_mean,
formation_error_per_active_count, forward_velocity_mean, stall_steps`.

Both files are append-only, so a crashed run still yields a partial
record. Loading examples:

```python
import pandas as pd
iters = pd.read_csv("runs/20260503_161721_e2e/iterations.csv")
iters.plot(x="iter", y=["mean_reward", "policy_loss"])

import json
episodes = [json.loads(l) for l in open("runs/.../episodes.jsonl")]
sum(e["reached_goal"] for e in episodes) / len(episodes)  # success rate
```

---

## Comparing runs

```
python code/scripts/compare_runs.py runs/<a> runs/<b> [runs/<c> ...] [--no-plot]
```

Prints a sortable table:

```
run                  iterations  episodes  last_mean_reward  last_policy_loss  ...
20260503_161721_e2e  10          40        -0.2087           0.0366            ...
```

If `matplotlib` is installed, also writes
`runs/_comparison/<timestamp>.png` with `mean_reward` and
`policy_loss` curves over iteration, one line per run.

---

## Tests

```
.venv/bin/python -m pytest tests/ -q
```

`tests/test_formation.py` covers:
- shape correctness for `target_formation_positions(n)` for `n ∈ {1,2,3,4}`
- centroid at origin, equilateral triangle, square sides, line horizontal
- env step shapes
- teleop override + release round-trip

`tests/fake_env.py` is a ~80-line stub satisfying the env contract,
used both inside the formal tests and by `code/teleop.py --demo` so
the teleop module can be developed without the real env.

---

## Reproducing the baseline

The original 5-robot AFOR experiments still work. From the repo root:

```bash
.venv/bin/python code/train.py             # baseline trainer
.venv/bin/python code/eval.py              # baseline eval (single env)
```

See `code/README.md` for the full original instructions, including how
to swap formation imports between `env_line` / `env_pentagon` / `env_wedge`.

---

## Limitations / out of scope

Deliberately not in v1 — see `plan.md` Section 8:

- No tensorboard / wandb (CSV/JSONL is intentional).
- No curriculum over the teleop probability — fixed `p_grab`.
- No formation rotation to arbitrary headings — long axis aligned to `+y`.
- No multi-policy ensemble per cluster size — one shared policy.
- No obstacle generalisation — the hallway is empty.
- The model is CPU by default. GPU works (change `device` in
  `make_config`) but isn't part of the verified path.
- I have not trained a converged policy — only verified the pipeline
  learns over a 10-iteration smoke run (mean reward `-2.77 → -0.21`,
  KL ~0.003, no NaNs). Real performance needs hours of training.

---

## License

This repo is released under the [Creative Commons Attribution-ShareAlike
4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
The website scaffold was originally adapted from
[Nerfies](https://github.com/nerfies/nerfies.github.io).
