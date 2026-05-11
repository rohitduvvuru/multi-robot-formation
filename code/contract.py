"""Interface contract for the always-circle hallway env.

Everything that touches FormationHallwayEnv (env code, teleop, trainer,
renderer, eval, demo) imports from here. Changing a constant ripples
through everyone's code, so do it deliberately.

Observation dict (per env):
    pos          : list[[x, y]] of length MAX_AGENTS
    vel          : list[[vx, vy]] of length MAX_AGENTS
    goal         : list[[gx, gy]] of length MAX_AGENTS  (broadcast for compat)
    teleop_mask  : list[float] of length MAX_AGENTS in {0., 1.}
    present_mask : list[float] of length MAX_AGENTS in {0., 1.}
    time         : list[[t]] of length MAX_AGENTS  (broadcast for compat)

Action: numpy array shape (num_envs, MAX_AGENTS, 2) of desired velocities.
The env overrides slots where teleop_mask == 1 with the stored teleop velocities,
and ignores actions for non-present slots (present_mask == 0).

Active cluster size = sum(present_mask * (1 - teleop_mask)) per env.
Target formation: a circle of radius r(n_active) centred on the active centroid.
"""

# --- agent population --------------------------------------------------------
MAX_AGENTS = 10
MIN_AGENTS = 1
INITIAL_AGENTS = 6

# --- world geometry (8 m x 8 m square) ---------------------------------------
DT = 0.05
WORLD_W = 8.0
WORLD_H = 8.0

SPAWN_Y = -3.5
GOAL_Y = 3.5

# --- kinematics --------------------------------------------------------------
MAX_V = 1.0
MAX_A = 2.0
MIN_A = -2.0

AGENT_RADIUS = 0.2

# --- formation ---------------------------------------------------------------
# Inter-neighbour chord length on the target circle. Must exceed
# 2 * AGENT_RADIUS so adjacent robots in formation don't permanently overlap.
# 0.6 leaves a 0.2 m centre-to-centre margin (50% of robot diameter).
CIRCLE_SIDE = 0.6

# --- run defaults ------------------------------------------------------------
DEFAULT_MAX_TIME_STEPS = 600
DEFAULT_RENDER_PX_PER_M = 60

# --- sentinel for non-present robots -----------------------------------------
# Park "deleted" robots well outside WORLD so radius_graph drops them and the
# renderer can skip them.
SENTINEL_X = 3 * WORLD_W
SENTINEL_Y = 3 * WORLD_H

REWARD_COEFFS = {
    "k_fwd": 5.0,
    "k_stall": 0.5,
    "k_form": 2.0,
    "k_coll": 5.0,
    "k_wall": 1.0,
    "k_goal": 20.0,
    "stall_window": 20,
    "stall_eps": 0.02,
}