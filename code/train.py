#%%

import os
import time
import numpy as np
import random
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from env_line import PassageEnv

from model import Agent


#%%

num_agents = 5
pentagon_coords = np.array([
    [np.cos(2 * np.pi * i / num_agents + np.pi/2 + np.pi/(num_agents+1)), \
     np.sin(2 * np.pi * i / num_agents + np.pi/2 + np.pi/(num_agents+1))]
    for i in range(num_agents)
])

scale_factor = 0.5
scaled_pentagon_coords = pentagon_coords * scale_factor

agent_formation = (np.array([[-0.5, 0],[0, 0], [1, 0],[1.5, 0], [2, 0]]) * 0.5).tolist()


config={
    "seed": 0,
    "framework": "torch",
    "env": "passage_env",
    "clip_param": 0.2,
    "entropy_coeff": 0.001,
    "train_batch_size": 65536,
    "sgd_minibatch_size": 4096,
    "vf_clip_param": 1.0,
    "vf_loss_coeff": 1.0,
    "max_grad_norm": 0.5,
    "norm_adv": True,
    "clip_vloss": True,
    "num_sgd_iter": 10,
    "num_gpus": 1,
    "num_envs_per_worker": 10,
    "lr": 5e-5,
    "gamma": 0.995,
    "lambda": 0.95,
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    "model": {
        "custom_model": "model",
        "custom_action_dist": "hom_multi_action",
        "custom_model_config": {
            "activation": "relu",
            "msg_features": 32,
            "comm_range": 2.0,
        },
    },
    "env_config": {
        "world_dim": (4.0, 5.0),
        "dt": 0.05,
        "num_envs": 32,
        "device": "cuda",
        "n_agents": num_agents,
        "agent_formation": agent_formation,
        "placement_keepout_border": 1.0,
        "placement_keepout_wall": 1.5,
        "pos_noise_std": 0.0,
        "max_time_steps": 750,
        "communication_range": 20.0,
        "wall_width": 5.0,
        "gap_length": 2.3,
        "grid_px_per_m": 40,
        "agent_radius": 0.13,
        "render": False,
        "render_px_per_m": 160,
        "max_v": 1.0,
        "max_a": 1.0,
        "min_a": -1.0,
    },
    "render_env": False,
    "evaluation_interval": 1,
    "evaluation_num_episodes": 1,
    "evaluation_num_workers": 1,
    "evaluation_parallel_to_training": True,
    "evaluation_config": {
        "record_env": "videos",
        "render_env": True,
    },
    "auto_teleop": {
        "enabled": True,
        "p_grab_start": 0.002,
        "p_grab_end": 0.02,
        "p_release": 0.01,
        "drift_speed": 0.6,
        "min_steps": 12,
        "max_steps": 40,
    },
    "reward_framing": {
        "maintenance_scale": 1.0,
        "adapt_scale": 1.15,
        "adapt_window": 20,
    },
    "comm_radius": {
        "start": 3.0,
        "end": 2.0,
        "anneal_frac": 0.4,
    },
}


def linear_schedule(start, end, frac):
    frac = min(max(frac, 0.0), 1.0)
    return start + (end - start) * frac

def apply_auto_teleop(raw_actions, obs, teleop_state, cfg, progress):
    if not cfg["enabled"]:
        return raw_actions, np.zeros((raw_actions.shape[0],), dtype=bool), np.zeros((raw_actions.shape[0],), dtype=bool)

    p_grab = linear_schedule(cfg["p_grab_start"], cfg["p_grab_end"], progress)
    p_release = cfg["p_release"]
    drift_speed = cfg["drift_speed"]

    actions = raw_actions.copy()
    active_mask = np.zeros((raw_actions.shape[0],), dtype=bool)
    toggled_mask = np.zeros((raw_actions.shape[0],), dtype=bool)
    for env_idx in range(raw_actions.shape[0]):
        state = teleop_state[env_idx]
        if state["active"] and np.random.rand() < p_release:
            state["active"] = False
            state["idx"] = -1
            state["remaining"] = 0
            toggled_mask[env_idx] = True
        elif (not state["active"]) and np.random.rand() < p_grab:
            state["active"] = True
            state["idx"] = np.random.randint(raw_actions.shape[1])
            state["remaining"] = np.random.randint(cfg["min_steps"], cfg["max_steps"] + 1)
            state["direction"] = 1.0 if np.random.rand() < 0.5 else -1.0
            toggled_mask[env_idx] = True

        if state["active"]:
            active_mask[env_idx] = True
            idx = state["idx"]
            centroid = np.mean(np.asarray(obs[env_idx]["pos"]), axis=0)
            pos = np.asarray(obs[env_idx]["pos"])[idx]
            lateral = np.array([1.0, 0.0])
            to_center = centroid - pos
            if np.linalg.norm(to_center) > 1e-6:
                sign = np.sign(np.dot(np.array([to_center[0], 0.0]), lateral))
                if sign != 0:
                    state["direction"] = -sign
            actions[env_idx, idx, :] = np.array([state["direction"] * drift_speed, 0.3 * drift_speed])
            state["remaining"] -= 1
            if state["remaining"] <= 0:
                state["active"] = False
                state["idx"] = -1
                toggled_mask[env_idx] = True

    return actions, active_mask, toggled_mask

#%%
random.seed(config['seed'])
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.backends.cudnn.deterministic = True

#%%
os.environ["SDL_VIDEODRIVER"]='dummy'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

env_config = config['env_config']
env_config['device'] = device
env = PassageEnv(env_config)

agent = Agent(env, config).to(device)
optimizer = optim.Adam(agent.parameters(), lr=config['lr'], eps=1e-5)


#%%
env.vector_reset()
returns = torch.zeros((env.cfg["num_envs"], env.cfg["n_agents"]))
selected_agent = 0
rew = 0


#%%

obs = list()
actions = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"]) + env.observation_space["pos"].shape).to(device)
logprobs = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"],env.cfg["n_agents"])).to(device)
rewards = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"],env.cfg["n_agents"])).to(device)
dones = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"])).to(device)
values = torch.zeros((env.cfg["max_time_steps"], env.cfg["num_envs"],env.cfg["n_agents"])).to(device)

global_step = 0
start_time = time.time()
next_obs = env.vector_reset()
next_done = torch.zeros(env.cfg['num_envs']).to(device)
teleop_state = [dict(active=False, idx=-1, remaining=0, direction=1.0) for _ in range(env.cfg['num_envs'])]
adapt_timers = np.zeros((env.cfg['num_envs'],), dtype=np.int32)

num_iterations = config['train_batch_size']
batch_size = env.cfg["max_time_steps"]
minibatch_size = config['sgd_minibatch_size']
assert batch_size >= minibatch_size, "sgd_minibatch_size should be <= max_time_steps"

weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'real-line2')
os.makedirs(weights_dir, exist_ok=True)

for iteration in range(1, num_iterations + 1):
    obs = list()
    progress = iteration / max(1, num_iterations)
    anneal_frac = config["comm_radius"]["anneal_frac"]
    comm_progress = min(progress / anneal_frac, 1.0) if anneal_frac > 0 else 1.0
    current_comm_radius = linear_schedule(config["comm_radius"]["start"], config["comm_radius"]["end"], comm_progress)
    agent.set_comm_range(current_comm_radius)
    frames = []

    for step in range(0, env.cfg["max_time_steps"]):
        global_step += env.cfg['num_envs']
        obs.append(agent.format_input(next_obs, device))
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(agent.format_input(next_obs, device))
            values[step] = value

        policy_actions_np = action.cpu().numpy()
        stepped_actions, teleop_active, teleop_toggled = apply_auto_teleop(
            policy_actions_np, next_obs, teleop_state, config["auto_teleop"], progress
        )
        adapt_timers[teleop_toggled] = config["reward_framing"]["adapt_window"]
        adapt_mask = adapt_timers > 0
        adapt_timers = np.maximum(adapt_timers - 1, 0)

        actions[step] = torch.tensor(stepped_actions, dtype=action.dtype, device=device)
        logprobs[step] = logprob

        next_obs, reward, done, infos = env.vector_step(stepped_actions)
        next_done = np.array(done)

        returns = torch.zeros((env.cfg["num_envs"], env.cfg["n_agents"]))
        for idx in range(env.cfg["num_envs"]):
            info_instance = infos[idx]
            for key, agent_reward in info_instance["rewards"].items():
                returns[idx, key] += agent_reward
        framing_scale = np.where(adapt_mask, config["reward_framing"]["adapt_scale"], config["reward_framing"]["maintenance_scale"])
        returns *= torch.tensor(framing_scale, dtype=returns.dtype, device=returns.device).unsqueeze(1)
        rewards[step] = returns.to(device)
        next_done = torch.Tensor(next_done).to(device)

        for idx, done_env in enumerate(done):
            if done_env:
                env.reset_at(idx)

    print('mean rewards at iter {:4d}:'.format(iteration), torch.mean(rewards))


    with torch.no_grad():
        next_value = agent.get_value(agent.format_input(next_obs, device)).to(device)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(env.cfg["max_time_steps"])):
            if t == env.cfg["max_time_steps"] - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(dim=-1)
            delta = rewards[t] + config['gamma'] * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + config['gamma'] * config['lambda'] * nextnonterminal * lastgaelam
        returns = advantages + values

    b_obs = obs
    b_logprobs = logprobs
    b_actions = actions
    b_advantages = advantages
    b_returns = returns
    b_values = values

    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in tqdm(range(config['num_sgd_iter'])):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > config['clip_param']).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if config['norm_adv']:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config['clip_param'], 1 + config['clip_param'])
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if config['clip_vloss']:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -config['vf_clip_param'],
                    config['vf_clip_param'],
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - config['entropy_coeff'] * entropy_loss + v_loss * config['vf_loss_coeff']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config['max_grad_norm'])
            optimizer.step()
    print('loss:', loss.detach().cpu())
    torch.save(agent.state_dict(), os.path.join(weights_dir, f'weights_epoch{iteration}.pt'))

#%%
