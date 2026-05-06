import os

# Important: make sure pygame is NOT headless
os.environ.pop("SDL_VIDEODRIVER", None)

import argparse
import random

import numpy as np
import pygame
import torch
from env_line import PassageEnv
from model import Agent

num_agents = 5

agent_formation = (
    np.array(
        [
            [-0.5, 0],
            [0, 0],
            [1, 0],
            [1.5, 0],
            [2, 0],
        ]
    )
    * 0.5
).tolist()

config = {
    "seed": 0,
    "lr": 5e-5,
    "gamma": 0.995,
    "lambda": 0.95,
    "clip_param": 0.2,
    "entropy_coeff": 0.001,
    "vf_clip_param": 1.0,
    "vf_loss_coeff": 1.0,
    "max_grad_norm": 0.5,
    "norm_adv": True,
    "clip_vloss": True,
    "model": {
        "custom_model_config": {
            "activation": "relu",
            "msg_features": 32,
            "comm_range": 2.0,
        },
    },
    "env_config": {
        "world_dim": (4.0, 5.0),
        "dt": 0.05,
        "num_envs": 1,
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
        "render": True,
        "render_px_per_m": 160,
        "max_v": 1.0,
        "max_a": 1.0,
        "min_a": -1.0,
    },
}


def run_episode(env, agent, device, render):
    obs = env.vector_reset()
    total_reward = 0.0
    done = False
    steps = 0
    clock = pygame.time.Clock() if render else None

    while not done and steps < env.cfg["max_time_steps"]:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    raise SystemExit

        with torch.no_grad():
            x = agent.format_input(obs, device)
            action, _, _, _ = agent.get_action_and_value(x)

        obs, reward, dones, _ = env.vector_step(action.cpu().numpy())
        total_reward += float(reward[0])
        done = bool(dones[0])
        steps += 1

        if render:
            env.render_ours(mode="human")
            clock.tick(int(1 / env.cfg["dt"]))

    return total_reward, steps, done


def main():
    parser = argparse.ArgumentParser(description="Evaluate AFOR policy checkpoints")
    parser.add_argument("--weights", default="weights/real-line2/weights_epoch1.pt", help="Checkpoint path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render with pygame")
    args = parser.parse_args()

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["env_config"]["device"] = device
    config["env_config"]["render"] = bool(args.render)

    env = PassageEnv(config["env_config"])
    agent = Agent(env, config).to(device)

    agent.load_state_dict(torch.load(args.weights, map_location=device))
    agent.eval()

    rewards = []
    lengths = []
    successes = 0

    for ep in range(args.episodes):
        ep_reward, ep_steps, done = run_episode(env, agent, device, args.render)
        rewards.append(ep_reward)
        lengths.append(ep_steps)
        successes += int(done)
        print(f"episode {ep+1}/{args.episodes}: reward={ep_reward:.3f}, steps={ep_steps}, done={done}")

    print("\n=== Evaluation Summary ===")
    print(f"device: {device}")
    print(f"weights: {args.weights}")
    print(f"episodes: {args.episodes}")
    print(f"success_rate: {successes / max(1, args.episodes):.3f}")
    print(f"mean_reward: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    print(f"mean_length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")

    env.close()


if __name__ == "__main__":
    main()
