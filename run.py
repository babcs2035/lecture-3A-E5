import matplotlib.pyplot as plt
from itertools import count
import torch
from uxsim import *
import copy
import sys
import os
import resource
from environment import *

resource.setrlimit(resource.RLIMIT_STACK, (-1, -1))
sys.setrecursionlimit(1000000)

args = sys.argv
num_array = [int(i) for i in args[1:]]
savefile_prefix = "_reward" + "".join([str(i) for i in num_array])
os.makedirs(f"./out{savefile_prefix}", exist_ok=True)
print("rewards:", num_array)
print("savefile_prefix:", savefile_prefix, "\n")
# sys.stdout = open(f"./out{savefile_prefix}/.out", "w")

num_episodes = 256

log_epi_average_delay = []
best_average_delay = 9999999999999999999999999


def train_marl(env):
    """マルチエージェント学習の実行"""
    log_rewards = []
    best_average_delay = float("inf")
    best_agents = None

    for i_episode in range(num_episodes):
        observations, _ = env.reset()
        episode_rewards = [[] for _ in range(env.num_agents)]

        while True:
            # 各エージェントの行動選択
            actions = []
            for i, agent in enumerate(env.agents):
                action = agent.select_action(observations[i])
                actions.append(action.item())

            # 環境ステップの実行
            next_observations, rewards, done, _ = env.step(actions)

            # 経験の保存
            for i, agent in enumerate(env.agents):
                agent.memory.push(
                    observations[i],
                    torch.tensor([[actions[i]]], device=env.device),
                    None if done else next_observations[i],
                    rewards[i],
                )

                # モデルの最適化
                agent.optimize_model()
                agent.update_target_net()

                episode_rewards[i].append(rewards[i].item())

            if done:
                break

            observations = next_observations

        # エピソード終了時の処理
        average_delay = env.base_env.W.analyzer.average_delay
        log_epi_average_delay.append(average_delay)
        print(f"episode {i_episode}: [{average_delay : .3f}]", end=" ")

        if average_delay < best_average_delay:
            print("current best episode!")
            best_average_delay = average_delay

            env.base_env.W.save_mode = True
            env.base_env.W.show_mode = False
            env.base_env.W.name = savefile_prefix
            env.base_env.W.analyzer.print_simple_stats(force_print=True)
            env.base_env.W.analyzer.macroscopic_fundamental_diagram()
            env.base_env.W.analyzer.time_space_diagram_traj_links(
                [
                    ["N0I0", "I0I4", "I4I8", "I8S8"],
                    ["N2I2", "I2I5", "I5I6", "I6I9", "I9S9"],
                    ["N3I3", "I3I7", "I7I10", "I10S10"],
                    ["N0I0", "I0I1", "I1I2", "I2I3", "I3E3"],
                    ["I4I6", "I6I7", "I7E7"],
                    ["W8I8", "I8I9", "I9I10", "I10E10"],
                ],
                figsize=(48, 3),
            )
            env.base_env.W.analyzer.network_anim(
                animation_speed_inverse=5,
                timestep_skip=64,
                detailed=1,
                network_font_size=0,
                figsize=(4, 4),
            )
            plt.figure(figsize=(8, 6))
            plt.plot(log_epi_average_delay, "r.")
            plt.xlabel("episode")
            plt.ylabel("average delay (s)")
            plt.ylim(0, 800)
            plt.grid()
            plt.savefig(f"out{savefile_prefix}/log_epi_average_delay.png")
        else:
            print("")

        if i_episode % 5 == 0 or i_episode == num_episodes - 1:
            plt.figure(figsize=(8, 6))
            plt.plot(log_epi_average_delay, "r.")
            plt.xlabel("episode")
            plt.ylabel("average delay (s)")
            plt.ylim(0, 800)
            plt.grid()
            plt.savefig(f"out{savefile_prefix}/log_epi_average_delay.png")

        if i_episode == num_episodes - 1:
            with open(f"out{savefile_prefix}/log_epi_average_delay.txt", "w") as f:
                for item in log_epi_average_delay:
                    f.write(f"{item}\n")

    return best_agents, log_rewards


# 環境の初期化
base_env = TrafficSim()
marl_env = MARLTrafficEnv(base_env)

# 学習の実行
best_agents, log_rewards = train_marl(marl_env)

# 最良のモデルの保存
for i, state_dict in enumerate(best_agents):
    torch.save(state_dict, f"agent_{i}_best.pth")
