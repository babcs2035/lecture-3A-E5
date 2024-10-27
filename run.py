import matplotlib.pyplot as plt
from itertools import count
import torch
from uxsim import *
import copy
import sys
import os
from dqn import *

args = sys.argv
num_array = [int(i) for i in args[1:]]
savefile_prefix = "_reward" + "".join([str(i) for i in num_array])
os.makedirs(f"./out{savefile_prefix}", exist_ok=True)
print("rewards:", num_array)
print("savefile_prefix:", savefile_prefix)
# sys.stdout = open(f"./out{savefile_prefix}/.out", "w")

num_episodes = 1024
# num_episodes = 1

log_states = []
log_epi_average_delay = []
best_average_delay = 9999999999999999999999999
best_W = None
best_i_episode = -1
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    log_states.append([])
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        log_states[-1].append(state)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            log_epi_average_delay.append(env.W.analyzer.average_delay)
            print(
                f"episode {i_episode}: [{env.W.analyzer.average_delay : .3f}]", end=" "
            )
            if env.W.analyzer.average_delay < best_average_delay:
                print("current best episode!")
                best_average_delay = env.W.analyzer.average_delay
                best_W = copy.deepcopy(env.W)
                best_i_episode = i_episode

                env.W.save_mode = True
                env.W.show_mode = False
                env.W.name = savefile_prefix
                env.W.analyzer.print_simple_stats(force_print=True)
                env.W.analyzer.macroscopic_fundamental_diagram()
                env.W.analyzer.time_space_diagram_traj_links(
                    [["W1I1", "I1I2", "I2E1"], ["N1I1", "I1I3", "I3S1"]],
                    figsize=(12, 3),
                    xlim=[3500, 4000],
                )
                for t in list(range(0, env.W.TMAX, int(env.W.TMAX / 4))):
                    env.W.analyzer.network(
                        t, detailed=1, network_font_size=0, figsize=(4, 4)
                    )
                plt.figure(figsize=(8, 6))
                plt.plot(log_epi_average_delay, "r.")
                plt.xlabel("episode")
                plt.ylabel("average delay (s)")
                plt.grid()
                plt.savefig(f"out{savefile_prefix}/log_epi_average_delay.png")
            else:
                print("")
            break

    if i_episode % 10 == 0 or i_episode == num_episodes - 1:
        plt.figure(figsize=(8, 6))
        plt.plot(log_epi_average_delay, "r.")
        plt.xlabel("episode")
        plt.ylabel("average delay (s)")
        plt.grid()
        plt.savefig(f"out{savefile_prefix}/log_epi_average_delay.png")
