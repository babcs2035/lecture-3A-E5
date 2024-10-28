import gymnasium as gym
import random
from uxsim import *
import random
import itertools
import csv
import sys

args = sys.argv
rewards_num = [int(i) for i in args[1:]]


class TrafficSim(gym.Env):
    def __init__(self):

        # consts
        self.intersections_num = 11

        # action
        self.n_action = 2**self.intersections_num
        self.action_space = gym.spaces.Discrete(self.n_action)

        # state
        self.n_state = 41  # number of links (15) * 2 + 11
        low = np.array([0 for i in range(self.n_state)])
        high = np.array([100 for i in range(self.n_state)])
        self.observation_space = gym.spaces.Box(low=low, high=high)

        self.reset()

    def reset(self):
        """
        reset the env
        """
        seed = None  # whether demand is always random or not
        W = World(
            name="",
            deltan=5,
            tmax=4000,
            # tmax=500,
            print_mode=0,
            save_mode=0,
            show_mode=1,
            random_seed=seed,
            duo_update_time=600,
        )
        random.seed(seed)

        # network definition
        inf = float("inf")
        is_first_row = True
        self.intersections = []
        with open("nodes.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                if is_first_row:
                    is_first_row = False
                    continue
                self.intersections.append(
                    W.addNode(
                        f"I{row[0]}",
                        int(row[1]),
                        abs(1800 - int(row[2])),
                        signal=[inf, inf],
                    )
                )

        self.nodes = []
        self.nodes.append(W.addNode("N0", 880, 1800 - 770, signal=[inf, inf]))
        self.nodes.append(W.addNode("N1", 978, 1800 - 486, signal=[inf, inf]))
        self.nodes.append(W.addNode("N2", 1054, 1800 - 454, signal=[inf, inf]))
        self.nodes.append(W.addNode("N3", 1687, 1800 - 313, signal=[inf, inf]))
        self.nodes.append(W.addNode("E3", 1737, 1800 - 363, signal=[inf, inf]))
        self.nodes.append(W.addNode("E7", 1870, 1800 - 911, signal=[inf, inf]))
        self.nodes.append(W.addNode("E10", 2176, 1800 - 1669, signal=[inf, inf]))
        self.nodes.append(W.addNode("S10", 2126, 1800 - 1719, signal=[inf, inf]))
        self.nodes.append(W.addNode("S9", 1529, 1800 - 1778, signal=[inf, inf]))
        self.nodes.append(W.addNode("S8", 1080, 1800 - 1705, signal=[inf, inf]))
        self.nodes.append(W.addNode("W8", 1030, 1800 - 1655, signal=[inf, inf]))

        # makelink
        is_first_row = True
        with open("dat.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                if is_first_row:
                    is_first_row = False
                    continue
                self.make_link(
                    W,
                    self.intersections[int(row[0])],
                    self.intersections[int(row[1])],
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                )
        # makelink_node
        is_first_row = True
        with open("dat2.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                if is_first_row:
                    is_first_row = False
                    continue
                self.make_link(
                    W,
                    self.intersections[int(row[0])],
                    self.nodes[int(row[1])],
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                )
        # random demand definition
        dt = 30
        demand = 0.22
        """
        intersections_ = []
        for I in self.intersections:
            if I.name != "I5" and I.name != "I6":
                intersections_.append(I)
        for n1, n2 in itertools.permutations(intersections_, 2):
            for t in range(0, 3600, dt):
                W.adddemand(n1, n2, t, t + dt, random.uniform(0, demand))
        """
        intersections_ = []
        for I in self.nodes:
            intersections_.append(I)
        for n1, n2 in itertools.permutations(intersections_, 2):
            for t in range(0, 3600, dt):
                W.adddemand(n1, n2, t, t + dt, random.uniform(0, demand))

        # store UXsim object for later re-use
        self.W = W
        self.INLINKS = list()
        for i in range(self.intersections_num):
            self.INLINKS += list(self.intersections[i].inlinks.values())

        # initial observation
        observation = np.array([0 for i in range(self.n_state)])

        # log
        self.log_state = []
        self.log_reward = []

        # signal phases
        self.current_step = 0
        self.last_phase_change_time = [0 for _ in range(self.intersections_num)]

        return observation, None

    def comp_state(self):
        """
        compute the current state
        """
        vehicles_per_links = {}
        for l in self.INLINKS:
            vehicles_per_links[l] = (
                l.num_vehicles_queue
            )  # l.num_vehicles_queue: the number of vehicles in queue in link l
        return list(vehicles_per_links.values())

    def comp_n_veh_queue(self):
        return sum(self.comp_state())

    def step(self, action_index):
        """
        proceed env by 1 step = `operation_timestep_width` seconds
        """
        self.current_step += 1
        operation_timestep_width = 10

        n_queue_veh_old = self.comp_n_veh_queue()

        # change signal by action
        # decode action
        binstr = f"{action_index:011b}"

        # set signal
        signal_points = 0
        # max_point = 30
        for i in range(self.intersections_num):
            new_phase = int(binstr[self.intersections_num - i - 1])
            new_time = self.current_step * operation_timestep_width
            point = 0
            if new_phase != self.intersections[i].signal_phase:
                delta_time = new_time - self.last_phase_change_time[i]
                self.last_phase_change_time[i] = new_time
                # if delta_time < max_point:
                #     point = delta_time / max_point
                # elif delta_time < 2 * max_point:
                #     point = 1
                # elif delta_time < 3 * max_point:
                #     point = 1 - (delta_time - 2 * max_point) / max_point
                if delta_time < 10 or 60 < delta_time:
                    point = -1
                signal_points += point
                # print(f"point: {point}, delta_time: {delta_time}")
            self.intersections[i].signal_phase = new_phase
            self.intersections[i].signal_t = 0

        # traffic dynamics. execute simulation for `operation_timestep_width` seconds
        if self.W.check_simulation_ongoing():
            self.W.exec_simulation(duration_t=operation_timestep_width)

        # observe state
        observation = np.array(self.comp_state())

        # compute reward
        total_rewards = 3
        rewards = [0 for _ in range(total_rewards)]

        ## reward 1: negative ratio of difference of total waiting vehicles
        n_queue_veh = self.comp_n_veh_queue()
        delta_n_queue_veh = n_queue_veh - n_queue_veh_old
        # reward += -delta_n_queue_veh
        total_vehicle = 0
        for l in self.INLINKS:
            total_vehicle += l.num_vehicles
        if total_vehicle == 0:
            delta_queue_veh_ratio = 0
        else:
            delta_queue_veh_ratio = delta_n_queue_veh / total_vehicle
        rewards[0] = -delta_queue_veh_ratio * 100

        ## reward 2: signal points
        rewards[1] = (signal_points / self.intersections_num) * 100

        # reward 3: pressure
        pressure = 0
        for i in range(self.intersections_num):  # i is checking node
            self.INLINKS_press = list(self.intersections[i].inlinks.values())
            self.OUTLINKS_press = list(self.intersections[i].outlinks.values())
            for l1 in self.INLINKS_press:
                j = l1.start_node  # j->i is checking line
                in_press = l1.num_vehicles_queue
                out_press = 0
                for l2 in self.OUTLINKS_press:
                    if l2.name.endswith(j.name) == 0:
                        out_press += l2.num_vehicles_queue
                out_press /= len(self.INLINKS_press) - 1
                pressure += abs(in_press - out_press)
        rewards[2] = -pressure / 100

        # print(rewards)
        reward = sum([rewards[a - 1] for a in rewards_num])

        # check termination
        done = False
        if self.W.check_simulation_ongoing() == False:
            done = True

        # log
        self.log_state.append(observation)
        self.log_reward.append(reward)

        return observation, reward, done, {}, None

    def make_link(
        self,
        W,
        n1,
        n2,
        length_0,
        free_flow_speed_0,
        jam_density_0,
        signal_group_a,
        signal_group_b,
    ):
        W.addLink(
            n1.name + n2.name,
            n1,
            n2,
            length=length_0,
            free_flow_speed=free_flow_speed_0,
            jam_density=jam_density_0 * 0.2,
            signal_group=signal_group_a,
        )
        W.addLink(
            n2.name + n1.name,
            n2,
            n1,
            length=length_0,
            free_flow_speed=free_flow_speed_0,
            jam_density=jam_density_0 * 0.2,
            signal_group=signal_group_b,
        )
