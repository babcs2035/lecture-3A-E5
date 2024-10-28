import gymnasium as gym
import random
from uxsim import *
import random
import itertools
import sys

args = sys.argv
rewards_num = [int(i) for i in args[1:]]


class TrafficSim(gym.Env):
    def __init__(self):
        """
        traffic scenario: 4 signalized intersections as shown below:
                N1  N2
                |   |
            W1--I1--I2--E1
                |   |
            W2--I3--I4--E2
                |   |
                S1  S2
        Traffic demand is generated from each boundary node to all other boundary nodes.
        action: to determine which direction should have greenlight for every 10 seconds for each intersection. 16 actions.
            action 1: greenlight for I1: direction 0, I2: 0, I3: 0, I4: 0, where direction 0 is E<->W, 1 is N<->S.
            action 2: greenlight for I1: 1, I2: 0, I3: 0, I4: 0
            action 3: greenlight for I1: 0, I2: 1, I3: 0, I4: 0
            action 4: greenlight for I1: 1, I2: 1, I3: 0, I4: 0
            action 5: ...
        state: number of waiting vehicles at each incoming link. 16 dimension.
        reward: negative of difference of total waiting vehicles
        """

        # consts
        self.intersections_num = 4

        # action
        self.n_action = 2**self.intersections_num
        self.action_space = gym.spaces.Discrete(self.n_action)

        # state
        self.n_state = self.intersections_num * self.intersections_num
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
        self.intersections = []
        self.intersections.append(W.addNode("I1", 0, 0, signal=[inf, inf]))
        self.intersections.append(W.addNode("I2", 1, 0, signal=[inf, inf]))
        self.intersections.append(W.addNode("I3", 0, -1, signal=[inf, inf]))
        self.intersections.append(W.addNode("I4", 1, -1, signal=[inf, inf]))
        W1 = W.addNode("W1", -1, 0)
        W2 = W.addNode("W2", -1, -1)
        E1 = W.addNode("E1", 2, 0)
        E2 = W.addNode("E2", 2, -1)
        N1 = W.addNode("N1", 0, 1)
        N2 = W.addNode("N2", 1, 1)
        S1 = W.addNode("S1", 0, -2)
        S2 = W.addNode("S2", 1, -2)
        # E <-> W direction: signal group 0
        for n1, n2 in [
            [W1, self.intersections[0]],
            [self.intersections[0], self.intersections[1]],
            [self.intersections[1], E1],
            [W2, self.intersections[2]],
            [self.intersections[2], self.intersections[3]],
            [self.intersections[3], E2],
        ]:
            W.addLink(
                n1.name + n2.name,
                n1,
                n2,
                length=500,
                free_flow_speed=10,
                jam_density=0.2,
                signal_group=0,
            )
            W.addLink(
                n2.name + n1.name,
                n2,
                n1,
                length=500,
                free_flow_speed=10,
                jam_density=0.2,
                signal_group=0,
            )
        # N <-> S direction: signal group 1
        for n1, n2 in [
            [N1, self.intersections[0]],
            [self.intersections[0], self.intersections[2]],
            [self.intersections[2], S1],
            [N2, self.intersections[1]],
            [self.intersections[1], self.intersections[3]],
            [self.intersections[3], S2],
        ]:
            W.addLink(
                n1.name + n2.name,
                n1,
                n2,
                length=500,
                free_flow_speed=10,
                jam_density=0.2,
                signal_group=1,
            )
            W.addLink(
                n2.name + n1.name,
                n2,
                n1,
                length=500,
                free_flow_speed=10,
                jam_density=0.2,
                signal_group=1,
            )

        # random demand definition
        dt = 30
        demand = 0.22
        for n1, n2 in itertools.permutations([W1, W2, E1, E2, N1, N2, S1, S2], 2):
            for t in range(0, 3600, dt):
                W.adddemand(n1, n2, t, t + dt, random.uniform(0, demand))

        # store UXsim object for later re-use
        self.W = W
        # self.INLINKS = (
        #     list(self.I1.inlinks.values())
        #     + list(self.I2.inlinks.values())
        #     + list(self.I3.inlinks.values())
        #     + list(self.I4.inlinks.values())
        # )
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
        binstr = f"{action_index:04b}"

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
                in_press *= 3
                pressure += abs(in_press - out_press)
        rewards[2] = -pressure / 100

        reward = sum([rewards[a - 1] for a in rewards_num])

        # check termination
        done = False
        if self.W.check_simulation_ongoing() == False:
            done = True

        # log
        self.log_state.append(observation)
        self.log_reward.append(reward)

        return observation, reward, done, {}, None
