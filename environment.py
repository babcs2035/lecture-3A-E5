# import gymnasium as gym
# import random
# from uxsim import *
# import random
# import itertools


# class TrafficSim(gym.Env):
#     def __init__(self):
#         """
#         traffic scenario: 4 signalized intersections as shown below:
#                 N1  N2
#                 |   |
#             W1--I1--I2--E1
#                 |   |
#             W2--I3--I4--E2
#                 |   |
#                 S1  S2
#         Traffic demand is generated from each boundary node to all other boundary nodes.
#         action: to determine which direction should have greenlight for every 10 seconds for each intersection. 16 actions.
#             action 1: greenlight for I1: direction 0, I2: 0, I3: 0, I4: 0, where direction 0 is E<->W, 1 is N<->S.
#             action 2: greenlight for I1: 1, I2: 0, I3: 0, I4: 0
#             action 3: greenlight for I1: 0, I2: 1, I3: 0, I4: 0
#             action 4: greenlight for I1: 1, I2: 1, I3: 0, I4: 0
#             action 5: ...
#         state: number of waiting vehicles at each incoming link. 16 dimension.
#         reward: negative of difference of total waiting vehicles
#         """

#         # consts
#         self.intersections_num = 4

#         # action
#         self.n_action = 2**self.intersections_num
#         self.action_space = gym.spaces.Discrete(self.n_action)

#         # state
#         self.n_state = self.intersections_num * self.intersections_num
#         low = np.array([0 for i in range(self.n_state)])
#         high = np.array([100 for i in range(self.n_state)])
#         self.observation_space = gym.spaces.Box(low=low, high=high)

#         self.reset()

#     def reset(self):
#         """
#         reset the env
#         """
#         seed = None  # whether demand is always random or not
#         W = World(
#             name="",
#             deltan=5,
#             tmax=4000,
#             # tmax=500,
#             print_mode=0,
#             save_mode=0,
#             show_mode=1,
#             random_seed=seed,
#             duo_update_time=600,
#         )
#         random.seed(seed)

#         # network definition
#         inf = float("inf")
#         self.intersections = []
#         self.intersections.append(W.addNode("I1", 0, 0, signal=[inf, inf]))
#         self.intersections.append(W.addNode("I2", 1, 0, signal=[inf, inf]))
#         self.intersections.append(W.addNode("I3", 0, -1, signal=[inf, inf]))
#         self.intersections.append(W.addNode("I4", 1, -1, signal=[inf, inf]))
#         W1 = W.addNode("W1", -1, 0)
#         W2 = W.addNode("W2", -1, -1)
#         E1 = W.addNode("E1", 2, 0)
#         E2 = W.addNode("E2", 2, -1)
#         N1 = W.addNode("N1", 0, 1)
#         N2 = W.addNode("N2", 1, 1)
#         S1 = W.addNode("S1", 0, -2)
#         S2 = W.addNode("S2", 1, -2)
#         # E <-> W direction: signal group 0
#         for n1, n2 in [
#             [W1, self.intersections[0]],
#             [self.intersections[0], self.intersections[1]],
#             [self.intersections[1], E1],
#             [W2, self.intersections[2]],
#             [self.intersections[2], self.intersections[3]],
#             [self.intersections[3], E2],
#         ]:
#             W.addLink(
#                 n1.name + n2.name,
#                 n1,
#                 n2,
#                 length=500,
#                 free_flow_speed=10,
#                 jam_density=0.2,
#                 signal_group=0,
#             )
#             W.addLink(
#                 n2.name + n1.name,
#                 n2,
#                 n1,
#                 length=500,
#                 free_flow_speed=10,
#                 jam_density=0.2,
#                 signal_group=0,
#             )
#         # N <-> S direction: signal group 1
#         for n1, n2 in [
#             [N1, self.intersections[0]],
#             [self.intersections[0], self.intersections[2]],
#             [self.intersections[2], S1],
#             [N2, self.intersections[1]],
#             [self.intersections[1], self.intersections[3]],
#             [self.intersections[3], S2],
#         ]:
#             W.addLink(
#                 n1.name + n2.name,
#                 n1,
#                 n2,
#                 length=500,
#                 free_flow_speed=10,
#                 jam_density=0.2,
#                 signal_group=1,
#             )
#             W.addLink(
#                 n2.name + n1.name,
#                 n2,
#                 n1,
#                 length=500,
#                 free_flow_speed=10,
#                 jam_density=0.2,
#                 signal_group=1,
#             )

#         # random demand definition
#         dt = 30
#         demand = 0.22
#         for n1, n2 in itertools.permutations([W1, W2, E1, E2, N1, N2, S1, S2], 2):
#             for t in range(0, 3600, dt):
#                 W.adddemand(n1, n2, t, t + dt, random.uniform(0, demand))

#         # store UXsim object for later re-use
#         self.W = W
#         # self.INLINKS = (
#         #     list(self.I1.inlinks.values())
#         #     + list(self.I2.inlinks.values())
#         #     + list(self.I3.inlinks.values())
#         #     + list(self.I4.inlinks.values())
#         # )
#         self.INLINKS = list()
#         for i in range(self.intersections_num):
#             self.INLINKS += list(self.intersections[i].inlinks.values())

#         # initial observation
#         observation = np.array([0 for i in range(self.n_state)])

#         # log
#         self.log_state = []
#         self.log_reward = []

#         # signal phases
#         self.current_step = 0
#         self.last_phase_change_time = [0 for _ in range(self.intersections_num)]

#         return observation, None

#     def comp_state(self):
#         """
#         compute the current state
#         """
#         vehicles_per_links = {}
#         for l in self.INLINKS:
#             vehicles_per_links[l] = (
#                 l.num_vehicles_queue
#             )  # l.num_vehicles_queue: the number of vehicles in queue in link l
#         return list(vehicles_per_links.values())

#     def comp_n_veh_queue(self):
#         return sum(self.comp_state())

#     def step(self, action_index):
#         """
#         proceed env by 1 step = `operation_timestep_width` seconds
#         """
#         self.current_step += 1
#         operation_timestep_width = 10

#         n_queue_veh_old = self.comp_n_veh_queue()

#         # change signal by action
#         # decode action
#         binstr = f"{action_index:04b}"

#         # set signal
#         signal_points = 0
#         max_point = 30
#         for i in range(self.intersections_num):
#             new_phase = int(binstr[self.intersections_num - i - 1])
#             new_time = self.current_step * operation_timestep_width
#             point = 0
#             if new_phase != self.intersections[i].signal_phase:
#                 delta_time = new_time - self.last_phase_change_time[i]
#                 self.last_phase_change_time[i] = new_time
#                 if delta_time < max_point:
#                     point = delta_time / max_point
#                 elif delta_time < 2 * max_point:
#                     point = 1
#                 elif delta_time < 3 * max_point:
#                     point = 1 - (delta_time - 2 * max_point) / max_point
#                 signal_points += point
#                 # print(f"point: {point}, delta_time: {delta_time}")
#             self.intersections[i].signal_phase = new_phase
#             self.intersections[i].signal_t = 0

#         # traffic dynamics. execute simulation for `operation_timestep_width` seconds
#         if self.W.check_simulation_ongoing():
#             self.W.exec_simulation(duration_t=operation_timestep_width)

#         # observe state
#         observation = np.array(self.comp_state())

#         # compute reward
#         reward = 0

#         n_queue_veh = self.comp_n_veh_queue()
#         delta_n_queue_veh = n_queue_veh - n_queue_veh_old
#         total_vehicle = 0
#         for l in self.INLINKS:
#             total_vehicle += l.num_vehicles
#         if total_vehicle == 0:
#             delta_queue_veh_ratio = 0
#         else:
#             delta_queue_veh_ratio = delta_n_queue_veh / total_vehicle
#         reward += -delta_queue_veh_ratio * 100

#         reward += (signal_points / self.intersections_num) * 100

#         # check termination
#         done = False
#         if self.W.check_simulation_ongoing() == False:
#             done = True

#         # log
#         self.log_state.append(observation)
#         self.log_reward.append(reward)

#         return observation, reward, done, {}, None


import gymnasium as gym
import random
from uxsim import *
import numpy as np
import itertools

class TrafficSim(gym.Env):
    def __init__(self):
        """
        拡張されたネットワークを持つ環境を定義します。
        """

        # アクションスペースの定義
        self.n_intersections = 9  # 交差点の数を9に拡張
        self.n_action = 2 ** self.n_intersections
        self.action_space = gym.spaces.Discrete(self.n_action)

        # ステートスペースの定義
        self.n_state = self.n_intersections * 4  # 各交差点に接続する4つの入力リンク
        low = np.zeros(self.n_state)
        high = np.full(self.n_state, 100)
        self.observation_space = gym.spaces.Box(low=low, high=high)

        self.reset()

    def reset(self):
        """
        環境をリセットします。
        """
        seed = None  # 需要が常にランダムかどうか
        W = World(
            name="",
            deltan=5,
            tmax=4000,
            #tmax=500,
            print_mode=0,
            save_mode=0,
            show_mode=1,
            random_seed=seed,
            duo_update_time=600,
        )
        random.seed(seed)

        # ネットワークの定義
        inf = float("inf")
        # 既存のノード
        I1 = W.addNode("I1", 0, 0, signal=[inf, inf])
        I2 = W.addNode("I2", 1, 0, signal=[inf, inf])
        I3 = W.addNode("I3", 2, 0, signal=[inf, inf])
        I4 = W.addNode("I4", 0, -1, signal=[inf, inf])
        I5 = W.addNode("I5", 1, -1, signal=[inf, inf])
        I6 = W.addNode("I6", 2, -1, signal=[inf, inf])
        I7 = W.addNode("I7", 0, -2, signal=[inf, inf])
        I8 = W.addNode("I8", 1, -2, signal=[inf, inf])
        I9 = W.addNode("I9", 2, -2, signal=[inf, inf])
        W1 = W.addNode("W1", -1, 0)
        W2 = W.addNode("W2", -1, -1)
        W3 = W.addNode("W3", -1, -2)
        E1 = W.addNode("E1", 3, 0)
        E2 = W.addNode("E2", 3, -1)
        E3 = W.addNode("E3", 3, -2)
        N1 = W.addNode("N1", 0, 1)
        N2 = W.addNode("N2", 1, 1)
        N3 = W.addNode("N3", 2, 1)
        S1 = W.addNode("S1", 0, -3)
        S2 = W.addNode("S2", 1, -3)
        S3 = W.addNode("S3", 2, -3)

        # リンクの定義
        # E <-> W 方向: 信号グループ 0
        EW_links = [
            (W1, I1), (I1, I2), (I2, I3), (I3, E1),
            (W2, I4), (I4, I5), (I5, I6), (I6, E2),
            (W3, I7), (I7, I8), (I8, I9), (I9, E3),
        ]
        for n1, n2 in EW_links:
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

        # N <-> S 方向: 信号グループ 1
        NS_links = [
            (N1, I1), (I1, I4), (I4, I7), (I7, S1),
            (N2, I2), (I2, I5), (I5, I8), (I8, S2),
            (N3, I3), (I3, I6), (I6, I9), (I9, S3),
        ]
        for n1, n2 in NS_links:
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

        # ランダム需要の定義
        dt = 30
        demand = 0.22
        boundary_nodes = [W1, W2, W3, E1, E2, E3, N1, N2, N3, S1, S2, S3]
        for n1, n2 in itertools.permutations(boundary_nodes, 2):
            for t in range(0, 3600, dt):
                W.adddemand(n1, n2, t, t + dt, random.uniform(0, demand))

        # UXsimオブジェクトを保存
        self.W = W
        self.intersections = [I1, I2, I3, I4, I5, I6, I7, I8, I9]
        self.INLINKS = []
        for node in self.intersections:
            self.INLINKS.extend(list(node.inlinks.values()))

        # 初期観測
        observation = np.array([0 for i in range(self.n_state)])

        # ログ
        self.log_state = []
        self.log_reward = []

        return observation, None

    def comp_state(self):
        """
        現在の状態を計算します。
        """
        vehicles_per_links = {}
        for l in self.INLINKS:
            vehicles_per_links[l] = l.num_vehicles_queue
        return list(vehicles_per_links.values())

    def comp_n_veh_queue(self):
        return sum(self.comp_state())

    def step(self, action_index):
        """
        環境を1ステップ進めます。
        """
        operation_timestep_width = 10

        n_queue_veh_old = self.comp_n_veh_queue()

        # アクションのデコード
        binstr = f"{action_index:0{self.n_intersections}b}"
        for idx, node in enumerate(self.intersections):
            phase = int(binstr[-(idx + 1)])
            node.signal_phase = phase
            node.signal_t = 0

        # シミュレーションの実行
        if self.W.check_simulation_ongoing():
            self.W.exec_simulation(duration_t=operation_timestep_width)

        # 状態の観測
        observation = np.array(self.comp_state())

        # compute reward
        reward = 0

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
        reward += -delta_queue_veh_ratio * 100

        ## reward 2: signal points
        # reward += (signal_points / self.intersections_num) * 100

        # 終了条件の確認
        done = False
        if not self.W.check_simulation_ongoing():
            done = True

        # ログ
        self.log_state.append(observation)
        self.log_reward.append(reward)

        return observation, reward, done, {}, None
