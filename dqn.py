import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from uxsim import *
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
steps_done = 0


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQNNetwork, self).__init__()
        n_neurons = 64
        n_layers = 3
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_observations, n_neurons))
        for _ in range(n_layers):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        self.layer_last = nn.Linear(n_neurons, n_actions)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.layer_last(x)


class DQNAgent:
    def __init__(self, n_observations, n_actions, device, agent_id):
        # ネットワークの定義
        self.policy_net = DQNNetwork(n_observations, n_actions).to(device)
        self.target_net = DQNNetwork(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # エージェント固有のパラメータ
        self.device = device
        self.agent_id = agent_id
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        # ハイパーパラメータ
        self.BATCH_SIZE = 128
        self.GAMMA = 0.95
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200  # 適切な値に調整
        self.TAU = 0.01
        self.LR = 2e-2

        # オプティマイザの設定
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.LR, amsgrad=True
        )

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.policy_net.layer_last.out_features)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Huberロスの計算
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 最適化ステップ
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        # ポリシーネットワークのパラメータをターゲットネットワークにソフト更新
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.TAU * policy_param.data + (1.0 - self.TAU) * target_param.data
            )
