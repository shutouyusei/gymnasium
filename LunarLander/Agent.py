import torch
import Net
import gymnasium as gym
import torch.optim as optim
from gymnasium.spaces import Box
import numpy as np

class Agent:
    def __init__(self, env:gym.Env, learning_rate:float,
                 initial_epsilon:float, epsilon_decay:float,
                 final_epsilon:float, discount_factor:float = 0.95):
        self.env = env
        # ニューラルネットワーク
        self.net = Net.Net()
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.optimizer = optim.Adam(self.net.parameters(),lr = 1e-3)

        self.training_error = []

    def act(self, state) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            torch_state = torch.tensor(state).float()
            #　勾配計算の無効化を行い、ニューラルネットワークの更新を止める
            # そのため、この場合のfowardは推論モード
            with torch.no_grad():
                q_value = self.net(torch_state)
            return q_value.argmax().item()

    def update(self, state, action, reward, next_state, done):
        torch_state = torch.tensor(state,requires_grad = True).float()
        future_q_value = self.net(torch_state).max()
        torch_state = torch.tensor(next_state,requires_grad = True).float()
        target = self.net(torch_state).max()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(target, future_q_value + reward )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

