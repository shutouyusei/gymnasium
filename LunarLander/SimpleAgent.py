import torch
import Net
import gymnasium as gym
import torch.optim as optim
from gymnasium.spaces import Box
import numpy as np
import torch.nn.functional as F

class Agent:
    def __init__(self, env:gym.Env, initial_epsilon:float, epsilon_decay:float, final_epsilon:float, discount_factor:float = 0.95):
        self.env = env
        # ニューラルネットワーク
        self.net = Net.Net()
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
            with torch.no_grad():
                action = self.net(torch_state)
            return action.argmax().item()

    def update(self, state, action, reward, next_state, done):
        q_value = self.net(torch.tensor(state, dtype = torch.float32))
        q_value = q_value[action]
        with torch.no_grad():
            future_q_value = self.net(torch.tensor(next_state, dtype = torch.float32)).max()
        target = torch.tensor(reward, dtype=torch.float32)
        if not done:
            target = torch.tensor(reward, dtype=torch.float32) + self.discount_factor * future_q_value
        loss_fn  = torch.nn.MSELoss()
        loss = loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_error.append(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
