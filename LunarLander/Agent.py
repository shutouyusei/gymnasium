import torch
import Net
import gymnasium as gym
import torch.optim as optim
from gymnasium.spaces import Box
import numpy as np
import torch.nn.functional as F

class Agent:
    def __init__(self, env:gym.Env, initial_epsilon:float, epsilon_decay:float, final_epsilon:float,memory, discount_factor:float = 0.98):
        self.env = env
        self.memory = memory
        self.net = Net.Net()
        self.target_net = Net.Net()
        self.target_net.load_state_dict(self.net.state_dict())
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.optimizer = optim.Adam(self.net.parameters(),lr = 1e-3)
        self.training_error = []
        self.steps = 0

    def act(self, state) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            torch_state = torch.tensor(state).float()
            with torch.no_grad():
                action = self.net(torch_state)
            return action.argmax().item()
    def update_target_network(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def update(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        if self.memory.size() > self.memory.batch_size * 10:
            self.steps += 1
            batch = self.memory.sample()
            state, action, reward, next_state, done = zip(*batch)
            state = torch.tensor(np.stack(state), dtype = torch.float32)
            action = torch.tensor(action, dtype = torch.int64).unsqueeze(1)
            reward = torch.tensor(reward, dtype = torch.float32)
            next_state = torch.tensor(np.stack(next_state), dtype = torch.float32)
            done = torch.tensor(done, dtype = torch.float32)
            q_value = self.net(state).gather(1, action)
            q_value = q_value.squeeze(1)
            with torch.no_grad():
                future_q_value = self.target_net(next_state).max(1)[0]
                target = reward + self.discount_factor * future_q_value * (1 - done)
            loss_fn  = torch.nn.MSELoss()
            loss = loss_fn(q_value, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.training_error.append(loss.item())
            if self.steps % 5 == 0:
                self.update_target_network()

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
