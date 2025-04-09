import gymnasium as gym
import Agent
import numpy as np
from tqdm import tqdm


# hyper parameters
learning_rate = 0.01
n_episodes = 100_00
start_epsilon = 1
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make("LunarLander-v3")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length = n_episodes)

agent = Agent.Agent(env,learning_rate = learning_rate,initial_epsilon = start_epsilon,epsilon_decay = epsilon_decay,final_epsilon = final_epsilon,)
obs, _= env.reset()
done = False
while not done:
    action = agent.act(obs)
    next_obs, reward, terminated, truncated, info = env.step(action) 
    agent.update(obs, action, reward , next_obs, terminated)
    done = terminated or truncated
    obs = next_obs
agent.decay_epsilon()

env.close()
