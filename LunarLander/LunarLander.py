import gymnasium as gym
import Agent
import numpy as np
from tqdm import tqdm


# hyper parameters
learning_rate = 0.01
n_episodes = 100_00
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make("LunarLander-v3")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length = n_episodes)

agent = Agent.Agent(env,learning_rate = learning_rate,initial_epsilon = start_epsilon,epsilon_decay = epsilon_decay,final_epsilon = final_epsilon,)
                    
for episode in tqdm(range(n_episodes)):
    obs, _= env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward , next_obs, terminated)
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon()

from matplotlib import pyplot as plt

def get_moving_avs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode = convolution_mode
    )

rolling_length = 500
figs, axs = plt.subplots(ncols = 3, figsize = (12, 5))

axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avs(
    env.return_queue,
    rolling_length,
    "valid"
)

axs[0].plot(range(len(reward_moving_average)),reward_moving_average)

axs[1].set_title("Episode lengths")
q_values_moving_average = get_moving_avs(
    agent.training_error,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(q_values_moving_average)), q_values_moving_average)

axs[2].set_title("training_error")
training_error_moving_average = get_moving_avs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

plt.tight_layout()
plt.show()

env = gym.make("LunarLander-v3", render_mode = "human")
obs, _ = env.reset()
done = False
while not done:
    action = agent.act(obs)
    obs,reward,terminated,truncated,info = env.step(action)
    done = terminated or truncated

env.close()
