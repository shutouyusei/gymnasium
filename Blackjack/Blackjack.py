import BlackjackAgent 
import gymnasium as gym
import numpy as np

# hyper parameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make("Blackjack-v1", sab = False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length = n_episodes)

agent = BlackjackAgent.BlackjackAgent(env = env,learning_rate = learning_rate,
                       initial_epsilon = start_epsilon,epsilon_decay = epsilon_decay,
                       final_epsilon = final_epsilon,
)

from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs = obs, action = action, reward = reward,
                     terminated = terminated, next_obs = next_obs)
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
