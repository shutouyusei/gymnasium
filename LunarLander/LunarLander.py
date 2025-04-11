import gymnasium as gym
import Agent
import numpy as np
from tqdm import tqdm
import Memory

# hyper parameters
learning_rate = 0.1
n_episodes = 300_0
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes)
final_epsilon = 0.1

env = gym.make("LunarLander-v3")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length = n_episodes)
memory = Memory.ExperienceReplay(n_episodes, 32)

agent = Agent.Agent(env,initial_epsilon = start_epsilon,epsilon_decay = epsilon_decay,final_epsilon = final_epsilon,memory = memory,)
episode_rewards = []
                    
for episode in tqdm(range(n_episodes)):
    obs, _= env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward , next_obs, terminated)
        done = terminated or truncated
        obs = next_obs
        total_reward += reward
    agent.decay_epsilon()
    episode_rewards.append(total_reward)

from matplotlib import pyplot as plt

def plot_rewards(reward_list, window=100):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list, label='Episode Reward')
    
    # 移動平均（平滑化）
    if len(reward_list) >= window:
        moving_avg = np.convolve(reward_list, np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(reward_list)), moving_avg, label=f'{window}-Episode Moving Average', color='orange')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_plot.png")  # 保存する場合
    plt.show()

plot_rewards(episode_rewards, window=100)

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
    np.array(env.return_queue),
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
plt.savefig("view.png")

env = gym.make("LunarLander-v3", render_mode = "human")
obs, _ = env.reset()
done = False
while not done:
    action = agent.act(obs)
    obs,reward,terminated,truncated,info = env.step(action)
    done = terminated or truncated

env.close()
