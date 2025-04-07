import BlackjackAgent 
import gymnasium as gym

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
