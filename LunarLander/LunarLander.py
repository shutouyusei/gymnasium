import gymnasium as gym
import Agent
from tqdm import tqdm

env = gym.make("LunarLander-v3")

n_episodes = 100_0

agent = Agent.Agent(env,learning_rate = 0.01,initial_epsilon = 0,epsilon_decay = 0,final_epsilon = 0,discount_factor = 0)
                    

for episode in tqdm(range(n_episodes)):
    obs, _= env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward , next_obs, terminated)
        done = terminated or truncated
        obs = next_obs

env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()
done  = False
while not done:
    action = agent.act(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
    obs = next_obs

env.close()

