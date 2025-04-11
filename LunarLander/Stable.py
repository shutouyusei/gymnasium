import gymnasium as gym

from stable_baseline3 import HerReplayBuffer, DQN

env = gym.make("LunarLander-v3")
model = HER("MlpPolicy", env, verbose = 1)
model.learn(total_timesteps = 3000)

env = gym.make("LunarLander-v3", render_mode = "human")
obs,_= env.reset()

while True:
    action, _states = model.predict(obs,deterministic = True)
    obs, rewards, terminated, truncated,info = env.step(action)
    if terminated or truncated :
        break

env.close()
