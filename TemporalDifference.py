import gymnasium as gym
import numpy as np
from collections import defaultdict

# Environment
env = gym.make("FrozenLake-v1", is_slippery=True)  
# env = gym.make("FrozenLake-v1", is_slippery=False)  deterministic for simplicity

# Parameters
alpha = 0.1         # learning rate
gamma = 0.9        # discount factor
num_episodes = 5000


V = {i:0.01 for i in range(env.observation_space.n)}

# TD(0) algorithm
for episode in range(100):
    state = env.reset()[0]  
    done = False

    while not done:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)

        V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        
        state = next_state

# Display learned value function
for s in range(env.observation_space.n):
    print(f"V({s}) = {V[s]:.2f}")



