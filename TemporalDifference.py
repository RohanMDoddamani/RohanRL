import gym
import numpy as np
from collections import defaultdict

# Environment
env = gym.make("FrozenLake-v1", is_slippery=False)  # deterministic for simplicity

# Parameters
alpha = 0.1         # learning rate
gamma = 0.99        # discount factor
num_episodes = 5000

# Initialize value function
V = defaultdict(float)

# Define a simple policy: move right (2) or down (1) with equal probability
def policy(state):
    return np.random.choice([1, 2])  # down or right

# TD(0) algorithm
for episode in range(num_episodes):
    state = env.reset()[0]  # Gym v0.26+ returns (obs, info)
    done = False

    while not done:
        action = policy(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        # TD(0) update
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        
        state = next_state

# Display learned value function
for s in range(env.observation_space.n):
    print(f"V({s}) = {V[s]:.2f}")
