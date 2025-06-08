import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Environment
env = gym.make("FrozenLake-v1", is_slippery=False)  # deterministic version

# Parameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1         # for epsilon-greedy
num_episodes = 5000

# Initialize Q-table
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# SARSA learning
for episode in range(num_episodes):
    state = env.reset()[0]
    action = epsilon_greedy_policy(state, epsilon)
    done = False

    while not done:
        next_state, reward, done, truncated, _ = env.step(action)
        next_action = epsilon_greedy_policy(next_state, epsilon)
        
        # SARSA update
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )
        
        state = next_state
        action = next_action

# Compute value function from Q
V = np.zeros(env.observation_space.n)
for state in range(env.observation_space.n):
    V[state] = np.max(Q[state])  # value under greedy policy

# Reshape to 4x4 grid and visualize
V_grid = V.reshape((4, 4))

plt.figure(figsize=(6, 6))
plt.imshow(V_grid, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="State Value")
plt.title("State-Value Function Learned by SARSA")
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{V_grid[i, j]:.2f}", ha='center', va='center', color='black')
plt.show()
