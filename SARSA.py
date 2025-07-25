import gymnasium as gym
import numpy as np



env = gym.make("FrozenLake-v1", is_slippery=True) 

# Parameters
alpha = 0.01
gamma = 0.9
epsilon = 0.1         # for epsilon-greedy
num_episodes = 5000


Q = {i:[0.0001 for j in range(env.action_space.n)] for i in range(env.observation_space.n)}

V = {i:0.00001 for i in range(env.observation_space.n)}

# Epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# SARSA learning
for episode in range(1000):
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



for state in range(env.observation_space.n):
    V[state] = np.max(Q[state])  # value under greedy policy


for s in range(env.observation_space.n):
    print(f"V({s}) = {V[s]}")


