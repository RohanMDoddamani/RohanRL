import numpy as np

# States (row, col)
states = [(0,0), (0,1), (1,0), (1,1)]
n_states = len(states)

# Initialize uniform belief
belief = np.full(n_states, 1.0 / n_states)

# Observation model: P(o | s)
def observation_model(state, obs):
    if obs == "Wall-North":
        if state[0] == 0:  # top row has north wall
            return 0.9
        else:
            return 0.1
    return 0.5

# Transition model: P(s' | s, a)
def transition_model(prev_state, action):
    next_states = []
    for s in states:
        prob = 0.0
        if action == "Up":
            if prev_state[0] > 0 and (prev_state[0]-1, prev_state[1]) == s:
                prob = 0.9
            elif s == prev_state:
                prob = 0.1
        next_states.append(prob)
    return np.array(next_states)

# Belief update
def update_belief(belief, action, observation):
    prior = np.zeros(n_states)
    for i, s_prime in enumerate(states):
        for j, s in enumerate(states):
            T = transition_model(s, action)[i]
            prior[i] += T * belief[j]

    # Update with observation
    new_belief = np.zeros(n_states)
    for i, s in enumerate(states):
        new_belief[i] = observation_model(s, observation) * prior[i]

    # Normalizing the distribution
    new_belief /= np.sum(new_belief)
    return new_belief

# Example belief update
belief = update_belief(belief, action="Up", observation="Wall-North")
print("Updated Belief:", belief)






