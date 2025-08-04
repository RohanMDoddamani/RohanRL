import numpy as np
import matplotlib.pyplot as plt

# bandit environment
class BernoulliBandit:
    def __init__(self, probs):
        self.probs = probs  # True reward probabilities

    def pull(self, arm):
        return 1 if np.random.rand() < self.probs[arm] else 0


num_arms = 3
true_probs = [0.2, 0.5, 0.75]
bandit = BernoulliBandit(true_probs)


alpha = np.ones(num_arms)
beta = np.ones(num_arms)

print(alpha)
print(beta)
num_rounds = 1000
rewards = []

for _ in range(num_rounds):

    sampled_theta = np.random.beta(alpha, beta)

    chosen_arm = np.argmax(sampled_theta)
    
    reward = bandit.pull(chosen_arm)
    rewards.append(reward)
    
    if reward == 1:
        alpha[chosen_arm] += 1
    else:
        beta[chosen_arm] += 1  

print(f"Total reward: {sum(rewards)}")
print(f"Estimated means: {alpha / (alpha + beta)}")
print(f"True probabilities: {true_probs}")


