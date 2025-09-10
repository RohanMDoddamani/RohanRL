import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 0.01
T = 2
N = int(T / dt)
t = np.linspace(0, T, N+1)

# Initial conditions: small perturbations
x0_values = [0.1, -0.1]

plt.figure(figsize=(8,5))

for x0 in x0_values:
    x = np.zeros(N+1)
    x[0] = x0
    for i in range(N):
        x[i+1] = x[i] + dt * (2 * x[i])  # dx/dt = 2x
    plt.plot(t, x, label=f'Perturbation x0={x0}')

# Plot zero line for reference
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.title('Unstable Equilibrium at x=0: Small Perturbations Grow')
plt.xlabel('Time t')
plt.ylabel('x(t)')
plt.legend()
plt.show()
