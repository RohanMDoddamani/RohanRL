# 🧭 Solving POMDPs using Bayesian Belief Updates

This repository implements POMDP (Partially Observable Markov Decision Process) solving using a Bayesian belief update approach.
Instead of assuming full knowledge of the environment’s state, the agent maintains a belief distribution over possible states, updated probabilistically using Bayes’ theorem after every observation.

This allows the agent to act optimally under uncertainty, where observations are noisy or incomplete.

# 🌍 Overview

POMDPs (Partially Observable Markov Decision Processes) extend traditional MDPs to cases where the agent cannot directly observe the underlying state.
Instead, the agent receives observations that provide partial information about the true state.

To make decisions, the agent maintains a belief state — a probability distribution over all possible world states — and updates it using Bayes’ theorem whenever it receives new evidence.


