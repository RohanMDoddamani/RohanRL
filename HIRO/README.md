# 🧠 HIRO — Hierarchical Reinforcement Learning with Off-Policy Correction

The complex tasks (like navigation, robotics, or language planning), learning directly at the primitive action level is inefficient.

* Hierarchical RL (HRL) decomposes control into multiple levels of abstraction:
* High-level policy (manager / meta-controller) → chooses subgoals or options.

Low-level policies (workers) → achieve those subgoals using primitive actions.
HIRO (Nachum et al., ICML 2018) is a two-level Hierarchical Reinforcement Learning algorithm designed for continuous control tasks.
It introduces a manager–worker architecture, where the high-level policy sets subgoals and the low-level policy executes primitive actions to achieve them.
HIRO achieves sample-efficient and stable learning through off-policy correction.

# 🚀 Overview

HIRO stands for Hierarchical Reinforcement Learning with Off-Policy Correction.
It decomposes a complex task into two layers:

- Manager (High-level policy): sets continuous subgoals in the state space.
- Worker (Low-level policy): executes actions to achieve the current subgoal using intrinsic rewards.

This architecture allows efficient learning in long-horizon environments where flat RL struggles.

# 🏗️ Architecture
        ┌──────────────────────────────────────────────┐
        │                  Manager                     │
        │  High-Level Policy πᴴ(g|s)                   │
        │  - Chooses subgoal g every c steps            │
        │  - Optimizes extrinsic (environment) reward   │
        └──────────────────────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────┐
        │                  Worker                      │
        │  Low-Level Policy πᴸ(a|s,g)                  │
        │  - Executes primitive actions                 │
        │  - Optimizes intrinsic reward:               │
        │      rᶦ = -‖(sₜ₊₁ - sₜ) - g‖₂                 │
        └──────────────────────────────────────────────┘
                             │
                             ▼
                     Environment Dynamics


* The manager (high-level) sets subgoals every c steps (say every 10 steps).
* The worker (low-level) acts for c steps, trying to reach that subgoal.

