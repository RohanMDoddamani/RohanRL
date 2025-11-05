import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras import layers


# Hyperparameters
c = 10
total_episodes = 1000
max_steps = 200
gamma = 0.7
lr = 0.01

env = gym.make("Pendulum-v1")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# ( obs + goal)
def build_policy():
    model = tf.keras.Sequential([
        layers.Input(shape=(obs_dim*2,)),
        layers.Dense(128, activation='relu'),
        # layers.Dense(128, activation='relu'),
        layers.Dense(act_dim, activation='tanh')])
    return model

#  ( obs, output: goal vector)
def build_manager():
    model = tf.keras.Sequential([
        layers.Input(shape=(obs_dim,)),
        layers.Dense(128, activation='relu'),
        # layers.Dense(128, activation='relu'),
        layers.Dense(obs_dim, activation='tanh')])  
    return model

worker_policy = build_policy()
manager_policy = build_manager()

worker_optimizer = tf.keras.optimizers.Adam(lr)
manager_optimizer = tf.keras.optimizers.Adam(lr)

def discount_rewards(r, gamma=0.99):
    discounted = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted[t] = running_add
    return discounted

for episode in range(1):
    obs = env.reset()
    obs  = env.step(env.action_space.sample())
    episode_worker_states = []
    episode_worker_actions = []
    episode_worker_rewards = []
    episode_manager_states = []
    episode_manager_goals = []
    episode_manager_rewards = []

    goal = manager_policy(obs[0].reshape((1,3))).numpy()[0] 
  
    total_reward = 0

 
    for step in range(100):
      

        worker_input = np.concatenate((obs[0], goal))

      
        action = worker_policy(tf.convert_to_tensor([worker_input]))
      

        next_obs, ext_reward, done, _ ,info = env.step(action)
    

     
        intrinsic_reward = -np.linalg.norm((obs[0] + goal) - next_obs[0])

        # Save worker experience
        episode_worker_states.append(tf.convert_to_tensor([worker_input]))
        episode_worker_actions.append(action)
        episode_worker_rewards.append(intrinsic_reward)

        total_reward += ext_reward

        # Manager chooses new goal every c steps
        if (step + 1) % c == 0:
            episode_manager_states.append(obs[0].reshape((1,3)))
            episode_manager_goals.append(goal)
            episode_manager_rewards.append(total_reward)
            print(total_reward)
            total_reward = 0
            goal = manager_policy(np.expand_dims(next_obs, 0)).numpy()[0]

        obs = next_obs.reshape((1,3))
        # print(obs)
        if done:
            break

    #  (discounted rewards) for worker and manager
    discounted_worker_rewards = discount_rewards(episode_worker_rewards, gamma)
    discounted_manager_rewards = discount_rewards(episode_manager_rewards, gamma)

    # Update worker ( REINFORCE )
    with tf.GradientTape() as tape:
        loss = 0
        for s, a, r in zip(episode_worker_states, episode_worker_actions, discounted_worker_rewards):
  
            pi = worker_policy(s)
            log_prob = -tf.reduce_sum((pi - a)**2)  # simple gaussian log_prob proxy
            loss += -log_prob * r

    grads = tape.gradient(loss, worker_policy.trainable_variables)
    worker_optimizer.apply_gradients(zip(grads, worker_policy.trainable_variables))

   # Manager worker ( REINFORCE )
    with tf.GradientTape() as tape:
        loss = 0
        for s, g, r in zip(episode_manager_states, episode_manager_goals, discounted_manager_rewards):

            pi = manager_policy(s)
            log_prob = -tf.reduce_sum((pi - g)**2)
            loss += -log_prob * r
    
    gradsm = tape.gradient(loss, manager_policy.trainable_variables)
    manager_optimizer.apply_gradients(zip(gradsm, manager_policy.trainable_variables))
    

    print(f"Episode {episode+1} completed")
    if episode % 10 == 0:
        print(f"Episode {episode}, Manager Total reward: {(discounted_manager_rewards[-1])}")
        print(f"Episode {episode}, Worker Total reward: {(discounted_worker_rewards[-1])}")
        print(episode_manager_rewards)




