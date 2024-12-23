import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque

import sys
import os

sys.path.append(os.path.abspath("TwoLink_v0.py"))
from two_link_manipulator_env import TwoLinkManipulatorEnv



# Neural Network for Q-value approximation
def build_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def store(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# Hyperparameters
EPISODES = 500
MAX_STEPS = 200
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10

# Initialize environment, model, and replay buffer
env = TwoLinkManipulatorEnv()
state_shape = (env.observation_space.shape[0],)
action_size = env.action_space.n

main_model = build_model(state_shape, action_size)
target_model = build_model(state_shape, action_size)
target_model.set_weights(main_model.get_weights())  # Synchronize weights
replay_buffer = ReplayBuffer(max_size=10000)

epsilon = EPSILON_START

# Training loop
for episode in range(EPISODES):
    state, _ = env.reset()
    state = np.expand_dims(state, axis=0)
    total_reward = 0
    done = False

    for step in range(MAX_STEPS):
        # Select action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = main_model.predict(state, verbose=0)
            action = np.argmax(q_values)

        # Take action
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        total_reward += reward

        # Store transition
        replay_buffer.store((state, action, reward, next_state, done))
        state = next_state

        # Train model
        if replay_buffer.size() > BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = np.vstack(states)
            next_states = np.vstack(next_states)
            
            # Q-learning update
            target_q = main_model.predict(states, verbose=0)
            next_q = target_model.predict(next_states, verbose=0)
            for i in range(BATCH_SIZE):
                target_q[i, actions[i]] = rewards[i] + (1 - dones[i]) * GAMMA * np.max(next_q[i])

            main_model.train_on_batch(states, target_q)
        
        if done:
            break

    # Update target network
    if episode % TARGET_UPDATE_FREQ == 0:
        target_model.set_weights(main_model.get_weights())
    
    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    print(f"Episode {episode+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

print("Training complete. Evaluate the trained model.")
