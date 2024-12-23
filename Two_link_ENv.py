import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TwoLinkArmEnv(gym.Env):
    def __init__(self):
        super(TwoLinkArmEnv, self).__init__()
        self.link1_length = 1.0  # Length of the first link
        self.link2_length = 1.0  # Length of the second link
        self.max_angle = np.pi  # Maximum angle (in radians)

        # Action space: two continuous variables for joint angles (theta1, theta2)
        self.action_space = spaces.Box(low=-self.max_angle, high=self.max_angle, shape=(2,), dtype=np.float32)

        # Observation space: two joint angles (theta1, theta2)
        self.observation_space = spaces.Box(low=-self.max_angle, high=self.max_angle, shape=(2,), dtype=np.float32)

        # Initial state: joint angles (theta1, theta2)
        self.state = np.zeros(2, dtype=np.float32)

        # Initialize the plot for visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')

    def step(self, action):
        # Apply the action (joint angles) to update the state
        self.state = np.clip(self.state + action, -self.max_angle, self.max_angle)

        # Calculate the end-effector position
        x1 = self.link1_length * np.cos(self.state[0])
        y1 = self.link1_length * np.sin(self.state[0])
        x2 = x1 + self.link2_length * np.cos(self.state[0] + self.state[1])
        y2 = y1 + self.link2_length * np.sin(self.state[0] + self.state[1])

        # Calculate the reward (e.g., distance to the target or any desired behavior)
        reward = -np.sqrt(x2**2 + y2**2)  # Negative distance to the origin as an example

        # Return the next state, reward, done flag, and additional info
        done = False  # No termination condition in this simple example
        return np.copy(self.state), reward, done, {}

    def reset(self):
        # Reset the state to the initial position
        self.state = np.zeros(2, dtype=np.float32)
        return np.copy(self.state)

    def render(self):
        # Calculate the joint positions for visualization
        x1 = self.link1_length * np.cos(self.state[0])
        y1 = self.link1_length * np.sin(self.state[0])
        x2 = x1 + self.link2_length * np.cos(self.state[0] + self.state[1])
        y2 = y1 + self.link2_length * np.sin(self.state[0] + self.state[1])

        # Plot the arm
        self.ax.clear()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')

        # Add a grid to the plot
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

        # Add x and y axes
        self.ax.axhline(0, color='black', linewidth=1)
        self.ax.axvline(0, color='black', linewidth=1)

        # Plot the first link
        self.ax.plot([0, x1], [0, y1], lw=4, color='blue')
        # Plot the second link
        self.ax.plot([x1, x2], [y1, y2], lw=4, color='green')

        # Plot the end effector (tip of the second link)
        self.ax.plot(x2, y2, 'ro')

        # Annotate the position of the end effector
        position_text = f"({x2:.2f}, {y2:.2f})"
        self.ax.text(x2, y2, position_text, fontsize=10, color='red', ha='left', va='bottom')

        # Display the plot
        plt.draw()
        plt.pause(1)

    def close(self):
        # Close the environment
        plt.close(self.fig)


# Create the environment
env = TwoLinkArmEnv()

# Test the environment
env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action
    state, reward, done, info = env.step(action)
    env.render()

env.close()
