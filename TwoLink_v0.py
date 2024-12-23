import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TwoLinkManipulatorEnv(gym.Env):
    """
    Custom Environment for simulating a two-link manipulator with discrete actions
    using Gymnasium API.
    """

    def __init__(self):
        super(TwoLinkManipulatorEnv, self).__init__()
        
        # Action space: 9 discrete actions for joint angle control
        self.action_space = spaces.Discrete(9)
        
        # Observation space: Joint angles [theta1, theta2], clipped between [-pi, pi]
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32)
        
        # Step size for joint angles
        self.theta_step = np.pi / 18  # 10 degrees increment
        
        # Actions mapping: List of joint changes corresponding to action space
        self.actions = [
            [self.theta_step, self.theta_step],    # [+theta1, +theta2]
            [-self.theta_step, self.theta_step],   # [-theta1, +theta2]
            [-self.theta_step, -self.theta_step],  # [-theta1, -theta2]
            [self.theta_step, -self.theta_step],   # [+theta1, -theta2]
            [0, self.theta_step],                 # [0, +theta2]
            [0, -self.theta_step],                # [0, -theta2]
            [self.theta_step, 0],                 # [+theta1, 0]
            [-self.theta_step, 0],                # [-theta1, 0]
            [0, 0]                               # [0, 0]
        ]
        
        # Initialize state: [theta1, theta2]
        self.state = np.array([0.0, 0.0])
        self.goal = np.array([np.pi / 4, np.pi / 4])  # Predefined goal position
        self.max_steps = 200  # Limit episode length
        self.current_step = 0

    def reset(self, seed=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0])  # Reset joint angles to 0
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        """Performs one step in the environment based on the action."""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Retrieve joint angle increments for the given action
        delta_theta1, delta_theta2 = self.actions[action]
        
        # Update joint angles and clip them within [-pi, pi]
        self.state[0] = np.clip(self.state[0] + delta_theta1, -np.pi, np.pi)
        self.state[1] = np.clip(self.state[1] + delta_theta2, -np.pi, np.pi)

        # Increment step count
        self.current_step += 1
        
        # Reward based on distance to goal
        distance_to_goal = np.linalg.norm(self.state - self.goal)
        reward = -distance_to_goal  # Negative reward for being far from goal

        # Termination condition: reaching goal or max steps
        done = distance_to_goal < 0.1 or self.current_step >= self.max_steps
        
        return self.state, reward, done, False, {}

    def render(self, mode='human'):
        """Renders the current state of the environment."""
        print(f"Joint Angles: theta1 = {self.state[0]:.2f}, theta2 = {self.state[1]:.2f}, Goal = {self.goal}")
        
    def close(self):
        """Cleans up resources, if any."""
        pass

if __name__ == "__main__":
    # Assignment Instructions
    print("Assignment: Train a Deep Q-Network (DQN) to control a two-link robot to reach a predefined goal using TensorFlow.")
    print("1. Use the custom environment 'TwoLinkManipulatorEnv'.")
    print("2. Implement a DQN algorithm using TensorFlow/Keras.")
    print("3. The robot should move its two joints to reach the predefined goal [x, y] = [xg, yg].")
    print("4. Evaluate the trained DQN model by testing the robot's ability to reach the goal in the minimum number of steps.")
    print("5. Document the results and provide a short explanation of your implementation, training process, and observations.")
    
    # Example Environment Execution
    env = TwoLinkManipulatorEnv()
    obs, _ = env.reset()
    done = False
    
    print("Starting simulation...")
    while not done:
        action = env.action_space.sample()  # Random action for demonstration
        obs, reward, done, _, _ = env.step(action)
        env.render()
    print("Simulation finished. Train your DQN model as per the assignment instructions.")
