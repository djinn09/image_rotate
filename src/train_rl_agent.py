import gymnasium as gym
import numpy as np
from src.rl_env import OrientationEnv

def train_agent():
    """
    Train a Q-learning agent to solve the OrientationEnv.
    """
    # Create the environment
    env = OrientationEnv()

    # Create the Q-table
    q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])

    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.6  # Discount factor
    epsilon = 0.1  # Exploration rate
    num_episodes = 1000

    # Training loop
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # Choose an action
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[0])  # Exploit

            # Take the action
            next_state, reward, done, _, _ = env.step(action)

            # Update the Q-table
            old_value = q_table[0, action]
            next_max = np.max(q_table[0])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[0, action] = new_value

    # Save the trained Q-table
    np.save("q_table.npy", q_table)

if __name__ == "__main__":
    train_agent()
