import cv2
import gymnasium as gym
import numpy as np
from src.rl_env import OrientationEnv
from src.optimize_rl_agent import grid_search
from src.data_augmentation import augment_image

def train_agent():
    """
    Train a Q-learning agent to solve the OrientationEnv.
    """
    # Get the best hyperparameters from the grid search
    best_params = grid_search()
    alpha = best_params["alpha"]
    gamma = best_params["gamma"]
    epsilon = best_params["epsilon"]

    # Load the training image
    image = cv2.imread("training_image.png")

    # Create the Q-table
    q_table = np.zeros([360, 360])  # 360 possible true angles, 360 possible actions

    # Hyperparameters
    num_episodes = 1000

    # Training loop
    for i in range(num_episodes):
        # Augment the image and get the true angle
        augmented_image, true_angle = augment_image(image)
        # Create the environment
        env = OrientationEnv(augmented_image, true_angle)
        state, _ = env.reset()
        done = False
        while not done:
            # Choose an action
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[int(true_angle)])  # Exploit

            # Take the action
            next_state, reward, done, _, _ = env.step(action)

            # Update the Q-table
            old_value = q_table[int(true_angle), action]
            next_max = np.max(q_table[int(true_angle)])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[int(true_angle), action] = new_value

    # Save the trained Q-table
    np.save("q_table.npy", q_table)

if __name__ == "__main__":
    train_agent()
