import numpy as np
from src.rl_env import OrientationEnv

def grid_search():
    """
    Perform a grid search to find the best hyperparameters for the Q-learning agent.
    """
    # Create the environment
    env = OrientationEnv()

    # Define the hyperparameter grid
    learning_rates = [0.1, 0.2, 0.3]
    discount_factors = [0.6, 0.7, 0.8]
    exploration_rates = [0.1, 0.2, 0.3]

    best_params = {}
    best_reward = -np.inf

    # Grid search loop
    for alpha in learning_rates:
        for gamma in discount_factors:
            for epsilon in exploration_rates:
                # Create the Q-table
                q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])
                total_reward = 0
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
                        total_reward += reward

                        # Update the Q-table
                        old_value = q_table[0, action]
                        next_max = np.max(q_table[0])
                        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                        q_table[0, action] = new_value

                # Update the best params if the current params are better
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}

    print(f"Best params: {best_params}")
    return best_params

if __name__ == "__main__":
    grid_search()
