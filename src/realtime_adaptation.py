import numpy as np

def update_ensemble_weights(weights, predicted_angles, corrected_angle):
    """
    Updates the weights of the ensemble model based on user feedback.
    """
    # Calculate the errors of the individual models
    errors = np.abs(predicted_angles - corrected_angle)

    # Update the weights based on the errors. The smaller the error, the higher the weight.
    new_weights = 1 / (errors + 1e-6)  # Add a small epsilon to avoid division by zero
    new_weights /= np.sum(new_weights)  # Normalize the weights to sum to 1

    return new_weights
