import numpy as np


def calculate_accuracy(predicted_angle, ground_truth_angle):
    """
    Calculates the accuracy of the predicted angle compared to the ground truth angle.
    """
    # A simple accuracy metric: 1 if the predicted angle is within a certain tolerance of the ground truth angle, 0 otherwise.
    tolerance = 1.0  # degrees
    if abs(predicted_angle - ground_truth_angle) <= tolerance:
        return 1
    else:
        return 0


def calculate_mae(predicted_angles, ground_truth_angles):
    """
    Calculates the Mean Absolute Error between the predicted and ground truth angles.
    """
    return np.mean(np.abs(predicted_angles - ground_truth_angles))


def calculate_rmse(predicted_angles, ground_truth_angles):
    """
    Calculates the Root Mean Squared Error between the predicted and ground truth angles.
    """
    return np.sqrt(np.mean((predicted_angles - ground_truth_angles) ** 2))
