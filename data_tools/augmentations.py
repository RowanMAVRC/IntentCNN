import numpy as np


def flip_trajectories_x(data):
    """
    Flip trajectories over the middle of the image along the x-axis.

    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).

    Returns:
    - flipped_data (numpy.ndarray): Flipped trajectory data along the x-axis.
    """
    flipped_data = np.copy(data)
    flipped_data[:, :, 0] = -flipped_data[:, :, 0]  # Flip x-coordinates
    return flipped_data


def augment_trajectories(data, rotation_range=(-10, 10), translation_range=(-5, 5), noise_std=0.1):
    """
    Apply augmentations to trajectories including random rotation, random translation, and random noise injection.

    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).
    - rotation_range (tuple): Range of random rotation angles in degrees.
    - translation_range (tuple): Range of random translation in both x and y directions.
    - noise_std (float): Standard deviation of random noise to be added to each coordinate.

    Returns:
    - augmented_data (numpy.ndarray): Augmented trajectory data.
    """
    num_trajectories, num_time_steps, num_dimensions = data.shape
    augmented_data = np.copy(data)

    # Random rotation
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                 [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])

    for i in range(num_trajectories):
        trajectory = augmented_data[i, :, :2]  # Consider only x and y coordinates for rotation
        rotated_trajectory = np.dot(trajectory, rotation_matrix.T)
        augmented_data[i, :, :2] = rotated_trajectory

    # Random translation
    translation_x = np.random.uniform(translation_range[0], translation_range[1])
    translation_y = np.random.uniform(translation_range[0], translation_range[1])

    augmented_data[:, :, 0] += translation_x
    augmented_data[:, :, 1] += translation_y

    # Random noise injection
    noise = np.random.normal(loc=0, scale=noise_std, size=(num_trajectories, num_time_steps, num_dimensions))
    augmented_data += noise

    return augmented_data
