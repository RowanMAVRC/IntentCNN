"""
 _____       _             _                           
|_   _|     | |           | |    ____  _   _  _   _                           
  | |  _ __ | |_ ___ _ __ | |_  / ___|| \ | || \ | |
  | | | '_ \| __/ _ \ '_ \| __|| |   ||  \| ||  \| |
 _| |_| | | | ||  __/ | | | |_ | |___|| |\  || |\  |
|_____|_| |_|\__\___|_| |_|\__| \____||_| \_||_| \_|
"""
# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

import numpy as np
from multipledispatch import dispatch

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

@dispatch(np.ndarray)
def flip_trajectories_x(data):
    """
    Flip trajectories over the middle of the image along the x-axis.
    
    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).
    
    Returns:
    - flipped_data (numpy.ndarray): Flipped trajectory data along the x-axis.
    """
    # Copy the original data to avoid modifying the input
    flipped_data = np.copy(data)
    # Flip the x-coordinates (assumed to be the first dimension in the trajectory data)
    flipped_data[:, :, 0] = -flipped_data[:, :, 0]  
    return flipped_data


@dispatch(np.ndarray, list, list)
def flip_trajectories_x(data, labels, classes=None):
    """
    Flip trajectories over the middle of the image along the x-axis for specified classes.
    
    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).
    - labels (list): List of class labels corresponding to each trajectory in data.
    - classes (list): List of classes to be flipped. If None, flip all trajectories.
    
    Returns:
    - flipped_data (numpy.ndarray): Flipped trajectory data along the x-axis for selected classes.
    - new_labels (list): Updated list of class labels corresponding to flipped trajectories.
    """
    # If no specific classes are provided, flip all trajectories
    if classes is None:
        return flip_trajectories_x(data), labels

    # Create a mask for trajectories that belong to the specified classes
    mask = np.isin(labels, classes)
    
    # Flip the x-axis of trajectories that match the class criteria
    flipped_data = data.copy()
    flipped_data[mask, :, 0] = -flipped_data[mask, :, 0]
    
    # Update labels for flipped trajectories
    new_labels = [label for label, m in zip(labels, mask) if m]
    
    return flipped_data, new_labels


def augment_trajectories(data, rotation_range=(-10, 10), translation_range=(-5, 5), noise_std=0.1):
    """
    Apply augmentations to trajectories including random rotation, random translation, and random noise injection.
    
    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).
    - rotation_range (tuple): Range of rotation angles in degrees to apply to the trajectories.
    - translation_range (tuple): Range of translation values for x and y coordinates.
    - noise_std (float): Standard deviation of Gaussian noise to be added to each data point.
    
    Returns:
    - augmented_data (numpy.ndarray): Augmented trajectory data after applying rotation, translation, and noise.
    """
    num_trajectories, num_time_steps, num_dimensions = data.shape
    augmented_data = np.copy(data)

    # Apply random rotations (only applied to the first two dimensions x and y)
    rotation_angles = np.random.uniform(rotation_range[0], rotation_range[1], size=num_trajectories)
    rotation_matrices = np.array([[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]] for angle in np.radians(rotation_angles)])
    augmented_data[:, :, :2] = np.einsum('ijk,ikl->ijl', augmented_data[:, :, :2], rotation_matrices)

    # Apply random translations (translations applied to all dimensions)
    translations = np.random.uniform(translation_range[0], translation_range[1], size=(num_trajectories, num_dimensions))
    augmented_data += translations[:, np.newaxis, :]

    # Add Gaussian noise to the entire trajectory
    noise = np.random.normal(loc=0, scale=noise_std, size=(num_trajectories, num_time_steps, num_dimensions))
    augmented_data += noise

    return augmented_data


def augment_with_jitters(trajectory, jitter_magnitude=0.1, num_augmentations=1):
    """
    Apply random jitters (Gaussian noise) to a given trajectory.
    
    Parameters:
    - trajectory (numpy.ndarray): Input trajectory with shape (num_time_steps, num_dimensions).
    - jitter_magnitude (float): Standard deviation of the jitter noise to apply to the trajectory.
    - num_augmentations (int): Number of augmented trajectories to generate with jitter.
    
    Returns:
    - augmented_trajectories (list of numpy.ndarray): List of augmented trajectories with applied jitter.
    """
    # Create multiple jittered trajectories
    return [trajectory + np.random.normal(loc=0, scale=jitter_magnitude, size=trajectory.shape) for _ in range(num_augmentations)]

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Example usage:
    # Generate random data representing 1000 trajectories, each with 50 data points (x, y, z)
    num_trajectories = 1000
    num_points_per_trajectory = 50
    data = np.random.rand(num_trajectories, num_points_per_trajectory, 3)

    # Flip the trajectories along the x-axis
    result = flip_trajectories_x(data)
    print("Original data shape:", data.shape)
    print("Flipped data shape:", result.shape)

    # Example usage of flipping trajectories with labels and classes
    data = np.array([
        [[1, 2], [3, 4], [5, 6]],   # Example trajectory 1
        [[-1, -2], [-3, -4], [-5, -6]],   # Example trajectory 2
        [[10, 20], [30, 40], [50, 60]]   # Example trajectory 3
    ])
    labels = ['A', 'B', 'A']
    classes = ['A']

    # Flip trajectories only for class 'A'
    flipped_data, new_labels = flip_trajectories_x(data, labels, classes)

    # Print results
    print("Original Data:")
    print(data)
    print("Original Labels:")
    print(labels)
    print("Flipped Data:")
    print(flipped_data)
    print("New Labels:")
    print(new_labels)

    # Example usage of augment_with_jitters:
    original_trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    augmented_trajectories = augment_with_jitters(original_trajectory, jitter_magnitude=0.1, num_augmentations=3)

    # Print the augmented trajectories
    for i, trajectory in enumerate(augmented_trajectories):
        print(f"Augmented Trajectory {i + 1}:\n", trajectory)
