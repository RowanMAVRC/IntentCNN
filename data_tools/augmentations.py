import numpy as np
from multipledispatch import dispatch


@dispatch(np.ndarray)
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

@dispatch(np.ndarray, list, list)
def flip_trajectories_x(data, labels, classes=None):
    """
    Flip trajectories over the middle of the image along the x-axis.

    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).
    - labels (list): List of class labels corresponding to each trajectory in data.
    - classes (list): List of classes to be flipped. If None, flip all trajectories.

    Returns:
    - flipped_data (numpy.ndarray): Flipped trajectory data along the x-axis.
    - new_labels (list): Updated list of class labels corresponding to flipped trajectories.
    """
    if not classes:
        return flip_trajectories_x(data), labels
    flipped_data = []
    new_labels = []

    for i in range(len(data)):
        if not classes or (labels[i] in classes):
            flipped_traj = np.copy(data[i])
            flipped_traj[:, 0] = -flipped_traj[:, 0]
            flipped_data.append(flipped_traj)
            if labels:
                new_labels.append(labels[i])

    return np.array(flipped_data), new_labels

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

def augment_with_jitters(trajectory, jitter_magnitude=0.1, num_augmentations=1):
    augmented_trajectories = []
    for _ in range(num_augmentations):
        # Generate jittered trajectory
        jittered_trajectory = trajectory + np.random.normal(loc=0, scale=jitter_magnitude, size=trajectory.shape)
        augmented_trajectories.append(jittered_trajectory)
    return augmented_trajectories


if __name__ == "__main__":
    # Example usage:
    # Suppose you have 1000 trajectories, each with 50 data points, and each data point containing x, y, z coordinates
    # Here's how you can create a sample dataset for testing
    num_trajectories = 1000
    num_points_per_trajectory = 50
    data = np.random.rand(num_trajectories, num_points_per_trajectory, 3)

    result = flip_trajectories_x(data)
    print("Original data shape:", data.shape)
    print("Mean-removed data shape:", result.shape)
    
     # Test data
    data = np.array([
        [[1, 2], [3, 4], [5, 6]],
        [[-1, -2], [-3, -4], [-5, -6]],
        [[10, 20], [30, 40], [50, 60]]
    ])
    labels = ['A', 'B', 'A']
    classes = ['A']

    # Call the function
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
    
    # Example usage:
    original_trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    augmented_trajectories = augment_with_jitters(original_trajectory, jitter_magnitude=0.1, num_augmentations=3)

    for i, trajectory in enumerate(augmented_trajectories):
        print(f"Augmented Trajectory {i + 1}:\n", trajectory)