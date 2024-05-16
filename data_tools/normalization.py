import numpy as np


def mean_removed_all(data):
    """
    Remove the mean of each dimension across all trajectories.

    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).

    Returns:
    - mean_removed_data (numpy.ndarray): Trajectory data with mean removed along each dimension.
    """
    mean_removed_data = data.copy()
    
    # Initialize lists for each dimension
    dimension_values = [[] for _ in range(data.shape[2])]
    
    # Extract values from all trajectories for each dimension
    for trajectory in mean_removed_data:
        for point in trajectory:
            for i, value in enumerate(point):
                dimension_values[i].append(value)
    
    # Calculate means for each dimension
    dimension_means = [np.mean(values) for values in dimension_values]
    
    # Remove means from all points
    for trajectory in mean_removed_data:
        for point in trajectory:
            for i, mean in enumerate(dimension_means):
                point[i] -= mean
    
    return mean_removed_data

def mean_removed_single(data):
    """
    Remove the mean of each dimension separately for each trajectory.

    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).

    Returns:
    - mean_removed_data (numpy.ndarray): Trajectory data with mean removed along each dimension separately for each trajectory.
    """
    mean_removed_data = data.copy()
    
    for trajectory in mean_removed_data:
        dimension_values = [[] for _ in range(data.shape[2])]
        
        # Extract values from the current trajectory for each dimension
        for point in trajectory:
            for i, value in enumerate(point):
                dimension_values[i].append(value)
        
        # Calculate means for each dimension
        dimension_means = [np.mean(values) for values in dimension_values]
        
        # Remove means from all points in the current trajectory
        for point in trajectory:
            for i, mean in enumerate(dimension_means):
                point[i] -= mean
    
    return mean_removed_data

def z_score_standardization_single(data):
    """
    Perform Z-score standardization on the given trajectory data separately for each trajectory.

    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).

    Returns:
    - standardized_data (numpy.ndarray): Standardized trajectory data with the same shape as the input.
    """
    standardized_data = data.copy()

    # Iterate over each trajectory
    for trajectory in standardized_data:
        num_dimensions = trajectory.shape[1]

        # Extract coordinates (x, y, z) for standardization if available
        coordinates = trajectory[:, :3] if num_dimensions >= 3 else trajectory[:, :2]

        # Compute mean and standard deviation along each axis
        mean = np.mean(coordinates, axis=0)  # Compute mean for the current trajectory
        std_dev = np.std(coordinates, axis=0)  # Compute standard deviation for the current trajectory
        
        # Check for zero standard deviations to avoid division by zero
        std_dev[std_dev == 0] = 1  # Replace zero standard deviations with 1

        # Perform z-score standardization for the current trajectory
        standardized_coordinates = (coordinates - mean) / std_dev

        # Replace the standardized coordinates in the current trajectory
        if num_dimensions >= 3:
            trajectory[:, :3] = standardized_coordinates
        else:
            trajectory[:, :2] = standardized_coordinates

    return standardized_data

def z_score_standardization_all(data):
    """
    Perform Z-score standardization on the given trajectory data.

    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).

    Returns:
    - standardized_data (numpy.ndarray): Standardized trajectory data with the same shape as the input.
    """
    num_dimensions = data.shape[2]
    # Extract coordinates (x, y, z) for standardization
    coordinates = data[:, :, :3] if num_dimensions >= 3 else data[:, :, :2]  # Extract the first 3 dimensions (x, y, z) if available

    # Compute mean and standard deviation along each axis
    mean = np.mean(coordinates, axis=(0, 1))  # Compute mean across all trajectories and time steps
    std_dev = np.std(coordinates, axis=(0, 1))  # Compute standard deviation across all trajectories and time steps
    
    # Check for zero standard deviations to avoid division by zero
    std_dev[std_dev == 0] = 1  # Replace zero standard deviations with 1

    # Perform z-score standardization
    standardized_coordinates = (coordinates - mean) / std_dev

    # Replace the standardized coordinates in the original data
    standardized_data = np.copy(data)
    if num_dimensions >= 3:
        standardized_data[:, :, :3] = standardized_coordinates
    else:
        standardized_data[:, :, :2] = standardized_coordinates

    return standardized_data

def min_max_scaling(data):
    """
    Perform min-max scaling on the given trajectory data.

    Parameters:
    - data (numpy.ndarray): Input trajectory data with shape (num_trajectories, num_time_steps, num_dimensions).

    Returns:
    - scaled_data (numpy.ndarray): Scaled trajectory data with the same shape as the input.
    """
    # Extract coordinates (x, y, z) for scaling
    coordinates = data[:, :, :3]  # Extract the first 3 dimensions (x, y, z)

    # Compute min and max values along each axis
    min_val = np.min(coordinates, axis=(0, 1))  # Compute minimum across all trajectories and time steps
    max_val = np.max(coordinates, axis=(0, 1))  # Compute maximum across all trajectories and time steps
    
    # Compute range of values along each axis
    range_val = max_val - min_val
    
    # Check for zero range values to avoid division by zero
    range_val[range_val == 0] = 1  # Replace zero range values with 1

    # Perform min-max scaling
    scaled_coordinates = (coordinates - min_val) / range_val

    # Replace the scaled coordinates in the original data
    scaled_data = np.copy(data)
    scaled_data[:, :, :3] = scaled_coordinates

    return scaled_data

def compute_trajectory_stats(trajectories):
    num_trajectories = len(trajectories)
    trajectory_lengths = np.array([len(trajectory) for trajectory in trajectories])
    total_length = np.sum(trajectory_lengths)
    min_x, min_y, min_z = np.min(trajectories[:, :, 0]), np.min(trajectories[:, :, 1]), np.min(trajectories[:, :, 2])
    max_x, max_y, max_z = np.max(trajectories[:, :, 0]), np.max(trajectories[:, :, 1]), np.max(trajectories[:, :, 2])
    avg_x, avg_y, avg_z = np.mean(trajectories[:, :, 0]), np.mean(trajectories[:, :, 1]), np.mean(trajectories[:, :, 2])
    
    return {
        'Num Trajectories': num_trajectories,
        'Total Length': total_length,
        'Min X': min_x,
        'Min Y': min_y,
        'Min Z': min_z,
        'Max X': max_x,
        'Max Y': max_y,
        'Max Z': max_z,
        'Average X': avg_x,
        'Average Y': avg_y,
        'Average Z': avg_z,
        'Min Length': np.min(trajectory_lengths),
        'Max Length': np.max(trajectory_lengths),
        'Average Length': np.mean(trajectory_lengths),
    }
    
def normalize(data):
    """
    Normalize the input data along each coordinate axis.

    Args:
    data (np.ndarray): Input data with shape (num_samples, num_timesteps, num_dimensions).

    Returns:
    np.ndarray: Normalized data with the same shape as the input data.
    """
    # Determine the maximum values for each coordinate axis
    max_values = np.max(data, axis=(0, 1))
    
    # Normalize each coordinate axis separately
    normalized_data = data / max_values
    
    return normalized_data
    

if __name__ == "__main__":
    # Example usage:
    # Suppose you have 1000 trajectories, each with 50 data points, and each data point containing x, y, z coordinates
    # Here's how you can create a sample dataset for testing
    num_trajectories = 1000
    num_points_per_trajectory = 50
    data = np.random.rand(num_trajectories, num_points_per_trajectory, 3)

    result = mean_removed_all(data)
    print("Original data shape:", data.shape)
    print("Mean-removed data shape:", result.shape)
    result = mean_removed_single(data)
    print("Original data shape:", data.shape)
    print("Mean-removed data shape:", result.shape)
    
    # Example usage:
    data = np.array([[[100, 200, 300], [150, 250, 350]], [[50, 100, 150], [75, 125, 175]]])
    normalized_data, max_values = normalize(data)
    print("Normalized Data:\n", normalized_data)
    print("Max Values (x, y, z):", max_values)
