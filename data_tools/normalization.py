import numpy as np


def z_score_2d(data):
    """
    Perform Z-score standardization on the given trajectory data.

    Parameters:
    - data (numpy.ndarray): The input trajectory data to be standardized,
                            with shape (num_trajectories, num_time_steps, num_dimensions).

    Returns:
    - standardized_data (numpy.ndarray): The standardized trajectory data, with the same shape as the input.

    Z-score standardization (also known as Z-score normalization) scales the data so that it has a mean of 0
    and a standard deviation of 1 along each axis. This normalization technique is useful for ensuring
    that different features or datasets are comparable and can improve the convergence of machine learning
    algorithms.
    """
    # Extract coordinates (x, y) for standardization
    coordinates = data[:, :, :2]  # Extract the first 2 dimensions (x, y)

    # Compute mean and standard deviation along each axis
    mean = np.mean(coordinates, axis=(0, 1))  # Compute mean across all trajectories and time steps
    std_dev = np.std(coordinates, axis=(0, 1))  # Compute standard deviation across all trajectories and time steps
    
    # Check for zero standard deviations to avoid division by zero
    std_dev[std_dev == 0] = 1  # Replace zero standard deviations with 1

    # Perform z-score standardization
    standardized_coordinates = (coordinates - mean) / std_dev

    # Replace the standardized coordinates in the original data
    standardized_data = np.copy(data)
    standardized_data[:, :, :2] = standardized_coordinates

    return standardized_data

def z_score_standardization(data):
    """
    Perform Z-score standardization on the given trajectory data.

    Parameters:
    - data (numpy.ndarray): The input trajectory data to be standardized,
                            with shape (num_trajectories, num_time_steps, num_dimensions).

    Returns:
    - standardized_data (numpy.ndarray): The standardized trajectory data, with the same shape as the input.

    Z-score standardization (also known as Z-score normalization) scales the data so that it has a mean of 0
    and a standard deviation of 1 along each axis. This normalization technique is useful for ensuring
    that different features or datasets are comparable and can improve the convergence of machine learning
    algorithms.
    """
    # Extract coordinates (x, y, z) for standardization
    coordinates = data[:, :, :3]  # Extract the first 3 dimensions (x, y, z)

    # Compute mean and standard deviation along each axis
    mean = np.mean(coordinates, axis=(0, 1))  # Compute mean across all trajectories and time steps
    std_dev = np.std(coordinates, axis=(0, 1))  # Compute standard deviation across all trajectories and time steps
    
    # Check for zero standard deviations to avoid division by zero
    std_dev[std_dev == 0] = 1  # Replace zero standard deviations with 1

    # Perform z-score standardization
    standardized_coordinates = (coordinates - mean) / std_dev

    # Replace the standardized coordinates in the original data
    standardized_data = np.copy(data)
    standardized_data[:, :, :3] = standardized_coordinates

    return standardized_data

def min_max_scaling(data):
    """
    Perform min-max scaling on the given trajectory data.

    Parameters:
    - data (numpy.ndarray): The input trajectory data to be scaled,
                            with shape (num_trajectories, num_time_steps, num_dimensions).

    Returns:
    - scaled_data (numpy.ndarray): The scaled trajectory data, with the same shape as the input.

    Min-max scaling (also known as min-max normalization) scales the data to a fixed range (e.g., [0, 1])
    by subtracting the minimum value and dividing by the range of the data along each axis.
    This normalization technique preserves the original distribution of the data and can be useful when the
    input features have similar scales.
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