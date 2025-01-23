import os
import cv2
import random
import pandas as pd
import numpy as np

# Dictionary to store the trajectory points for each object_id
object_trajectories = {}
# Dictionary to store the last frame number each object was seen
object_last_seen = {}

# Function to calculate normalized depth
def calculate_normalized_depth(depth, max_depth):
    """Calculate normalized depth given the current depth and maximum depth."""
    return depth / max_depth


# Function to calculate normalized displacement
def calculate_normalized_displacement(trajectory, frame_width, frame_height):
    """Calculate normalized displacement given a trajectory and frame dimensions."""
    if len(trajectory) < 2:
        return 0.0
    displacement_x = trajectory[-1][0] - trajectory[0][0]
    displacement_y = trajectory[-1][1] - trajectory[0][1]
    xy_displacement = np.sqrt(displacement_x**2 + displacement_y**2)
    diagonal = np.sqrt(frame_width**2 + frame_height**2)
    return xy_displacement / diagonal


# Function to calculate the probability of interaction
def calculate_probability_of_interaction(intention, trajectory, frame_width, frame_height, depth, max_depth):
    normalized_depth = calculate_normalized_depth(depth, max_depth)
    
    if intention == "Recon":
        return 1.0
    elif intention in ["Area Denial", "Travel"]:
        return max(0.01, 1 - normalized_depth)
    elif intention == "Kamikaze":
        normalized_displacement = calculate_normalized_displacement(trajectory, frame_width, frame_height)
        return max(0.01, (1 - normalized_displacement) * (1 - normalized_depth))
    else:
        return 0.01


def draw_bounding_box(frame, coords, label, intention, object_id, color, frame_height, normalized_depth, normalized_displacement, severity, prob_of_interaction, normalized_threat_level, current_frame_num):
    cx, cy, w, h = coords
    cx = int(cx)
    cy = frame_height - int(cy)  # Flip the y coordinate
    w = int(w)
    h = int(h)

    # Convert center-based coordinates (cx, cy) to top-left corner coordinates (x, y)
    x = cx - w // 2
    y = cy - h // 2

    # Calculate the four corners of the bounding box
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_left = (x, y + h)
    bottom_right = (x + w, y + h)

    # Line thickness settings
    edge_thickness = 2
    mid_thickness = 1

    # Draw the edges (thicker lines)
    cv2.line(frame, top_left, (top_left[0] + int(w * 0.2), top_left[1]), color, edge_thickness)
    cv2.line(frame, top_left, (top_left[0], top_left[1] + int(h * 0.2)), color, edge_thickness)
    cv2.line(frame, top_right, (top_right[0] - int(w * 0.2), top_right[1]), color, edge_thickness)
    cv2.line(frame, top_right, (top_right[0], top_right[1] + int(h * 0.2)), color, edge_thickness)
    cv2.line(frame, bottom_left, (bottom_left[0] + int(w * 0.2), bottom_left[1]), color, edge_thickness)
    cv2.line(frame, bottom_left, (bottom_left[0], bottom_left[1] - int(h * 0.2)), color, edge_thickness)
    cv2.line(frame, bottom_right, (bottom_right[0] - int(w * 0.2), bottom_right[1]), color, edge_thickness)
    cv2.line(frame, (bottom_right[0], bottom_right[1] - int(h * 0.2)), bottom_right, color, edge_thickness)

    # Draw the middles (thinner lines)
    cv2.line(frame, (top_left[0] + int(w * 0.2), top_left[1]), (top_right[0] - int(w * 0.2), top_right[1]), color, mid_thickness)
    cv2.line(frame, (bottom_left[0] + int(w * 0.2), bottom_left[1]), (bottom_right[0] - int(w * 0.2), bottom_right[1]), color, mid_thickness)
    cv2.line(frame, (top_left[0], top_left[1] + int(h * 0.2)), (bottom_left[0], bottom_left[1] - int(h * 0.2)), color, mid_thickness)
    cv2.line(frame, (top_right[0], top_right[1] + int(h * 0.2)), (bottom_right[0], bottom_right[1] - int(h * 0.2)), color, mid_thickness)

    # Draw the trajectory
    if not np.isnan(object_id).any():
        for point in object_trajectories[object_id]:
            point_x, point_y = int(point[0]), frame_height - int(point[1])  # Adjust for flipped y-coordinate
            cv2.circle(frame, (point_x, point_y), 2, color, -1)

    # Prepare the text labels
    bottom_text = f"Norm. Depth: {normalized_depth:.2f}\nNorm. Disp: {normalized_displacement:.2f}\nProb: {prob_of_interaction:.2f}\nThreat Level: {normalized_threat_level:.2f}"
    top_text = f"{label} | {intention} | Severity: {severity:.2f}"
    bottom_text = f"Norm. Depth: {normalized_depth:.2f}\nNorm. Disp: {normalized_displacement:.2f}\nProb: {prob_of_interaction:.2f}\nThreat Level: {normalized_threat_level:.2f}"

    # Determine the position for the top and bottom labels
    offset = 5
    top_text_position = (x, y - offset if y - offset > offset else y + offset)
    bottom_text_position = (x, bottom_right[1] + offset)

    # Add the top label (label, intention, and severity) above the bounding box
    cv2.putText(frame, top_text, top_text_position, cv2.FONT_HERSHEY_COMPLEX, 0.25, color, 1, lineType=cv2.LINE_AA)

    # Add the bottom label (normalized depth, normalized displacement, probability of interaction, and threat level) below the bounding box
    for i, line in enumerate(bottom_text.split('\n')):
        cv2.putText(frame, line, (bottom_text_position[0], bottom_text_position[1] + i * 10), cv2.FONT_HERSHEY_COMPLEX, 0.25, color, 1, lineType=cv2.LINE_AA)

    # Update the last seen frame for this object
    object_last_seen[object_id] = current_frame_num


def remove_stale_objects(current_frame_num, max_frames=5):
    global object_trajectories, object_last_seen

    stale_ids = [obj_id for obj_id, last_seen in object_last_seen.items() if current_frame_num - last_seen > max_frames]
    
    for obj_id in stale_ids:
        # Remove the trajectory history and last seen info for the stale object
        if obj_id in object_trajectories:
            del object_trajectories[obj_id]
        if obj_id in object_last_seen:
            del object_last_seen[obj_id]


def save_frame(frame, frame_num):
    save_dir = "prob_of_interaction_visuals"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{frame_num}.png")
    cv2.imwrite(filename, frame)
    print(f"Frame {frame_num} saved as {filename}")

def generate_random_colors(n):
    np.random.seed(42)  # For consistent color generation
    return {i: tuple(np.random.randint(0, 255, size=3).tolist()) for i in range(1, n+1)}

if __name__ == "__main__":

    data_path = "/data/TGSSE/DyViR Conference Paper 2024/Changing Intentions (10k)/SimData_2024-03-17__22-17-01_Optical"
    labels_filename = "SimData_2024-03-17__22-17-01_Optical.csv"
    mp4_filename = "SimData_2024-03-17__22-17-01_Optical.mp4"
    output_filename = "temp.mp4"  # Name of the output file
    csv_output_filename = "threat_levels.csv"  # Name of the output CSV file

    labels_path = os.path.join(data_path, labels_filename)
    video_path = os.path.join(data_path, mp4_filename)

    print("Loading labels...")
    if labels_path.endswith("csv"):
        labels_df = pd.read_csv(labels_path)
    elif labels_path.endswith("xlsx"):
        labels_df = pd.read_excel(labels_path)
    print("Done")

    # Calculate the maximum depth from the labels DataFrame
    max_depth = labels_df['depth_meters'].max()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
    else:
        # Get the video's width, height, and frame rate
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Generate random colors for object IDs
        unique_object_ids = labels_df['id'].unique()
        object_colors = generate_random_colors(len(unique_object_ids))

        # Define severity for each intention
        severity_dict = {
            "Recon": 0.01,
            "Kamikaze": 1.0,
            "Area Denial": 0.25,
            "Travel": 0.42
        }

        # Determine a random frame number to save
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame_num = random.randint(1, total_frames)

        # DataFrame to store threat levels
        threat_levels_df = pd.DataFrame(columns=["frame", "object_id", "intention", "severity", "prob_of_interaction", "threat_level"])

        # Process and save the video
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Filter the labels DataFrame to get all rows where 'frame' equals frame_num
            frame_labels = labels_df[labels_df['frame'] == frame_num]

            # Remove stale objects that haven't been seen for more than 5 frames
            remove_stale_objects(frame_num, max_frames=5)

            # List to store threat levels
            threat_levels = []

            # First pass: calculate all threat levels
            for index, row in frame_labels.iterrows():
                coords = eval(row['coords'])  # Convert string to list
                intention = row['intention']
                object_id = row.get('id', None)  # Assuming 'id' column exists
                depth = row.get('depth_meters', 0)  # Assuming 'depth' column exists

                # Update object trajectories and last seen frame
                if object_id not in object_trajectories:
                    object_trajectories[object_id] = []
                object_trajectories[object_id].append((coords[0], coords[1]))
                object_last_seen[object_id] = frame_num

                # Limit the trajectory to the last 100 points
                if len(object_trajectories[object_id]) > 100:
                    object_trajectories[object_id] = object_trajectories[object_id][-100:]

                # Calculate normalized depth and displacement using the helper functions
                normalized_depth = calculate_normalized_depth(depth, max_depth)
                trajectory = object_trajectories.get(object_id, [(coords[0], coords[1])])
                normalized_displacement = calculate_normalized_displacement(trajectory, frame_width, frame_height)

                # Get severity
                severity = severity_dict.get(intention, 0.0)

                # Calculate probability of interaction
                prob_of_interaction = calculate_probability_of_interaction(
                    intention, trajectory, frame_width, frame_height, depth, max_depth
                )

                # Calculate threat level
                threat_level = prob_of_interaction * severity
                threat_levels.append(threat_level)

                # Store the threat level information in the DataFrame
                new_row = pd.DataFrame([{
                    "frame": frame_num,
                    "object_id": object_id,
                    "intention": intention,
                    "severity": severity,
                    "prob_of_interaction": prob_of_interaction,
                    "threat_level": threat_level
                }])

                threat_levels_df = pd.concat([threat_levels_df, new_row], ignore_index=True)

            # Normalize the threat levels
            max_threat_level = max(threat_levels) if threat_levels else 1
            if max_threat_level == 0:
                max_threat_level = 1  # Prevent division by zero

            normalized_threat_levels = [t / max_threat_level for t in threat_levels]

            # Rank threat levels and assign colors
            sorted_indices = np.argsort(threat_levels)[::-1]  # Sort in descending order

            # Second pass: draw bounding boxes with normalized threat levels
            for index, (row, normalized_threat_level) in enumerate(zip(frame_labels.iterrows(), normalized_threat_levels)):
                coords = eval(row[1]['coords'])  # Convert string to list
                label = row[1]['label']
                intention = row[1]['intention']
                object_id = row[1].get('id', None)
                depth = row[1].get('depth_meters', 0)

                # Use the same calculated values from the first pass
                normalized_depth = calculate_normalized_depth(depth, max_depth)
                trajectory = object_trajectories.get(object_id, [(coords[0], coords[1])])
                normalized_displacement = calculate_normalized_displacement(trajectory, frame_width, frame_height)

                severity = severity_dict.get(intention, 0.0)
                prob_of_interaction = calculate_probability_of_interaction(intention, trajectory, frame_width, frame_height, depth, max_depth)
                threat_level = prob_of_interaction * severity

                # Sort the threat levels to determine the top 3
                top_3_threshold = sorted(threat_levels, reverse=True)[:3]

                # Assign the color based on whether the threat_level is in the top 3
                if threat_level in top_3_threshold:
                    color = (0, 0, 139)  # Dark cherry red for top 3 threats
                else:
                    color = (230, 230, 250)  # Lavender purple for the rest

                draw_bounding_box(
                    frame, coords, label, intention, object_id, color, frame_height,
                    normalized_depth, normalized_displacement, severity, prob_of_interaction, 
                    threat_level, frame_num
                )

            # Save the selected random frame
            if frame_num == random_frame_num:
                save_frame(frame, frame_num)

            # Write the processed frame to the output video
            out.write(frame)
            
            frame_num += 1

        # Release video capture and writer objects
        cap.release()
        out.release()


        threat_levels_df.to_csv(csv_output_filename, index=False)
        print(f"Threat levels saved to {csv_output_filename}")
        print(f"Output video saved as {output_filename}")
