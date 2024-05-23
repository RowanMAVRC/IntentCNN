# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Python Imports
import time
from datetime import timedelta
import multiprocessing
import os
from collections import defaultdict, deque
import sys

# Package Imports
import cv2
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import numpy as np

# File Imports
import cfgs.models.intentCNN
from tools.normalization import normalize, mean_removed_all

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def get_output_path(video_path):
    video_name = os.path.basename(video_path)
    if video_name.endswith(".mp4"):
        output_name = "Tracked_" + video_name
    else:
        output_name = "Tracked_" + video_name + ".mp4"

    output_path = output_name
    run_number = 0
    while os.path.exists(output_path):
        run_number += 1
        base_name, extension = os.path.splitext(output_name)
        output_path = f"{base_name}_run{run_number}{extension}"

    return output_path

def calculate_distance(coord1, coord2):
    """
    Calculate the Euclidean distance between two sets of coordinates.

    Parameters:
    - coord1 (tuple or list): Coordinates of the first object in xyxy format (x1, y1, x2, y2).
    - coord2 (tuple or list): Coordinates of the second object in xyxy format (x1, y1, x2, y2).

    Returns:
    - distance (float): Euclidean distance between the two sets of coordinates.
    """
    x1, y1, x2, y2 = coord1
    x3, y3, x4, y4 = coord2
    centroid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    centroid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
    distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
    return distance

def intention_tracking(detect_path: str, 
                       intent_path: str,
                       video_path: str,
                       label_path: str,
                       label_detailed_path: str, 
                       tracker_path: str, 
                       cfg_path: str,
                       dimensions: int=3,
                       **kwargs) -> None:
    start_time = time.monotonic()       # Set start time of function
    try:
        # Check if all provided paths exist
        paths = [detect_path, intent_path, video_path, label_path, label_detailed_path, tracker_path, cfg_path]
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file {path} does not exist.")
            
        detect_model = YOLO(detect_path)    # Load a detection model from path (Necessary to load a separate model for each thread)
        
        # Load labels into a dictionary
        id2label = {}
        with open(label_path, 'r') as file:
            for line in file:
                key, value = line.split(': ')
                id2label[int(key)] = value.strip()
        # Create the reverse dictionary label2id
        label2id = {value: key for key, value in id2label.items()}
        print("label2id: ", label2id)  # For debugging purposes
        print("id2label: ", id2label)  # For debugging purposes
        
        # Load detailed labels into a dictionary
        id2label_detailed = {}
        with open(label_detailed_path, 'r') as file:
            for line in file:
                key, value = line.split(': ')
                id2label_detailed[int(key)] = value.strip()
        # Create the reverse dictionary label_detailed2id
        label_detailed2id = {value: key for key, value in id2label_detailed.items()}
        print("label_detailed2id: ", label_detailed2id)  # For debugging purposes
        print("id2label_detailed", id2label_detailed)  # For debugging purposes
        
        intent_model = cfgs.models.intentCNN.load_model(cfgs.models.intentCNN.CNNModel, 
                                                        intent_path, 
                                                        3, 
                                                        len(label2id))  # Load the intent model
        intent_model.cuda()  # Move the model to the GPU
                
        cap = cv2.VideoCapture(video_path)  # Load video from path
        
        # Load CSV data
        csv_file = video_path.replace(".mp4", ".csv")
        if os.path.exists(csv_file):
            frame_data = pd.read_csv(csv_file)
        else:
            print(f"CSV file {csv_file} does not exist. Defualting to intent with XY coordinates only.")
            dimensions = 2
        output_path = get_output_path(video_path)
        print("Tracking " + video_path + " to " + output_path)
        
        # Get the properties of the input video
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), input_fps, (input_width, input_height))
        
        # Read kwargs
        verbose = kwargs.get("verbose", False)
        device = kwargs.get("device", "cuda")
        show = kwargs.get("show", False)
        plot_tracks = kwargs.get("plot_tracks", True)
        num_track_frames = kwargs.get("num_track_frames", 90)
        font = kwargs.get("font", cv2.FONT_HERSHEY_SIMPLEX)
        
        # Initialize variables
        track_history = defaultdict(lambda: deque(maxlen=100))  # Store the track history
        trajectory_history = defaultdict(lambda: deque(maxlen=100))  # Store the trajectory history
        intents = defaultdict(lambda: [])   # Store the intents
        frame_counter = 0
        
        # Loop through the video frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc='Tracking Progress', unit='frame'):
            success, frame = cap.read()    # Read the next frame
            
            if success:
                frame_counter += 1
                
                ## Run YOLOv8 tracking on the frame, persisting tracks between frames, verbose=False to suppress console output
                results = detect_model.track(source=frame, 
                                            persist=True, 
                                            tracker=tracker_path, 
                                            cfg=cfg_path, 
                                            verbose=verbose, 
                                            device=device)
                
                # Get the boxes and track IDs (if available)
                boxes = results[0].boxes.xywh.cpu()
                boxesXYXY = results[0].boxes.xyxy.cpu()
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    track_classes = results[0].boxes.cls.int().cpu().tolist()
                else:
                    track_ids = []
                    track_classes = []
                
                # Visualize the results on the frame (Bounding boxes, labels, and track IDs)
                annotated_frame = results[0].plot(probs=False, conf=False)
                
                # Plot the tracks
                for box, boxXYXY, track_id, track_class in zip(boxes, boxesXYXY, track_ids, track_classes):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center points
                    if len(track) > num_track_frames//3:  # retain 90 tracks for 90 frames
                        track.pop()
                    if plot_tracks:
                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                    
                    # Intention prediction
                    if track_class < 4:
                        trajectory = trajectory_history[track_id]
                        
                        if dimensions == 2:
                            trajectory.append([x, y])
                        else:
                            closest_distance = float('inf')
                            # Filter the DataFrame for rows with the current frame number
                            frame_data = frame_data[frame_data['frame'] == frame_counter]
                            # Iterate over each row in the filtered DataFrame
                            for index, row in frame_data.iterrows():
                                actual_xyxy = row['coords']
                                actual_xyxy = actual_xyxy.strip("[]").split(", ")
                                actual_xyxy = [float(coord) for coord in actual_xyxy]
                                distance = calculate_distance(actual_xyxy, boxXYXY)
                                if distance < closest_distance:
                                    closest_distance = distance
                                    depth = row['depth_meters']
                            trajectory.append([x, y, depth])
                            
                        intent = intents[track_id]
                        leftX, leftY, rightX, rightY = boxXYXY
                        if (len(trajectory) > 9) & (len(trajectory) % 25 == 0):
                            trajectory_array = np.array(trajectory)  # Convert deque to NumPy array
                            
                            # Ensure the array has 3 dimensions
                            if len(trajectory_array.shape) == 2:
                                trajectory_array = trajectory_array[np.newaxis, :, :]
                            
                            trajectory_array = normalize(trajectory_array)
                            trajectory_array = mean_removed_all(trajectory_array)

                            # Ensure the array is padded to 800 length with zeros initially
                            if trajectory_array.shape[1] < 100:
                                padding = np.zeros((trajectory_array.shape[0], 100 - trajectory_array.shape[1], trajectory_array.shape[2]))
                                trajectory_array = np.hstack((padding, trajectory_array))
                            
                            if dimensions == 2:
                                # Element to be added at index 0
                                drone_class = np.array([[track_class, track_class]])
                            else:
                                # Element to be added at index 0
                                drone_class = np.array([[track_class, track_class, track_class]])
                            
                            # Reshape new_element to match the dimensions for concatenation
                            drone_class = drone_class.reshape(1, 1, -1)
                            # Concatenate the new element along axis 1
                            trajectory_array = np.concatenate((drone_class, trajectory_array), axis=1)
                            intent_results = cfgs.models.intentCNN.predict(intent_model, trajectory_array)
                            intention = id2label[intent_results[0]].split(" - ")[1]
                            intent.append(intention)

                            if verbose:
                                print(f"Predicted class of ID {track_id}:  {intention}")
                        if intent:
                            cv2.putText(annotated_frame, intent[len(intent)-1], (int(leftX), int(leftY+1.5*h)), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Display the annotated frame
                if show:
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)
                output.write(annotated_frame)
                
            else:
                # Break the loop if the end of the video is reached
                break
            
        # Release the video capture object, writer object, and close the display window
        output.release()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print the time taken to track the video
        print("Finished: " + output_path + " in " + str(timedelta(seconds=time.monotonic() - start_time)))
    
    except KeyboardInterrupt:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'output' in locals():
            output.release()
        cv2.destroyAllWindows()
        print("Code terminated by user. Exiting gracefully...")
        sys.exit(0)

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    batch = False
    
    # Set start time of the entire batch
    start_time = time.monotonic()
    
    # Settings
    detect_path = "/data/TGSSE/weights/detection/DyViR_Combined.pt"    
    # intent_path = "/home/cipoll17/IntentFormer/weights/CNN_fold_4.pth"
    intent_path = "/home/cipoll17/IntentFormer/trained_models/400pad_66/fold_0.pth"
    # intent_path = "/home/cipoll17/IntentFormer/weights/CNN_fold_0.pth"
    label_path = "/data/TGSSE/UpdatedIntentions/labels.yaml"
    label_detailed_path = "/data/TGSSE/UpdatedIntentions/XYZ/800pad_66/trajectory_with_intentions_800_pad_533_min_101108_label_detailed.yaml"
    
    tracker_path = "cfgs/tracking/trackers/botsort_90.yaml"
    cfg_path = "cfgs/tracking/botsort.yaml"
    plot_tracks = False
    device = 0
    show = False
    
    video_paths = ["/data/TGSSE/UpdatedIntentions/DyViR_DS_240410_095823_Optical_6D0A0B0H/DyViR_DS_240410_095823_Optical_6D0A0B0H.mp4"]
    
    if batch:
        # Array to store processes
        processes = []
            
        # Create a process for each video
        for video_path in video_paths:
            process = multiprocessing.Process(target=intention_tracking, 
                                            args=(detect_path, 
                                                    intent_path, 
                                                    video_path, 
                                                    tracker_path, 
                                                    cfg_path),
                                            kwargs={"plot_tracks": plot_tracks, 
                                                    "device": device,
                                                    "show": show})
            processes.append(process)
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    else:
        # Run the tracking function for a single video
        # intention_tracking(detect_path, intent_path, video_paths[0], tracker_path, cfg_path, plot_tracks=plot_tracks, device=device, show=show)
        intention_tracking(detect_path, intent_path, video_paths[0], label_path, label_detailed_path, tracker_path, cfg_path, plot_tracks=plot_tracks, device=device, show=show)
    
    # Close windows
    cv2.destroyAllWindows()
    # Print the total time taken to track the batch
    print("Total Runtime: " + str(timedelta(seconds=time.monotonic() - start_time)))