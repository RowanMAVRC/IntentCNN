# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Python Imports
import time
from datetime import timedelta
import multiprocessing
import os
from collections import defaultdict
import sys

# Package Imports
import cv2
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import numpy as np

# File Imports
import cfgs.models.intentCNN

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

def intention_tracking(detect_path: str, 
                       intent_path: str,
                       video_path: str,
                       label_path: str,
                       label_detailed_path: str, 
                       tracker_path: str, 
                       cfg_path: str,
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
        label_dict = {}
        with open(label_path, 'r') as file:
            for line in file:
                key, value = line.split(': ')
                label_dict[int(key)] = value.strip()
        print(label_dict)  # For debugging purposes
        
        # Load detailed labels into a dictionary
        label_detailed_dict = {}
        with open(label_path, 'r') as file:
            for line in file:
                key, value = line.split(': ')
                label_detailed_dict[int(key)] = value.strip()
        print(label_detailed_dict)  # For debugging purposes
        
        intent_model = cfgs.models.intentCNN.load_model(cfgs.models.intentCNN.CNNModel, 
                                                        intent_path, 
                                                        3, 
                                                        len(label_dict))  # Load the intent model
        intent_model.cuda()  # Move the model to the GPU
                
        cap = cv2.VideoCapture(video_path)  # Load video from path
        # Load CSV data
        csv_file = video_path.replace(".mp4", ".csv")
        if os.path.exists(csv_file):
            frame_data = pd.read_csv(csv_file)
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
        
        # Initialize variables
        track_history = defaultdict(lambda: [])     # Store the track history
        trajectory_history = defaultdict(lambda: [])    # Store the trajectory history
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
                        track.pop(0)
                    if plot_tracks:
                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                    if track_class < 4:
                        trajectory = trajectory_history[track_id]
                        trajectory.append([x, y, 0])
                        intent = intents[track_id]
                        leftX, leftY, rightX, rightY = boxXYXY
                        if (len(trajectory) > 9) & (len(trajectory) % 25 == 0):
                            trajectory_array = np.array([trajectory])  # Convert list of lists to NumPy array
                            trajectory_array = z_score_standardization(trajectory_array)
                            trajectory = [" ".join(map(str, np.array(trajectory_array).flatten()))]
                            inputs = tokenizer(trajectory, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
                            intent_model.eval()
                            with torch.no_grad():
                                outputs = intent_model(**inputs)
                                logits = outputs.logits
                                probabilities = torch.softmax(logits, dim=1)
                                predicted_class = torch.argmax(probabilities, dim=1)
                                if id2label:
                                    intention = id2label[str(predicted_class.item())].split(" - ")[1]
                                else:
                                    intention = str(predicted_class.item())
                                intent.append(intention)
                                if verbose:
                                    print(f"Predicted class of ID {track_id}:  {intention}")
                        if intent:
                            cv2.putText(annotated_frame, intent[len(intent)-1], (int(leftX), int(leftY+1.7*h)), font, 1.7, (0, 0, 0), 2, cv2.LINE_AA)
                
                
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
    
    intent_path = "/home/cipoll17/IntentFormer/weights/CNN_fold_4.pth"
    
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
        intention_tracking(detect_path, intent_path, video_paths[0], tracker_path, cfg_path, plot_tracks=plot_tracks, device=device, show=show)
    
    # Close windows
    cv2.destroyAllWindows()
    # Print the total time taken to track the batch
    print("Total Runtime: " + str(timedelta(seconds=time.monotonic() - start_time)))