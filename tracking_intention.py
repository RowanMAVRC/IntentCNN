from ultralytics import YOLO
import cv2
import sys
import numpy as np
from collections import defaultdict
import time
from datetime import timedelta
import yaml
import os
import pandas as pd
import json
import torch
import multiprocessing
from tqdm import tqdm
## Intention model imports
from transformers import (MobileBertForSequenceClassification,
                          MobileBertTokenizer,
                          AutoTokenizer)
from data_tools.normalization import z_score_standardization

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

def intention_tracking_2d(
        detect_path: str, 
        intent_path: str,
        video_path: str, 
        tracker_path: str, 
        cfg_path: str,
        **kwargs
        ) -> None:
    start_time = time.monotonic()   # Set start time of function
    try:
        detect_model = YOLO(detect_path)    # Load a detection model from path (Necessary to load a separate model for each thread)
        label_path = intent_path + "config.json"
        if os.path.exists(label_path):
            jsonFile = open(label_path, 'r')
            values = json.load(jsonFile)
            jsonFile.close()
            # Extract id2label and label2id fields
            id2label = values.get("id2label", {})
            label2id = values.get("label2id", {})
            num_labels = len(id2label)
            intent_model = MobileBertForSequenceClassification.from_pretrained(intent_path,
                                                                                num_labels=num_labels,
                                                                                id2label=id2label,
                                                                                label2id=label2id,  
                                                                                local_files_only=True)   # Load an intention model from path (A directory instead of a file)
        else:
            intent_model = MobileBertForSequenceClassification.from_pretrained(intent_path, 
                                                                                local_files_only=True)   # Load an intention model from path (A directory instead of a file)
        ## Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(intent_path, local_files_only=True)
        # tokenizer = MobileBertTokenizer.from_pretrained(intent_path, local_files_only=True)
        
        cap = cv2.VideoCapture(video_path)  # Load video from path
        output_path = get_output_path(video_path)
        print("Tracking " + video_path + " to " + output_path)
        
        ## Get the properties of the input video
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), input_fps, (input_width, input_height))
        
        # describe the type of font 
        # to be used. 
        font = cv2.FONT_HERSHEY_SIMPLEX
        show = kwargs.get("show", False)
        ## Read the tracker file
        if tracker_path.endswith(".yaml"):
            with open(tracker_path, 'r', encoding='utf8') as stream:
                content = yaml.safe_load(stream)
            num_track_frames = content['num_track_frames']
            verbose = content['verbose']
        else:
            num_track_frames = 90
            verbose = False
        plot_tracks = kwargs.get("plot_tracks", True)
        device = kwargs.get("device", "cuda")
        
        track_history = defaultdict(lambda: [])     # Store the track history
        trajectory_history = defaultdict(lambda: [])    # Store the trajectory history
        intents = defaultdict(lambda: [])   # Store the intents
        frame_counter = 0   
        
        # Replace your while loop with this:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc='Tracking Progress', unit='frame'):
            success, frame = cap.read()
            if success:
                frame_counter += 1
                ## Run YOLOv8 tracking on the frame, persisting tracks between frames, verbose=False to suppress console output
                results = detect_model.track(source=frame, 
                                            persist=True, 
                                            tracker=tracker_path, 
                                            cfg=cfg_path, 
                                            verbose=verbose, 
                                            device=device,
                                            classes=[1],
                                            conf=0.5)

                ## Get the boxes and track IDs (if available)
                boxes = results[0].boxes.xywh.cpu()
                boxesXYXY = results[0].boxes.xyxy.cpu()
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    track_classes = results[0].boxes.cls.int().cpu().tolist()
                else:
                    track_ids = []
                    track_classes = []
                
                ## Visualize the results on the frame (Bounding boxes, labels, and track IDs)
                annotated_frame = results[0].plot(probs=False, conf=False)

                ## Plot the tracks
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

        ## Release the video capture object, writer object, and close the display window
        output.release()
        cap.release()
        cv2.destroyAllWindows()
        
        ## Print the time taken to track the video
        print("Finished: " + output_path + " in " + str(timedelta(seconds=time.monotonic() - start_time)))

    except KeyboardInterrupt:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'output' in locals():
            output.release()
        cv2.destroyAllWindows()
        print("Code terminated by user. Exiting gracefully...")
        sys.exit(0)

def intention_tracking(
        detect_path: str, 
        intent_path: str,
        video_path: str, 
        tracker_path: str, 
        cfg_path: str,
        **kwargs
        ) -> None:
    start_time = time.monotonic()   # Set start time of function
    detect_model = YOLO(detect_path)    # Load a detection model from path (Necessary to load a separate model for each thread)
    label_path = intent_path + "config.json"
    if os.path.exists(label_path):
        jsonFile = open(label_path, 'r')
        values = json.load(jsonFile)
        jsonFile.close()
        # Extract id2label and label2id fields
        id2label = values.get("id2label", {})
        label2id = values.get("label2id", {})
        num_labels = len(id2label)
        intent_model = MobileBertForSequenceClassification.from_pretrained(intent_path,
                                                                            num_labels=num_labels,
                                                                            id2label=id2label,
                                                                            label2id=label2id,  
                                                                            local_files_only=True)   # Load an intention model from path (A directory instead of a file)
    else:
        intent_model = MobileBertForSequenceClassification.from_pretrained(intent_path, 
                                                                            local_files_only=True)   # Load an intention model from path (A directory instead of a file)
    ## Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(intent_path, local_files_only=True)
    # tokenizer = MobileBertTokenizer.from_pretrained(intent_path, local_files_only=True)
    
    cap = cv2.VideoCapture(video_path)  # Load video from path
    # Load CSV data
    csv_file = video_path.replace(".mp4", ".csv")
    if os.path.exists(csv_file):
        frame_data = pd.read_csv(csv_file)
    output_path = get_output_path(video_path)
    print("Tracking " + video_path + " to " + output_path)
    
    ## Get the properties of the input video
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), input_fps, (input_width, input_height))
    
    # describe the type of font 
    # to be used. 
    font = cv2.FONT_HERSHEY_SIMPLEX
    show = kwargs.get("show", False)
    ## Read the tracker file
    if tracker_path.endswith(".yaml"):
        with open(tracker_path, 'r', encoding='utf8') as stream:
            content = yaml.safe_load(stream)
        num_track_frames = content['num_track_frames']
        verbose = content['verbose']
    else:
        num_track_frames = 90
        verbose = False
    plot_tracks = kwargs.get("plot_tracks", True)
    device = kwargs.get("device", "cuda")
    
    track_history = defaultdict(lambda: [])     # Store the track history
    trajectory_history = defaultdict(lambda: [])    # Store the trajectory history
    intents = defaultdict(lambda: [])   # Store the intents
    frame_counter = 0   
    
    ## Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()     # Read a frame from the video

        if success:
            frame_counter += 1
            ## Run YOLOv8 tracking on the frame, persisting tracks between frames, verbose=False to suppress console output
            results = detect_model.track(source=frame, 
                                         persist=True, 
                                         tracker=tracker_path, 
                                         cfg=cfg_path, 
                                         verbose=verbose, 
                                         device=device)

            ## Get the boxes and track IDs (if available)
            boxes = results[0].boxes.xywh.cpu()
            boxesXYXY = results[0].boxes.xyxy.cpu()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                track_classes = results[0].boxes.cls.int().cpu().tolist()
            else:
                track_ids = []
                track_classes = []
            
            ## Visualize the results on the frame (Bounding boxes, labels, and track IDs)
            annotated_frame = results[0].plot(probs=False, conf=False)

            ## Plot the tracks
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
                        cv2.putText(annotated_frame, intent[len(intent)-1], (int(leftX), int(leftY+1.5*h)), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Display the annotated frame
            if show:
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
            output.write(annotated_frame)
    
        else:
            # Break the loop if the end of the video is reached
            break

    ## Release the video capture object, writer object, and close the display window
    output.release()
    cap.release()
    cv2.destroyAllWindows()
    
    ## Print the time taken to track the video
    print("Finished: " + output_path + " in " + str(timedelta(seconds=time.monotonic() - start_time)))
    
if __name__ == "__main__":
    batch = False
    ## Set start time of the entire batch
    start_time = time.monotonic()
    
    ## Settings
    # detect_path = "/data/TGSSE/weights/detection/DyViR_Combined.pt"
    detect_path = "/data/TGSSE/weights/detection/ijcnn_full_vr_set_finetuned_rw_experiment-0_best.pt"
    
    
    # intent_path = "/data/TGSSE/weights/intention/IntentFormer/"
    # intent_path = "/home/cipoll17/IntentFormer/results/IntentFormerXYZ66/400pad_66_Fold2/checkpoint-29175/"
    intent_path = "/home/cipoll17/IntentFormer/results/IntentFormerXYZ0/600pad_0_Fold2/checkpoint-10050/"
    # intent_path = "/data/TGSSE/weights/intention/newLabels_750pad/"
    # intent_path = "/data/TGSSE/weights/intention/newLabels_500pad/"
    # intent_path = "/data/TGSSE/weights/intention/newLabels_250pad/"
    
    tracker_path = "cfgs/tracking/trackers/botsort_90.yaml"
    cfg_path = "cfgs/tracking/botsort.yaml"
    plot_tracks = False
    device = 0
    show = False
    
    # video_paths = ["/data/TGSSE/DyViR Conference Paper 2024/Changing Intentions (10k)/SimData_2024-03-17__22-17-01_Optical/SimData_2024-03-17__22-17-01_Optical.mp4"]
    # video_paths = ["/data/TGSSE/DyViR Conference Paper 2024/Same POV Multi-Modality (300k)/SimData_2024-03-17__11-53-42_Optical/SimData_2024-03-17__11-53-42_Optical.mp4"]
    # video_paths = ["/data/TGSSE/DyViR Conference Paper 2024/Upgraded Environments (10k)/SimData_2024-03-17__23-56-30_Optical/SimData_2024-03-17__23-56-30_Optical.mp4"]
    # video_paths = ["/data/TGSSE/DyViR Conference Paper 2024/Camera Scratches (10k)/SimData_2024-03-17__22-09-12_Optical/SimData_2024-03-17__22-09-12_Optical.mp4"]
    # video_paths = ["/data/TGSSE/UpdatedIntentions/DyViR_DS_240410_095823_Optical_6D0A0B0H/DyViR_DS_240410_095823_Optical_6D0A0B0H.mp4"]
    # video_paths = ["/data/TGSSE/Real_World_Intentions/Data Drone Flying/5_ Area Denial, Hovers Target Area/IMG_0866.MOV",
    #                "/data/TGSSE/Real_World_Intentions/Data Drone Flying/1_ Travel, Random Smooth Movement/IMG_1199.MOV",
    #                "/data/TGSSE/Real_World_Intentions/Data Drone Flying/1_ Travel, Random Smooth Movement/IMG_1468.MOV",
    #                "/data/TGSSE/Real_World_Intentions/Data Drone Flying/2_ Follow Path, Circular Path/Far_away_circles_from_car.mov",
    #                "/data/TGSSE/Real_World_Intentions/Data Drone Flying/2_ Follow Path, Circular Path/IMG_1200.MOV",
    #                "/data/TGSSE/Real_World_Intentions/Data Drone Flying/2_ Follow Path, Circular Path/IMG_1487.MOV"]
    video_paths = ["/data/TGSSE/Real_World_Intentions/Data Drone Flying/5_ Area Denial, Hovers Target Area/IMG_0866.MOV"]
    
    if batch:
        ## Array to store processes
        processes = []
            
        ## Create a process for each video
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
        ## Run the tracking function for a single video
        # intention_tracking(detect_path, intent_path, video_paths[0], tracker_path, cfg_path, plot_tracks=plot_tracks, device=device, show=show)
        for video_path in video_paths:
            intention_tracking_2d(detect_path, intent_path, video_path, tracker_path, cfg_path, plot_tracks=plot_tracks, device=device, show=show)
    
    ## Close windows
    cv2.destroyAllWindows()
    ## Print the total time taken to track the batch
    print("Total Runtime: " + str(timedelta(seconds=time.monotonic() - start_time)))