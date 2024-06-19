# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

import time
from datetime import timedelta
import multiprocessing
import os
from collections import defaultdict, deque
import sys
import cv2
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import numpy as np

import cfgs.models.intentCNN
from tools.normalization import normalize_single, mean_removed_single

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def get_output_path(video_path):
    """
    Generate a unique output path for the tracked video to avoid overwriting existing files.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        str: Unique output path for the tracked video.
    """
    video_name = os.path.basename(video_path)
    output_name = f"Tracked_{video_name}" if video_name.endswith(".mp4") else f"Tracked_{video_name}.mp4"
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

    Args:
        coord1 (tuple or list): Coordinates of the first object in xyxy format (x1, y1, x2, y2).
        coord2 (tuple or list): Coordinates of the second object in xyxy format (x1, y1, x2, y2).

    Returns:
        float: Euclidean distance between the two sets of coordinates.
    """
    x1, y1, x2, y2 = coord1
    x3, y3, x4, y4 = coord2
    centroid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    centroid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
    return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)

def load_labels(label_path, label_detailed_path):
    """
    Load labels from files into dictionaries.

    Args:
        label_path (str): Path to the file containing labels.
        label_detailed_path (str): Path to the file containing detailed labels.

    Returns:
        tuple: (id2label, label2id, id2label_detailed, label_detailed2id)
    """
    id2label = {}
    with open(label_path, 'r') as file:
        for line in file:
            key, value = line.split(': ')
            id2label[int(key)] = value.strip()
    label2id = {value: key for key, value in id2label.items()}

    id2label_detailed = {}
    with open(label_detailed_path, 'r') as file:
        for line in file:
            key, value = line.split(': ')
            id2label_detailed[int(key)] = value.strip()
    label_detailed2id = {value: key for key, value in id2label_detailed.items()}

    return id2label, label2id, id2label_detailed, label_detailed2id

def initialize_video_writer(video_path, output_path):
    """
    Initialize video writer for saving the output video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to the output video file.

    Returns:
        tuple: (cv2.VideoCapture, cv2.VideoWriter, float, int, int)
    """
    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), input_fps, (input_width, input_height))
    return cap, output, input_fps, input_width, input_height

def process_frame(frame, detect_model, tracker_path, cfg_path, verbose, device):
    """
    Process a single video frame to detect and track objects.

    Args:
        frame (numpy.ndarray): Input video frame.
        detect_model (YOLO): YOLO detection model.
        tracker_path (str): Path to the tracker configuration file.
        cfg_path (str): Path to the tracker configuration file.
        verbose (bool): Whether to print verbose output.
        device (str): Device to run the detection on.

    Returns:
        tuple: (list of boxes, list of boxes in XYXY format, list of track IDs, list of track classes, numpy.ndarray of annotated frame)
    """
    results = detect_model.track(source=frame, persist=True, tracker=tracker_path, cfg=cfg_path, verbose=verbose, device=device)
    boxes = results[0].boxes.xywh.cpu()
    boxesXYXY = results[0].boxes.xyxy.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
    track_classes = results[0].boxes.cls.int().cpu().tolist() if results[0].boxes.id is not None else []
    annotated_frame = results[0].plot(probs=False, conf=False)
    return boxes, boxesXYXY, track_ids, track_classes, annotated_frame

def intention_tracking(detect_path, intent_path, video_path, label_path, label_detailed_path, tracker_path, cfg_path, dimensions=2, **kwargs):
    """
    Perform object detection, tracking, and intention prediction on a video.

    Args:
        detect_path (str): Path to the detection model.
        intent_path (str): Path to the intent model.
        video_path (str): Path to the input video file.
        label_path (str): Path to the labels file.
        label_detailed_path (str): Path to the detailed labels file.
        tracker_path (str): Path to the tracker configuration file.
        cfg_path (str): Path to the tracker configuration file.
        dimensions (int, optional): Number of dimensions for intention prediction. Defaults to 3.
        **kwargs: Additional keyword arguments.
    """
    start_time = time.monotonic()
    output_path = None  # Initialize output_path

    try:
        paths = [detect_path, intent_path, video_path, label_path, label_detailed_path, tracker_path, cfg_path]
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file {path} does not exist.")
        
        detect_model = YOLO(detect_path)
        id2label, label2id, id2label_detailed, label_detailed2id = load_labels(label_path, label_detailed_path)

        intent_model = cfgs.models.intentCNN.load_model(cfgs.models.intentCNN.CNNModel, intent_path, dimensions, len(label2id), device="cuda:1")
        intent_model.to("cuda:1")

        output_path = get_output_path(video_path)
        cap, output, input_fps, input_width, input_height = initialize_video_writer(video_path, output_path)
        
        csv_file = video_path.replace(".mp4", ".csv")
        frame_data = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()
        if frame_data.empty:
            print(f"CSV file {csv_file} does not exist. Defaulting to intent with XY coordinates only.")
            dimensions = 2

        verbose = kwargs.get("verbose", False)
        device = kwargs.get("device", "cuda:1")
        show = kwargs.get("show", False)
        plot_tracks = kwargs.get("plot_tracks", True)
        num_track_frames = kwargs.get("num_track_frames", 90)
        font = kwargs.get("font", cv2.FONT_HERSHEY_SIMPLEX)
        
        track_history = defaultdict(lambda: deque(maxlen=100))
        trajectory_history = defaultdict(lambda: deque(maxlen=100))
        intents = defaultdict(list)
        frame_counter = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc='Tracking Progress', unit='frame'):
            success, frame = cap.read()
            if not success:
                break

            frame_counter += 1
            boxes, boxesXYXY, track_ids, track_classes, annotated_frame = process_frame(frame, detect_model, tracker_path, cfg_path, verbose, device)
            
            for box, boxXYXY, track_id, track_class in zip(boxes, boxesXYXY, track_ids, track_classes):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > num_track_frames // 3:
                    track.pop()
                if plot_tracks:
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                
                if track_class < 4:
                    trajectory = trajectory_history[track_id]
                    if dimensions == 2:
                        trajectory.append([x, y])
                    else:
                        frame_subset = frame_data[frame_data['frame'] == frame_counter]
                        closest_distance = float('inf')
                        for _, row in frame_subset.iterrows():
                            actual_xyxy = [float(coord) for coord in row['coords'].strip("[]").split(", ")]
                            distance = calculate_distance(actual_xyxy, boxXYXY)
                            if distance < closest_distance:
                                closest_distance = distance
                                depth = row['depth_meters']
                        trajectory.append([x, y, depth])
                    
                    if len(trajectory) > 9 and len(trajectory) % 25 == 0:
                        trajectory_array = np.array(trajectory)
                        if len(trajectory_array.shape) == 2:
                            trajectory_array = trajectory_array[np.newaxis, :, :]
                        trajectory_array = normalize_single(trajectory_array)
                        trajectory_array = mean_removed_single(trajectory_array)
                        if trajectory_array.shape[1] < 100:
                            padding = np.zeros((trajectory_array.shape[0], 100 - trajectory_array.shape[1], trajectory_array.shape[2]))
                            trajectory_array = np.hstack((padding, trajectory_array))
                        drone_class = np.array([[track_class] * dimensions]).reshape(1, 1, -1)
                        trajectory_array = np.concatenate((drone_class, trajectory_array), axis=1)
                        intent_results = cfgs.models.intentCNN.predict(intent_model, trajectory_array, device="cuda:1")
                        intention = id2label[intent_results[0]].split(" - ")[1]
                        intents[track_id].append(intention)

                        if verbose:
                            print(f"Predicted class of ID {track_id}: {intention}")
                    if intents[track_id]:
                        leftX, leftY, rightX, rightY = boxXYXY
                        cv2.putText(annotated_frame, intents[track_id][-1], (int(leftX), int(leftY + 1.5 * h)), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            
            if show:
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
            output.write(annotated_frame)
        
        output.release()
        cap.release()
        cv2.destroyAllWindows()
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
    start_time = time.monotonic()
    
    detect_path = "/data/TGSSE/weights/detection/DyViR_Combined.pt"    
    intent_path = "/home/cipoll17/IntentFormer/trained_models/100pad_0/fold_0.pth"
    label_path = "/data/TGSSE/UpdatedIntentions/intent_labels.yaml"
    label_detailed_path = "/data/TGSSE/UpdatedIntentions/XYZ/800pad_66/trajectory_with_intentions_800_pad_533_min_101108_label_detailed.yaml"
    tracker_path = "cfgs/tracking/trackers/botsort_90.yaml"
    cfg_path = "cfgs/tracking/botsort.yaml"
    plot_tracks = False
    device = "cuda:1"
    show = False
    
    video_paths = ["/data/TGSSE/UpdatedIntentions/DyViR_DS_240410_095823_Optical_6D0A0B0H/DyViR_DS_240410_095823_Optical_6D0A0B0H.mp4"]
    
    if batch:
        processes = []
        for video_path in video_paths:
            process = multiprocessing.Process(target=intention_tracking, 
                                              args=(detect_path, intent_path, video_path, label_path, label_detailed_path, tracker_path, cfg_path),
                                              kwargs={"plot_tracks": plot_tracks, "device": device, "show": show})
            processes.append(process)
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    else:
        intention_tracking(detect_path, intent_path, video_paths[0], label_path, label_detailed_path, tracker_path, cfg_path, plot_tracks=plot_tracks, device=device, show=show)
    
    cv2.destroyAllWindows()
    print("Total Runtime: " + str(timedelta(seconds=time.monotonic() - start_time)))
