# **Drone Intention Detection System**

## **Project Overview**

This project is a comprehensive solution for detecting, tracking, and classifying the intention of drones using a multi-headed CNN model integrated with a YOLO object detection model. The system processes video inputs, detects and tracks drones, and predicts their intentions based on flight trajectory data. The project also supports augmenting trajectory data and uses a multi-headed convolutional neural network (CNN) to handle different drone types.

### **Main Features**
- **Object Detection and Tracking:** Utilizes YOLOv8 for real-time detection and tracking of drones in video feeds.
- **Intention Prediction:** Based on drone trajectories, a multi-head CNN predicts the intention of drones (Recon, Kamikaze, Area Denial, etc.).
- **Visualization:** The system can visualize drone trajectories and predicted intentions on video frames.

## **Prerequisites**

Before running the project, ensure you have installed the following dependencies:
- Python 3.8+
- CUDA-enabled GPU with PyTorch installed for faster processing (optional but recommended)
- YOLOv8 for object detection and tracking
- Various Python libraries such as:
  - `numpy`
  - `pandas`
  - `torch`
  - `opencv-python`
  - `tqdm`
  - `yaml`
  - `multiprocessing`
  - `ultralytics` (for YOLO)
  
Install the dependencies by running:

```bash
pip install -r requirements.txt
```

## **Files Overview**

### **1. `intentCNN.py`**
This file contains the multi-head convolutional neural network (CNN) model used to classify drone intentions. The model is designed to handle different types of drones (e.g., ROTARY, FIXEDWING), each having its own specific head in the multi-headed CNN architecture.

**Functions and Classes:**
- `CNNModel`: Defines a simple CNN model for trajectory classification.
- `MultiHeadCNNModel`: A multi-headed CNN model used for handling various drone types and their corresponding intentions.
- `train_model()`, `evaluate_model()`, `train_cnn()`, and `inference()`: Functions that handle training, evaluation, and inference of the CNN model.

### **2. `intent_tracking.py`**
This is the main script that orchestrates the detection, tracking, and intention classification pipeline. The script reads a video file, tracks drones using YOLO, and predicts drone intentions based on their flight trajectories.

**Key Sections:**
- **Video Processing and Detection:** Handles loading video frames and detecting drones using a YOLO model.
- **Intention Prediction:** Based on the detected drone's trajectory, this section uses the trained CNN model to predict intentions.
- **File Handling:** Ensures proper loading of label files and setting up paths for output videos.
- **Visualization:** Draws trajectories and predicted intentions on video frames.

**How to run**: 
To run the tracking and intention prediction on a video, use:

```bash
python intent_tracking.py --detect_path <path_to_detection_model> --intent_path <path_to_intent_model> --video_path <path_to_video_file> --label_path <path_to_labels> --label_detailed_path <path_to_detailed_labels> --tracker_path <path_to_tracker_config> --cfg_path <path_to_tracker_cfg> --show <True/False>
```

### **3. `data_tools/normalization.py`**
Contains functions for normalizing trajectory data. Normalization is a critical preprocessing step to ensure that drone trajectories are transformed into a format that can be effectively processed by the CNN.

**Functions:**
- `z_score_standardization_all`, `z_score_standardization_single`: Standardizes trajectories using z-score normalization.
- `mean_removed_all`, `mean_removed_single`: Removes the mean from trajectory data.
- `normalize()`: Applies standard normalization to trajectories.

### **4. `data_tools/chart_generator.py`**
This file generates visualizations such as histograms and pie charts for analyzing drone trajectories and intentions. These charts can be useful for understanding the dataset distribution.

**Functions:**
- `generate_histogram_and_pie_chart()`: Generates overall stats for trajectory labels.
- `generate_histogram_and_pie_chart_for_split()`: Similar to the above but splits data into training and validation sets.

### **5. `data_tools/augmentations.py`**
This file provides augmentation techniques to artificially increase the diversity of the training data. These augmentations are used to improve the generalization of the CNN model.

**Functions:**
- `flip_trajectories_x()`: Flips trajectories along the x-axis to simulate different drone movements.
- `augment_with_jitters()`: Adds random jitter to drone trajectories to simulate noise.

### **6. `README.md`**
The file you are currently reading. Provides an overview of the project and explains the structure, usage, and additional information.

### **7. `cfgs/`**
This folder contains configuration files used by the tracking algorithm (e.g., BOTSORT). These configurations define parameters like tracker settings, which are used to track drones in videos.

## **How to Run the Project**

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <project_directory>
    ```

2. **Install Dependencies**
   Make sure you have Python 3.8+ and install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
    ```

3. **Prepare Model Weights**
  - **Detection Model**: Download or train a YOLO model for drone detection. Place the model weights in the weights/detection/ directory.
  - **Intent Model**: Ensure that a trained CNN model is available for predicting drone intentions. Place the weights in the trained_models/ directory.

4. **Run the Script** Use the following command to run the intention tracking on a sample video:
  ```bash
  python intent_tracking.py --detect_path <path_to_detection_model> --intent_path <path_to_intent_model> --video_path <path_to_video_file> --label_path <path_to_labels> --label_detailed_path <path_to_detailed_labels> --tracker_path <path_to_tracker_config> --cfg_path <path_to_tracker_cfg> --show True
  ```

5. **Batch Processing (Optional)** If you have multiple videos to process in batch mode, modify the video_paths list in intent_tracking.py to include multiple video files, then run the script in batch mode.

6. **Visualization** You can set show=True in the intent_tracking.py script to display real-time tracking and prediction results. The processed video with tracking will be saved in the project folder.
