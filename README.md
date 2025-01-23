# **Drone Intention & Detection System**

## **Project Overview**
This project is a comprehensive solution for detecting, tracking, and classifying the intention of drones using a multi-headed CNN model integrated with a YOLO object detection model. The system processes video inputs, detects and tracks drones, and predicts their intentions based on flight trajectory data. The project also supports augmenting trajectory data and uses a multi-headed convolutional neural network (CNN) to handle different drone types.

### **Main Features**
- **Object Detection and Tracking:** Utilizes YOLOv8 for real-time detection and tracking of drones in video feeds.
- **Intention Prediction:** Based on drone trajectories, a multi-head CNN predicts the intention of drones (Recon, Kamikaze, Area Denial, etc.).
- **Visualization:** The system can visualize drone trajectories and predicted intentions on video frames.

## **How to Run the Project**

1. **Clone the Repository**
    ```bash
    git clone https://github.com/RowanMAVRC/IntentCNN
    cd IntentCNN/
    ```

2. **Install Dependencies**
    Make sure you have Python>=3.8,<3.12 and install the necessary dependencies. It is reccomended to use a python environment manager of your choice (venv, conda, etc.). Make sure your environment is active before running the command below:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download and Prepare Data** 
    - **Download the Data:** Download the necessary data from [this link](https://drive.google.com/drive/folders/1chhkDlgXcXTwapc9j2JU7MgO1toZ6oF6?usp=drive_link) and place the .zip file in the project directory. You can either download just the .zip file or the entire folder through drive.
    - **Unzip the Data:** Unzip the downloaded `.zip` file in the main project directory. This should create a folder named `IntentCNN`.
    ```bash
    # Assuming you've downloaded IntentCNN.zip to the project directory
    unzip IntentCNN.zip
    ```
    - **Data Structure:** The `IntentCNN/` folder should contain all the necessary data for training the CNN model, including trajectory data and labels.

4. **Prepare Model Weights**
    - **Detection Model**: The YOLO model weights for drone detection are inside the `IntentCNN/Weights/detection_models` folder under the file name `detection.pt`
    - **Intent Model**: 
      - Every intention model weight used for the experiment are inside the `IntentCNN/Weights/intention_models` folder.
      - If you plan to train your own model, proceed to the next step to train the CNN model.

5. **Train the CNN Model**
    Below is the command used to train a CNN model using the data in the `IntentCNN/` folder.
    ```bash
    python cnn_main.py --data_path IntentCNN/Useable/XY/800pad_66 --num_epochs 50 --batch_size 32
    ```
      - Put the desired Useable data folder as the datapath.
      - Put with the desired number of epochs.
      - Put the desired batch size.

6. **Run the Tracking and Intention Prediction Script** Use the following command to run the intention tracking on a sample video:
    ```bash
    python tracking_w_intent.py
    ```

6. **Batch Processing (Optional)** If you have multiple videos to process in batch mode, modify the video_paths list in intent_tracking.py to include multiple video files, then run the script in batch mode.

7. **Visualization** You can set show=True in the intent_tracking.py script to display real-time tracking and prediction results. The processed video with tracking will be saved in the project folder.


## **Files Overview**

### **1. `cnn_main.py`**
This file contains the main script for training the CNN model that is used for classifying drone intentions. Before running the detection and tracking script, you should first train the CNN model using this file.
**How to run:** To train the model, execute the following command: 
```bash
python cnn_main.py --data_path <path_to_data> --epochs <number_of_epochs> --batch_size <batch_size> --save_model_path <path_to_save_model>
```
Ensure you have the necessary trajectory data in the `IntentCNN/` folder as mentioned earlier.

### **2. `intentCNN.py`**
This file contains the architecture for the multi-head convolutional neural network (CNN) model used for classifying drone intentions. The model is designed to handle different drone types, each type having its own head in the multi-headed CNN architecture. The `cnn_main.py` script will use this model for training and inference.

**Functions and Classes:**
- `CNNModel`: Defines a simple CNN model for trajectory classification.
- `MultiHeadCNNModel`: A multi-headed CNN model used for handling various drone types and their corresponding intentions.
- `train_model()`, `evaluate_model()`, `train_cnn()`, and `inference()`: Functions that handle training, evaluation, and inference of the CNN model.

### **3. `tracking_w_intent.py`**
This is the main script that orchestrates the detection, tracking, and intention classification pipeline. The script reads a video file, tracks drones using a YOLO model, and predicts drone intentions based on their flight trajectories using the pre-trained CNN model.
**How to run:** Once you have trained the CNN model, you can use this script to track drones in a video and classify their intentions by executing the following command:
```bash
python tracking_w_intent.py --detect_path <path_to_detection_model> --intent_path <path_to_intent_model> --video_path <path_to_video_file> --label_path <path_to_labels> --label_detailed_path <path_to_detailed_labels> --tracker_path <path_to_tracker_config> --cfg_path <path_to_tracker_cfg> --show <True/False>
```

**Key Sections:**
- **Video Processing and Detection:** Handles loading video frames and detecting drones using a YOLO model.
- **Intention Prediction:** Based on the detected drone's trajectory, this section uses the trained CNN model to predict intentions.
- **File Handling:** Ensures proper loading of label files and setting up paths for output videos.
- **Visualization:** Draws trajectories and predicted intentions on video frames.

### **4. `mobileB_trajectory_train.py`**
This is an accessory script that trains a CNN model using MobileBERT for trajectory-based classification. While it’s not part of the primary workflow, it can be used as an additional tool if needed.
**How to run:** You can execute this script separately if you wish to experiment with MobileBERT for trajectory prediction:
```bash
python mobileB_trajectory_train.py --data_path <path_to_data> --epochs <number_of_epochs> --batch_size <batch_size> --save_model_path <path_to_save_model>
```

### **5. `tools/normalization.py`**
Contains functions for normalizing trajectory data. Normalization is a critical preprocessing step to ensure that drone trajectories are transformed into a format that can be effectively processed by the CNN.

**Functions:**
- `z_score_standardization_all`, `z_score_standardization_single`: Standardizes trajectories using z-score normalization.
- `mean_removed_all`, `mean_removed_single`: Removes the mean from trajectory data.
- `normalize()`: Applies standard normalization to trajectories.

### **6. `tools/chart_generator.py`**
This file generates visualizations such as histograms and pie charts for analyzing drone trajectories and intentions. These charts can be useful for understanding the dataset distribution.

**Functions:**
- `generate_histogram_and_pie_chart()`: Generates overall stats for trajectory labels.
- `generate_histogram_and_pie_chart_for_split()`: Similar to the above but splits data into training and validation sets.

### **7. `tools/augmentations.py`**
This file provides augmentation techniques to artificially increase the diversity of the training data. These augmentations are used to improve the generalization of the CNN model.

**Functions:**
- `flip_trajectories_x()`: Flips trajectories along the x-axis to simulate different drone movements.
- `augment_with_jitters()`: Adds random jitter to drone trajectories to simulate noise.

### **8. `README.md`**
The file you are currently reading. Provides an overview of the project and explains the structure, usage, and additional information.

### **9. `cfgs/`**
This folder contains configuration files used by the tracking algorithm (e.g., BOTSORT). These configurations define parameters like tracker settings, which are used to track drones in videos.

### **10. `tools/grab_trajectories_from_csv.py`**
This Python script is designed to convert CSV and MP4 files into labeled drone trajectory sequences. It aligns video frames with their corresponding labels, organizes them into sequences by drone ID and intention, and saves the processed data for further analysis or training machine learning models.
- Finding Matching Files: The script identifies pairs of CSV files (containing labels) and MP4 files (containing videos) by matching their base filenames within the specified data directory.
- Processing Video Data: Each video is read frame by frame, and the corresponding labels are extracted from the matching CSV file. Labels include details like drone ID, intention, coordinates, and depth. Frames with the same drone ID and intention are grouped into sequences.
- Sequence Merging: Sequences from multiple files are merged into a global dictionary, organized by drone ID and intention.
- Statistics Calculation: The script calculates sequence length statistics for all intentions, including the longest, shortest, and average sequence lengths.
- Saving Results: The sequences are saved in a .pickle file for easy loading in future tasks. A bar chart of sequence statistics is generated and saved as a .png file.

### **11. `tools/format_trajectories.py`**
This script processes trajectory data from .pickle files, adjusts them for length consistency, and saves the formatted data for further analysis or training machine learning models.
- Trajectory Padding and Adjustment: The script ensures all trajectory sequences have a consistent length (max_length) by adding padding where necessary. Sequences shorter than a minimum length (min_length, derived from min_factor) are excluded.
- Padding Statistics: The script calculates and records padding statistics for each trajectory intention group. These include total padding, average padding per chunk, and the number of trajectory chunks.
- Label Mapping: Trajectories are categorized by drone type (ROTARY or FIXEDWING) and intention, with mappings created for string-to-number conversion (str2num and num2str) to facilitate tensor representation.
- Dimensionality Options: The script supports 2D (XY coordinates) and 3D (XYZ coordinates with depth) trajectory formats, determined by the dimensions parameter.
- Saving Outputs: The processed trajectories are saved as PyTorch .pt tensor files, along with YAML files containing label mappings for easy lookup.
- Visualization: A bar chart of padding statistics is generated for each processed file, showing the number of trajectories, total padding, and average padding per intention group.