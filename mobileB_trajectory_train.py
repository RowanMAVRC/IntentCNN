"""
 _____       _             _   ______                             
|_   _|     | |           | | |  ____|                            
  | |  _ __ | |_ ___ _ __ | |_| |__ ___  _ __ _ __ ___   ___ _ __ 
  | | | '_ \| __/ _ \ '_ \| __|  __/ _ \| '__| '_ ` _ \ / _ \ '__|
 _| |_| | | | ||  __/ | | | |_| | | (_) | |  | | | | | |  __/ |   
|_____|_| |_|\__\___|_| |_|\__|_|  \___/|_|  |_| |_| |_|\___|_|   
                                                                  
## Summary
This script sets up an environment for training a BERT-based deep learning model to classify flight trajectories,
utilizing PyTorch, Hugging Face's transformers, datasets, and evaluate libraries. It begins by configuring CUDA
settings for GPU use, then generates a synthetic dataset of flight trajectories with binary labels. This data is
prepared for processing by converting the trajectories into a text-like format suitable for BERT. The model is
trained with custom training arguments, including learning rate, batch size, and evaluation strategy, leveraging a
Trainer object for efficient training and evaluation. Accuracy is used as the metric for performance evaluation.
The script concludes by demonstrating inference on a new trajectory, showcasing the end-to-end process from data
preparation, model training, to making predictions in a domain-specific application.
"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# This needs to be done before other imports
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Python Imports
import argparse

# Package Imports
import numpy as np
import pandas as pd
import torch
import yaml
import wandb
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    MobileBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
from datasets import Dataset, DatasetDict

# File Imports
from data_tools.chart_generator import generate_histogram_and_pie_chart, generate_histogram_and_pie_chart_for_split
from data_tools.normalization import (
    z_score_standardization_all, z_score_standardization_single, 
    mean_removed_all, mean_removed_single,
    compute_trajectory_stats,
    normalize
)
from data_tools.augmentations import flip_trajectories_x, augment_with_jitters

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def load_flight_data(data_path: str, 
                     labels_path: str,
                     label_detailed_path: str,
                     augment: bool = False) -> tuple:
    """
    Load flight trajectory data and associated labels for a binary classification task.

    Args:
        data_path (str): Path to the directory containing trajectory data (.pt files) or to a single trajectory data file.
        labels_path (str): Path to the file containing labels corresponding to trajectory data (.yaml).
        augment (bool, optional): Whether to augment the data by flipping trajectories. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Training trajectories represented as 3-dimensional coordinates.
            - list: Binary labels corresponding to the training trajectories.
            - dict: Mapping of trajectory IDs to their labels.
            - dict: Mapping of labels to their corresponding IDs.
            
    Raises:
        FileNotFoundError: If either data_path or labels_path does not exist.
    """
    # Check if data directory exists
    if not os.path.exists(data_path) or data_path == "" or data_path is None:
        raise FileNotFoundError(f"Data directory '{data_path}' does not exist.")
    
    # Check if labels file exists
    if not os.path.exists(labels_path) or labels_path == "" or labels_path is None:
        raise FileNotFoundError(f"Labels file '{labels_path}' does not exist.")
    
    # Check if detailed labels file exists
    if not os.path.exists(label_detailed_path) or label_detailed_path == "" or label_detailed_path is None:
        raise FileNotFoundError(f"Labels file '{label_detailed_path}' does not exist.")
    
    # Load training trajectories
    if os.path.isfile(data_path):
        # If data_path is a file, load just that file
        trajectory_files = [data_path]
    else:
        # If data_path is a directory, load all .pt files in that directory
        trajectory_files = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('.pt')]

    # Load training trajectories from .pt files in the directory
    train_trajectories = []
    for file_name in trajectory_files:
        trajectory = torch.load(file_name).numpy()
        # Concatenate sequences within each trajectory
        train_trajectories.extend(trajectory)

    # Convert the list of trajectories to a numpy array
    train_trajectories = np.array(train_trajectories)

    # Copy all the first points
    first_points = [trajectory[0].copy() for trajectory in train_trajectories]
    
    # Remove the first points from all trajectories
    remaining_trajectories = [trajectory[1:] for trajectory in train_trajectories]
    remaining_trajectories = np.array(remaining_trajectories)
    
    # Extract labels from trajectories
    train_labels = [int(trajectory[0][0]) for trajectory in remaining_trajectories]
    remaining_trajectories = np.delete(remaining_trajectories, 0, axis=2)

    # Normalize and mean remove the remaining trajectories
    normalized_trajectories = normalize(remaining_trajectories)
    mean_removed_trajectories = mean_removed_all(normalized_trajectories)
    
    if augment:
        print("Augment")
        # Flip trajectories along the x-axis
        flipped_trajectories = flip_trajectories_x(mean_removed_trajectories)
        # Concatenate original and flipped trajectories along the first axis (num_trajectories)
        mean_removed_trajectories = np.concatenate((mean_removed_trajectories, flipped_trajectories), axis=0)
        train_labels += train_labels
        first_points += first_points
    
    # Add the first points back to the processed trajectories
    processed_trajectories = []
    for first_point, mean_removed_trajectory in zip(first_points, mean_removed_trajectories):
        processed_trajectory = np.vstack((first_point[1:], mean_removed_trajectory))
        processed_trajectories.append(processed_trajectory)
    
    train_trajectories = np.array(processed_trajectories)

    # Create a map of the expected ids to their labels with `id2label` and `label2id`
    with open(labels_path, "r") as stream:
        id2label = yaml.safe_load(stream)
    label2id = {v: k for k, v in id2label.items()}
    
    with open(label_detailed_path, "r") as stream:
        id2label_detailed = yaml.safe_load(stream)
    label_detailed2id = {v: k for k, v in id2label_detailed.items()}

    return train_trajectories, train_labels, id2label, label2id, id2label_detailed

def preprocess_function(examples):
    """
    Preprocesses trajectory data for BERT model input.

    Args:
        examples: A batch from a dataset containing 'trajectory' fields.

    Returns:
        A dictionary with tokenized input suitable for BERT, including attention masks.
    """
    exp = "Original"
    
    def format_trajectory(traj):
        # Format each point as (x, y, z) and join them with spaces
        return " ".join(f"({x}, {y}, {z})" for x, y, z in np.array(traj).reshape(-1, 3))

    # Get the drone model names
    drone_models = [id2label_detailed[int(traj[0][0])].split('|')[0] for traj in examples["trajectory"]]
    for traj in examples["trajectory"]:
        traj.pop(0)
    
    special_tokens = 2  # [CLS] and [SEP]

    if exp == "Original":
        # Construct detailed contextual sentences
        contextual_sentences = [
            f"This trajectory data is from a {model} drone flight."
            for model in drone_models
        ]
        
        # Tokenize the contextual sentences to determine their lengths
        context_token_lengths = [len(tokenizer.tokenize(sentence)) for sentence in contextual_sentences]
        max_context_length = max(context_token_lengths)

        # Calculate the available tokens for the trajectory data
        available_tokens_for_trajectory = 512 - max_context_length - special_tokens

        # Each coordinate point (x, y, z) will result in multiple tokens due to labeling and structure
        tokens_per_point = 10  # Estimate more tokens due to descriptive text
        max_trajectory_points = available_tokens_for_trajectory // tokens_per_point
        
        # Use the format_trajectory function to create the formatted trajectory strings
        trajectories_str = [
            f"{context_sentence} Trajectory: {format_trajectory(traj[:max_trajectory_points])}"
            for context_sentence, traj in zip(contextual_sentences, examples["trajectory"])
        ]

    tokenized_inputs = tokenizer(
        trajectories_str, 
        padding="max_length", 
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    )
    
    return tokenized_inputs

def compute_metrics(eval_pred):
    """
    Computes the accuracy of the model's predictions.

    Args:
        eval_pred: A tuple of logits and true labels for the evaluation dataset.

    Returns:
        A dictionary with the computed accuracy.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    new_df['pred_label'] = predictions
    new_df['true_label'] = val_labels
    return {"accuracy": (predictions == labels).mean()}
    
# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    debug = False

    if debug:
        # Debug configuration
        data_path = "/data/TGSSE/UpdatedIntentions/XYZ/800pad_66" 
        labels_path = "/data/TGSSE/UpdatedIntentions/labels.yaml"
        label_detailed_path = "/data/TGSSE/UpdatedIntentions/XYZ/800pad_66/trajectory_with_intentions_800_pad_533_min_151221_label_detailed.yaml"
        num_train_epochs = 75
        learning_rate = 1e-3
        per_device_eval_batch_size = 8
        per_device_train_batch_size = 8
        n_splits = 5
        weight_decay = 0.95
        project_name = 'trajectory_classification'
        run_name = 'mobilebert_run'
        augment = True
    else:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Flight Trajectory Classification')
        parser.add_argument('--data_path', type=str, help='Path to trajectory data: Directory containing .pt files')
        parser.add_argument('--labels_path', type=str, help='Path to labels data: .yaml file')
        parser.add_argument('--label_detailed_path', type=str, help='Path to detailed labels data: .yaml file')
        parser.add_argument('--num_train_epochs', type=int, default=75, help='Number of training epochs')
        parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size for training')
        parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Batch size for evaluation')
        parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for KFold cross-validation')
        parser.add_argument('--weight_decay', type=float, default=0.95, help='Weight decay')
        parser.add_argument('--project_name', type=str, default='trajectory_classification', help='Project name for wandb')
        parser.add_argument('--run_name', type=str, default='mobilebert_run', help='Run name for wandb')
        parser.add_argument('--augment', type=str, default='True', help='Whether or not to augment the data')
        args = parser.parse_args()
        # Definitions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_path = args.data_path
        labels_path = args.labels_path
        label_detailed_path = args.label_detailed_path
        num_train_epochs = args.num_train_epochs
        learning_rate = args.learning_rate
        per_device_train_batch_size = args.per_device_train_batch_size
        per_device_eval_batch_size = args.per_device_eval_batch_size
        n_splits = args.n_splits
        weight_decay = args.weight_decay
        project_name = args.project_name
        run_name = args.run_name
        augment = args.augment.lower() in ['true', '1', 't', 'y', 'yes']
        # Print configuration
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        
    # Load dataset
    flight_data, flight_labels, id2label, label2id, id2label_detailed = load_flight_data(data_path, labels_path, label_detailed_path, augment)
    num_labels = len(id2label)
    
    # Generate charts for the entire flight data
    generate_histogram_and_pie_chart(flight_labels, id2label, f'Overall_{run_name}_{augment}')
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Iterate over folds
    for fold_idx, (train_index, val_index) in enumerate(kf.split(flight_data)):
        # Create output directory
        output_dir = f"./results/{project_name}/{run_name}_{augment}_Fold{fold_idx}"
        new_df = pd.DataFrame()
        
        # Initialize a new wandb run for each fold
        wandb.init(project=project_name, group=f"{run_name}", name=f"{run_name}_Fold{fold_idx}")

        # Split data into train and validation sets
        train_trajectories, val_trajectories = flight_data[train_index], flight_data[val_index]
        train_labels, val_labels = np.array(flight_labels)[train_index], np.array(flight_labels)[val_index]
        
        # Generate charts for the split
        generate_histogram_and_pie_chart_for_split(train_labels, val_labels, id2label, f'{run_name}_Fold{fold_idx}')
        wandb.log({"Overall Distribution": wandb.Image(f'graphs/Overall_{run_name}_{augment}_overall_distribution.png')})
        wandb.log({"Split Distribution": wandb.Image(f'graphs/{run_name}_Fold{fold_idx}_split_distribution.png')})
        
        # Convert to Hugging Face datasets
        train_ds = Dataset.from_dict({"trajectory": train_trajectories.tolist(), "labels": train_labels})
        val_ds = Dataset.from_dict({"trajectory": val_trajectories.tolist(), "labels": val_labels})
        data_dict = DatasetDict({"train": train_ds, "test": val_ds})
        
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        
        # Fix tokenizer config
        tokenizer.model_max_length = 512
        
        # Apply preprocessing
        tokenized_data = data_dict.map(preprocess_function, batched=True)

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Define model
        model = MobileBertForSequenceClassification.from_pretrained(
            "google/mobilebert-uncased",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id 
        )  

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            logging_strategy="epoch",
        )

        # Trainer setup
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train the model
        results = trainer.train()
        my_table = wandb.Table(dataframe=new_df)
        wandb.log({f"Predictions for {run_name}_Fold{fold_idx}": my_table})
        
        # End the current wandb run
        wandb.finish()
