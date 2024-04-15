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
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import numpy as np
import yaml
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    MobileBertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from sklearn.model_selection import KFold
from data_tools.normalization import z_score_standardization, z_score_2d
import argparse
import wandb

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def load_flight_data(data_dir, labels_path):
    """
    Load flight trajectory data and associated labels for a binary classification task.

    Args:
        data_dir (str): Path to the directory containing trajectory data (.pt files).
        labels_path (str): Path to the file containing labels corresponding to trajectory data (.yaml).

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Training trajectories represented as 3-dimensional coordinates.
            - list: Binary labels corresponding to the training trajectories.
            - dict: Mapping of trajectory IDs to their labels.
            - dict: Mapping of labels to their corresponding IDs.
    """
    # Load training trajectories from .pt files in the directory
    train_trajectories = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.pt'):
            trajectory = torch.load(os.path.join(data_dir, file_name)).numpy()
            # Concatenate sequences within each trajectory
            train_trajectories.extend(trajectory)

    # Convert the list of trajectories to a numpy array
    train_trajectories = np.array(train_trajectories)

    # Extract labels from trajectories
    train_labels = [int(trajectory[0][0]) for trajectory in train_trajectories]
    train_trajectories = np.delete(train_trajectories, 0, axis=2)

    # Normalize the training trajectories
    train_trajectories = z_score_standardization(train_trajectories)

    # Create a map of the expected ids to their labels with `id2label` and `label2id`
    with open(labels_path, "r") as stream:
        id2label = yaml.safe_load(stream)
        stream.close()
    label2id = {v: k for k, v in id2label.items()}

    return train_trajectories, train_labels, id2label, label2id

def load_flight_data_XY(data_dir, labels_path):
    """
    Load flight trajectory data and associated labels for a binary classification task.

    Args:
        data_dir (str): Path to the directory containing trajectory data (.pt files).
        labels_path (str): Path to the file containing labels corresponding to trajectory data (.yaml).

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Training trajectories represented as 3-dimensional coordinates.
            - list: Binary labels corresponding to the training trajectories.
            - dict: Mapping of trajectory IDs to their labels.
            - dict: Mapping of labels to their corresponding IDs.
    """
    # Load training trajectories from .pt files in the directory
    train_trajectories = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.pt'):
            trajectory = torch.load(os.path.join(data_dir, file_name)).numpy()
            # Concatenate sequences within each trajectory
            train_trajectories.extend(trajectory)

    # Convert the list of trajectories to a numpy array
    train_trajectories = np.array(train_trajectories)

    # Extract labels from trajectories
    train_labels = [int(trajectory[0][0]) for trajectory in train_trajectories]
    train_trajectories = np.delete(train_trajectories, 0, axis=2)

    # Normalize the training trajectories
    train_trajectories = z_score_2d(train_trajectories)

    # Create a map of the expected ids to their labels with `id2label` and `label2id`
    with open(labels_path, "r") as stream:
        id2label = yaml.safe_load(stream)
        stream.close()
    label2id = {v: k for k, v in id2label.items()}

    return train_trajectories, train_labels, id2label, label2id

def preprocess_function(examples):
    """
    Preprocesses trajectory data for BERT model input.

    Args:
        examples: A batch from a dataset containing 'trajectory' fields.

    Returns:
        A dictionary with tokenized input suitable for BERT, including attention masks.
    """
    # Convert trajectories to strings
    trajectories_str = [" ".join(map(str, np.array(traj).flatten())) for traj in examples["trajectory"]]
    return tokenizer(trajectories_str, padding="max_length", truncation=True, max_length=512)

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
    return {"accuracy": (predictions == labels).mean()}

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flight Trajectory Classification')
    parser.add_argument('--data_dir', type=str, help='Path to trajectory data: Directory containing .pt files')
    parser.add_argument('--labels_path', type=str, help='Path to labels data: .yaml file')
    parser.add_argument('--num_train_epochs', type=int, default=75, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for KFold cross-validation')
    parser.add_argument('--weight_decay', type=float, default=0.95, help='Weight decay')
    parser.add_argument('--project_name', type=str, default='trajectory_classification', help='Project name for wandb')
    parser.add_argument('--run_name', type=str, default='mobilebert_run', help='Run name for wandb')
    args = parser.parse_args()
    
    # Definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    labels_path = args.labels_path
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    n_splits = args.n_splits
    weight_decay = args.weight_decay
    project_name = args.project_name
    run_name = args.run_name
    # Print configuration
    print(f'Data Path: {data_dir}')
    print(f'Labels Path: {labels_path}')
    print(f'Number of Training Epochs: {num_train_epochs}')
    print(f'Learning Rate: {learning_rate}')
    print(f'Batch Size for Training: {per_device_train_batch_size}')
    print(f'Batch Size for Evaluation: {per_device_eval_batch_size}')
    print(f'Number of Splits for KFold Cross-Validation: {n_splits}')
    print(f'Weight Decay: {weight_decay}')
    print(f'Project Name: {project_name}')
    print(f'Run Name: {run_name}')

    # Load dataset
    flight_data, flight_labels, id2label, label2id = load_flight_data(data_dir, labels_path)
    # flight_data, flight_labels, id2label, label2id = load_flight_data_XY(data_dir, labels_path)
    num_labels = len(id2label)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Iterate over folds
    for fold_idx, (train_index, val_index) in enumerate(kf.split(flight_data)):
        # Create output directory
        output_dir = f"./results/{project_name}/{run_name}_Fold{fold_idx}"
        
        # Initialize a new wandb run for each fold
        wandb.init(project=project_name, name=f"{run_name}_Fold{fold_idx}")

        # Split data into train and validation sets
        train_trajectories, val_trajectories = flight_data[train_index], flight_data[val_index]
        train_labels, val_labels = np.array(flight_labels)[train_index], np.array(flight_labels)[val_index]

        # Convert to Hugging Face datasets
        train_ds = Dataset.from_dict({"trajectory": train_trajectories.tolist(), "labels": train_labels})
        val_ds = Dataset.from_dict({"trajectory": val_trajectories.tolist(), "labels": val_labels})
        data_dict = DatasetDict({"train": train_ds, "test": val_ds})
        
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        
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
        trainer.train()
        
        # End the current wandb run
        wandb.finish()
