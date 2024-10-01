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
settings for GPU use, then generates a synthetic dataset of flight trajectories with binary labels. The data is
prepared by converting the trajectories into a text-like format suitable for BERT. The model is trained with custom
arguments, leveraging a Trainer object for efficient training and evaluation. Accuracy is used as the evaluation metric.
The script concludes by demonstrating inference on a new trajectory, showcasing the end-to-end process.
"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# This needs to be done before other imports
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Standard Python imports
import argparse

# Third-party imports
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

# Custom file imports
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

def load_flight_data(data_path: str, labels_path: str, label_detailed_path: str, augment: bool = False) -> tuple:
    """
    Load flight trajectory data and associated labels for a binary classification task.

    Args:
        data_path (str): Path to the directory containing trajectory data (.pt files).
        labels_path (str): Path to the file containing labels (.yaml).
        label_detailed_path (str): Path to the file containing detailed labels (.yaml).
        augment (bool, optional): Whether to augment the data by flipping trajectories. Defaults to False.

    Returns:
        tuple: Contains training trajectories, binary labels, and mappings of trajectory IDs to their labels.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory '{data_path}' does not exist.")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file '{labels_path}' does not exist.")
    
    if not os.path.exists(label_detailed_path):
        raise FileNotFoundError(f"Detailed labels file '{label_detailed_path}' does not exist.")
    
    # Load trajectory data
    trajectory_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')]
    train_trajectories = [torch.load(f).numpy() for f in trajectory_files]
    train_trajectories = np.array(train_trajectories)

    first_points = [traj[0].copy() for traj in train_trajectories]
    remaining_trajectories = [traj[1:] for traj in train_trajectories]
    
    train_labels = [int(traj[0][0]) for traj in remaining_trajectories]
    remaining_trajectories = np.delete(np.array(remaining_trajectories), 0, axis=2)

    normalized_trajectories = normalize(remaining_trajectories)
    mean_removed_trajectories = mean_removed_all(normalized_trajectories)
    
    if augment:
        flipped_trajectories = flip_trajectories_x(mean_removed_trajectories)
        mean_removed_trajectories = np.concatenate((mean_removed_trajectories, flipped_trajectories), axis=0)
        train_labels += train_labels
        first_points += first_points
    
    processed_trajectories = [
        np.vstack((fp[1:], mrt)) for fp, mrt in zip(first_points, mean_removed_trajectories)
    ]
    train_trajectories = np.array(processed_trajectories)

    with open(labels_path, "r") as stream:
        id2label = yaml.safe_load(stream)
    label2id = {v: k for k, v in id2label.items()}
    
    with open(label_detailed_path, "r") as stream:
        id2label_detailed = yaml.safe_load(stream)

    return train_trajectories, train_labels, id2label, label2id, id2label_detailed

def preprocess_function(examples):
    """
    Preprocesses trajectory data for BERT model input.

    Args:
        examples: Batch from a dataset containing 'trajectory' fields.

    Returns:
        dict: Tokenized input suitable for BERT, including attention masks.
    """
    def format_trajectory(traj):
        return " ".join(f"({x}, {y}, {z})" for x, y, z in np.array(traj).reshape(-1, 3))

    drone_models = [id2label_detailed[int(traj[0][0])].split('|')[0] for traj in examples["trajectory"]]
    for traj in examples["trajectory"]:
        traj.pop(0)
    
    special_tokens = 2  
    contextual_sentences = [
        f"This trajectory data is from a {model} drone flight." for model in drone_models
    ]
    context_token_lengths = [len(tokenizer.tokenize(sentence)) for sentence in contextual_sentences]
    max_context_length = max(context_token_lengths)
    available_tokens_for_trajectory = 512 - max_context_length - special_tokens
    tokens_per_point = 10  
    max_trajectory_points = available_tokens_for_trajectory // tokens_per_point
    
    trajectories_str = [
        f"{context_sentence} Trajectory: {format_trajectory(traj[:max_trajectory_points])}"
        for context_sentence, traj in zip(contextual_sentences, examples["trajectory"])
    ]

    return tokenizer(trajectories_str, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

def compute_metrics(eval_pred):
    """
    Computes the accuracy of the model's predictions.

    Args:
        eval_pred: Tuple of logits and true labels for evaluation.

    Returns:
        dict: Computed accuracy metric.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    debug = False

    if debug:
        data_path = "IntentCNN/Useable/XYZ/800pad_66" 
        labels_path = "IntentCNN/Useable/intent_labels.yaml"
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
        parser = argparse.ArgumentParser(description='Flight Trajectory Classification')
        parser.add_argument('--data_path', type=str, help='Path to trajectory data (.pt files)')
        parser.add_argument('--labels_path', type=str, help='Path to labels data (.yaml)')
        parser.add_argument('--label_detailed_path', type=str, help='Path to detailed labels data (.yaml)')
        parser.add_argument('--num_train_epochs', type=int, default=75, help='Number of training epochs')
        parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size for training')
        parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Batch size for evaluation')
        parser.add_argument('--n_splits', type=int, default=5, help='Number of KFold splits')
        parser.add_argument('--weight_decay', type=float, default=0.95, help='Weight decay')
        parser.add_argument('--project_name', type=str, default='trajectory_classification', help='Project name for wandb')
        parser.add_argument('--run_name', type=str, default='mobilebert_run', help='Run name for wandb')
        parser.add_argument('--augment', type=str, default='True', help='Whether to augment the data')
        args = parser.parse_args()
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

    flight_data, flight_labels, id2label, label2id, id2label_detailed = load_flight_data(data_path, labels_path, label_detailed_path, augment)
    num_labels = len(id2label)
    
    generate_histogram_and_pie_chart(flight_labels, id2label, f'Overall_{run_name}_{augment}')
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold_idx, (train_index, val_index) in enumerate(kf.split(flight_data)):
        output_dir = f"./results/{project_name}/{run_name}_{augment}_Fold{fold_idx}"
        new_df = pd.DataFrame()
        
        wandb.init(project=project_name, group=f"{run_name}", name=f"{run_name}_Fold{fold_idx}")

        train_trajectories, val_trajectories = flight_data[train_index], flight_data[val_index]
        train_labels, val_labels = np.array(flight_labels)[train_index], np.array(flight_labels)[val_index]
        
        generate_histogram_and_pie_chart_for_split(train_labels, val_labels, id2label, f'{run_name}_Fold{fold_idx}')
        wandb.log({"Overall Distribution": wandb.Image(f'graphs/Overall_{run_name}_{augment}_overall_distribution.png')})
        wandb.log({"Split Distribution": wandb.Image(f'graphs/{run_name}_Fold{fold_idx}_split_distribution.png')})
        
        train_ds = Dataset.from_dict({"trajectory": train_trajectories.tolist(), "labels": train_labels})
        val_ds = Dataset.from_dict({"trajectory": val_trajectories.tolist(), "labels": val_labels})
        data_dict = DatasetDict({"train": train_ds, "test": val_ds})
        
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        tokenizer.model_max_length = 512
        tokenized_data = data_dict.map(preprocess_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        model = MobileBertForSequenceClassification.from_pretrained(
            "google/mobilebert-uncased",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id 
        )  

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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        results = trainer.train()
        my_table = wandb.Table(dataframe=new_df)
        wandb.log({f"Predictions for {run_name}_Fold{fold_idx}": my_table})
        
        wandb.finish()
