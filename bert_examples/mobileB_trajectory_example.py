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

# Imports
#---------------------------------------------------------------------------#

import torch
import numpy as np

# This needs to be done before other imports
# *********************************************************************

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *********************************************************************

import evaluate
from datasets import Dataset, DatasetDict
from transformers import (AutoModelForSequenceClassification, MobileBertForSequenceClassification,
                          AutoTokenizer, 
                          TrainingArguments, Trainer, DataCollatorWithPadding)

# Definitions
#---------------------------------------------------------------------------------------#
num_time_steps = 10
num_train_examples = 100
num_test_examples = 20
num_train_epochs = 16
learning_rate = 2e-6
weight_decay = 0.02
per_device_train_batch_size = 16
per_device_eval_batch_size = 16

#---------------------------------------------------------------------------#


# Functions
#---------------------------------------------------------------------------------------#

def load_flight_data():
    """
    Generates synthetic flight trajectory data for a binary classification task.

    Returns:
        A DatasetDict containing two subsets: 'train' and 'test', each with 
        trajectories represented as 3-dimensional coordinates and binary labels.
    """

    # Example data: 100 training examples, each with 10 time steps and 3D coordinates
    train_trajectories = np.random.rand(num_train_examples, num_time_steps, 3)
    train_labels = np.random.randint(0, 2, size=num_train_examples)

    # 20 test examples
    test_trajectories = np.random.rand(num_test_examples, num_time_steps, 3)
    test_labels = np.random.randint(0, 2, size=num_test_examples)
    
    # Converting to Hugging Face datasets
    train_ds = Dataset.from_dict({"trajectory": train_trajectories.tolist(), "labels": train_labels})
    test_ds = Dataset.from_dict({"trajectory": test_trajectories.tolist(), "labels": test_labels})
    
    return DatasetDict({"train": train_ds, "test": test_ds})


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
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

def create_output_dir(output_dir):
    """
    Create a new output directory if the current doesn't exist. Otherwise, increment the run number.

    Args:
        output_dir(str): The output directory path.

    Returns:
        str: The new output directory path.
    """
    if os.path.exists(output_dir):
        run_num = int(output_dir[len(output_dir) - 1])
        output_dir = output_dir[:-1]
        output_dir += str(run_num + 1)
        create_output_dir(output_dir)
    return output_dir


#---------------------------------------------------------------------------------------#

# Load dataset
flight_data = load_flight_data()

# Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")          # BERT
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")  #MobileBERT

# Apply preprocessing
tokenized_data = flight_data.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load accuracy metric
accuracy = evaluate.load("accuracy")

# Create a map of the expected ids to their labels with `id2label` and `label2id`
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Define model
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)             # BERT
model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased",num_labels=2,
                                                                                        id2label=id2label,
                                                                                        label2id=label2id )  # MobileBERT
# model.to(device)

output_dir = "./results/mobilebert_" + str(num_train_examples) + "exs_" + str(num_time_steps) + "timesteps_run0"
output_dir = create_output_dir(output_dir)

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
    push_to_hub=True,
    hub_token='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'
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

## Push the model to HuggingFace (optional)
trainer.push_to_hub()

# Perform prediction on a new example
new_trajectory = np.random.rand(1, num_time_steps, 3).tolist()
new_trajectory_str = [" ".join(map(str, np.array(new_trajectory).flatten()))]
inputs = tokenizer(new_trajectory_str, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()} # Move the tokenized inputs to the same device as the model

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

print(f"Predicted class: {predicted_class.item()}")