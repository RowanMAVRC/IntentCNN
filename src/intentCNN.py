"""
 _____       _             _                           
|_   _|     | |           | |    ____  _   _  _   _                           
  | |  _ __ | |_ ___ _ __ | |_  / ___|| \ | || \ | |
  | | | '_ \| __/ _ \ '_ \| __|| |   ||  \| ||  \| |
 _| |_| | | | ||  __/ | | | |_ | |___|| |\  || |\  |
|_____|_| |_|\__\___|_| |_|\__| \____||_| \_||_| \_|

## Summary
This script sets up an environment for training a CNN-based deep learning model to classify flight trajectories.
It utilizes PyTorch, sklearn, and Wandb for cross-validation and tracking. The data consists of flight trajectories,
and the model predicts the intention of the object based on these trajectories. The script includes functions for 
loading and preprocessing data, defining the model architecture, training the model, evaluating performance, and 
running inference. It includes a multi-head CNN model (IntentCNN) for aircraft classification based on trajectory data.
"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Standard library imports
import os

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import wandb

# ------------------------------------------------------------------------------------- #
# Classes
# ------------------------------------------------------------------------------------- #

class CNNModel(nn.Module):
    """
    CNN Model for flight trajectory classification.

    Args:
        input_dim (int): Number of input features (dimensions of the trajectory).
        output_dim (int): Number of output classes (intentions).
        kernel_size (int, optional): Size of the convolutional kernels. Defaults to 8.
    """
    def __init__(self, input_dim, output_dim, kernel_size=8):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_dim, seq_len)
        x = torch.relu(self.conv1(x))  # Apply first convolution and ReLU
        x = torch.relu(self.conv2(x))  # Apply second convolution and ReLU
        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)  # Apply fully connected layer
        return x
    
class MultiHeadCNNModel(nn.Module):
    """
    Multi-Head CNN Model for flight trajectory classification with hard parameter sharing.

    Args:
        input_dim (int): Number of input features (dimensions of the trajectory).
        heads_info (dict): A dictionary with aircraft identifiers as keys and the number of unique intention
                           classes as values for each aircraft.
        kernel_size (int, optional): Size of the convolutional kernels. Defaults to 8.
    """
    def __init__(self, input_dim, heads_info, kernel_size=8):
        super(MultiHeadCNNModel, self).__init__()
        # Shared convolutional layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=1)
        self.dropout = nn.Dropout(0.5)
        
        # Create a dictionary of fully connected layers for each aircraft
        self.heads_info = heads_info
        self.heads = nn.ModuleDict({
            aircraft: nn.Linear(128, num_classes) for aircraft, num_classes in heads_info.items()
        })

    def forward(self, x, aircraft_id):
        """
        Forward pass of the Multi-Head CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            aircraft_id (str): Aircraft identifier to select the appropriate classification head.

        Returns:
            torch.Tensor: Output tensor from the selected head.
        """
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_dim, seq_len)
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU
        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = self.dropout(x)  # Apply dropout
        output = self.heads[aircraft_id](x)  # Apply the specific fully connected layer for the aircraft
        return output

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, fold, device, id2label, global2local_label_map):
    """
    Trains the model and logs training progress to WandB.

    Args:
        model (nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        optimizer (Optimizer): Optimizer for updating model parameters.
        num_epochs (int): Number of training epochs.
        fold (int): Current fold number for cross-validation.
        device (torch.device): The device to run the training on.

    Returns:
        nn.Module: Trained model.
    """

    with tqdm(total=num_epochs, desc=f"Training Fold {fold}") as pbar:
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                masks = {
                    "FIXEDWING" :   (labels == 0) | (labels == 1), # Can be seen with print(id2label)
                    "ROTARY"    : ~((labels == 0) | (labels == 1))
                }


                optimizer.zero_grad()  # Reset gradients
                loss = 0
                for i, aircraft_id in enumerate(model.heads_info.keys()):
                    local_labels = torch.tensor(
                        [global2local_label_map[label.item()] for label in labels[masks[aircraft_id]]], 
                        device=labels.device,
                        dtype=torch.long  # Ensure local_labels are of type Long
                    )
                 
                    outputs = model(inputs[masks[aircraft_id]], aircraft_id)  # Forward pass for each aircraft
                    loss += criterion(outputs, local_labels)  # Compute loss
               
                loss /= labels.size(0)

                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters

                epoch_loss += loss.item()  # Accumulate loss

            # Evaluate model on validation data
            val_loss, val_accuracy = evaluate_model(
                model, 
                val_loader, 
                criterion, 
                device, 
                id2label,
                global2local_label_map
            )
            
            # Update progress bar and WandB logs
            pbar.set_postfix({
                'Epoch': f'{epoch+1}/{num_epochs}',
                'Train Loss': f'{epoch_loss/len(train_loader):.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val Acc': f'{val_accuracy:.4f}'
            })
            pbar.update(1)
            wandb.log({
                "train_loss": epoch_loss/len(train_loader),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
    return model



def evaluate_model(model, data_loader, criterion, device, id2label, global2local_label_map):
    """
    Evaluates the model on the validation data.

    Args:
        model (nn.Module): The trained CNN model.
        data_loader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): The device to run the evaluation on.

    Returns:
        tuple: (Average validation loss, Validation accuracy)
    """
    
    def invert_dict(d):
        return {v: [k for k in d if d[k] == v] for v in set(d.values())}

    # Split and invert the dictionaries
    local2global_label_map = {
        "FIXEDWING" : invert_dict({k: global2local_label_map[k] for k in list(global2local_label_map)[:2]}),
        "ROTARY" : invert_dict({k: global2local_label_map[k] for k in list(global2local_label_map)[2:]})
    }

    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            masks = {"FIXEDWING" : (labels == 0) | (labels == 1), # Can be seen with print(id2label)
                     "ROTARY" : ~((labels == 0) | (labels == 1))}

            loss = 0
            global_preds = torch.empty_like(labels, dtype=torch.long)  # Ensure dtype is long
            for i, aircraft_id in enumerate(model.heads_info.keys()):
                local_labels = torch.tensor([global2local_label_map[label.item()] for label in labels[masks[aircraft_id]]], device=labels.device, dtype=torch.long)
                
                outputs = model(inputs[masks[aircraft_id]], aircraft_id)  # Forward pass for each aircraft
                loss += criterion(outputs, local_labels)  # Compute loss

                _, local_preds = torch.max(outputs, 1)  # Get predictions
                global_preds[masks[aircraft_id]] = torch.tensor(
                    [local2global_label_map[aircraft_id][local_pred.item()] for local_pred in local_preds], 
                    device=labels.device, dtype=torch.long
                ).squeeze()
            
            loss /= labels.size(0)

            total_loss += loss.item()  # Accumulate loss
            
            all_preds.extend(global_preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(data_loader), accuracy




def prepare_dataloader(trajectories, labels, batch_size=32, shuffle=True):
    """
    Prepares the DataLoader for training and validation datasets.

    Args:
        trajectories (numpy.ndarray): Array of trajectories.
        labels (list): List of labels corresponding to the trajectories.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    class FlightDataset(torch.utils.data.Dataset):
        def __init__(self, trajectories, labels):
            self.trajectories = torch.tensor(trajectories, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.trajectories)

        def __getitem__(self, idx):
            return self.trajectories[idx], self.labels[idx]

    dataset = FlightDataset(trajectories, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_cnn(train_trajectories, train_labels, val_trajectories, val_labels, fold, model_name, 
              device, id2label, lr=0.001, num_epochs=10, batch_size=32, kernel_size=8,):
    """
    Trains the CNN model with the provided data and cross-validation fold.

    Args:
        train_trajectories (numpy.ndarray): Training trajectories.
        train_labels (list): Labels for the training trajectories.
        val_trajectories (numpy.ndarray): Validation trajectories.
        val_labels (list): Labels for the validation trajectories.
        fold (int): Current fold number for cross-validation.
        model_name (str): Name of the model.
        device (torch.device): The device to run the training on.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        kernel_size (int, optional): Kernel size for CNN layers. Defaults to 8.

    Returns:
        nn.Module: Trained CNN model.
    """
    print(f"Training CNN model for fold {fold}...")

    # Prepare data loaders for training and validation sets
    train_loader = prepare_dataloader(train_trajectories, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = prepare_dataloader(val_trajectories, val_labels, batch_size=batch_size, shuffle=True)

    # Define input and output dimensions
    input_dim = train_trajectories.shape[2]
    output_dim = len(np.unique(train_labels))

    # Initialize the model, criterion, and optimizer
    # model = CNNModel(input_dim, output_dim, kernel_size).to(device)
    global2local_label_map = {
        0 : 0,  # 0: 'FIXEDWING - Kamikaze' -> 0
        1 : 1,  # 1: 'FIXEDWING - Recon'    -> 1

        2 : 0,  # 2: 'ROTARY - Area Denial' -> 0
        3 : 1,  # 3: 'ROTARY - Recon'       -> 1
        4 : 2   # 4: 'ROTARY - Travel'      -> 2
    }

    heads_info = {
        'FIXEDWING': 2,
        'ROTARY': 3,
    }
    model = MultiHeadCNNModel(input_dim=2, heads_info=heads_info).to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=num_epochs, 
        fold=fold, 
        device=device, 
        id2label=id2label, 
        global2local_label_map=global2local_label_map
    )

    # Save the trained model
    model_save_path = f"trained_models/{model_name}/fold_{fold}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    print(f"Finished training CNN model for fold {fold}.")
    return model

def inference(model, data_loader, device):
    """
    Performs inference on the validation data.

    Args:
        model (nn.Module): Trained CNN model.
        data_loader (DataLoader): DataLoader for the validation data.
        device (torch.device): The device to run the inference on.

    Returns:
        list: Predictions for the validation data.
    """

    global2local_label_map = {
        0 : 0,  # 0: 'FIXEDWING - Kamikaze' -> 0
        1 : 1,  # 1: 'FIXEDWING - Recon'    -> 1
        2 : 0,  # 2: 'ROTARY - Area Denial' -> 0
        3 : 1,  # 3: 'ROTARY - Recon'       -> 1
        4 : 2   # 4: 'ROTARY - Travel'      -> 2
    }

    def invert_dict(d):
        return {v: [k for k in d if d[k] == v] for v in set(d.values())}

    # Split and invert the dictionaries
    local2global_label_map = {
        "FIXEDWING" : invert_dict({k: global2local_label_map[k] for k in list(global2local_label_map)[:2]}),
        "ROTARY" : invert_dict({k: global2local_label_map[k] for k in list(global2local_label_map)[2:]})
    }

    model.eval()  # Set model to evaluation mode
    all_preds = []
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            masks = {"FIXEDWING" : (labels == 0) | (labels == 1), # Can be seen with print(id2label)
                     "ROTARY" : ~((labels == 0) | (labels == 1))}

            global_preds = torch.zeros_like(labels, dtype=torch.long)  # Ensure dtype is long
            for i, aircraft_id in enumerate(model.heads_info.keys()):
                outputs = model(inputs[masks[aircraft_id]], aircraft_id)  

                _, local_preds = torch.max(outputs, 1)  # Get predictions
                
                global_preds[masks[aircraft_id]] = torch.tensor(
                        [local2global_label_map[aircraft_id][local_pred.item()] for local_pred in local_preds], 
                        device=labels.device,
                        dtype=torch.long  # Ensure dtype is long
                    ).squeeze()

            all_preds.extend(global_preds.cpu().numpy())

    return all_preds



def load_model(model_class, model_path, input_dim, device, heads_info):
    """
    Loads a model from the specified path.

    Args:
        model_class (type): The class of the model to be loaded.
        model_path (str): Path to the saved model state dictionary.
        input_dim (int): Number of input features (dimensions of the trajectory).
        output_dim (int): Number of output classes (intentions).
        device (torch.device): The device to load the model on.

    Returns:
        nn.Module: Loaded model.
    """
    model = model_class(input_dim, heads_info)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

def predict(model, input_data, aircraft_id, device, local2global_label_map):
    """
    Performs prediction on new data using the trained model.

    Args:
        model (nn.Module): Trained CNN model.
        input_data (numpy.ndarray): New input data for prediction.
        device (torch.device): The device to run the prediction on.

    Returns:
        numpy.ndarray: Predictions for the new data.
    """
    model.eval()  # Set model to evaluation mode
    
    # Convert input data to a PyTorch tensor and create a DataLoader
    inputs = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    with torch.no_grad():  # Disable gradient computation
        outputs = model(inputs, aircraft_id)  # Forward pass

        _, local_preds = torch.max(outputs, 1)  # Get predictions
        global_preds = torch.tensor(
                [local2global_label_map[aircraft_id][local_pred.item()] for local_pred in local_preds], 
                device=device
            ).squeeze()
    
    return global_preds.item()
