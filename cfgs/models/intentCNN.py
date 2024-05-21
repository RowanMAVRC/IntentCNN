# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Package Imports
import os
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import wandb
import numpy as np

# ------------------------------------------------------------------------------------- #
# Functions & Definitions
# ------------------------------------------------------------------------------------- #

class CNNModel(nn.Module):
    """
    CNN Model for flight trajectory classification.

    Args:
        input_dim (int): Number of input features (dimensions of the trajectory).
        output_dim (int): Number of output classes (intentions).

    Methods:
        forward(x): Forward pass of the model.
    """
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, fold, model_name):
    """
    Trains the model and logs training progress to Wandb.

    Args:
        model (nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        optimizer (Optimizer): Optimizer for updating model parameters.
        num_epochs (int): Number of training epochs.
        fold (int): Current fold number for cross-validation.
        model_name (str): Name of the model.

    Returns:
        nn.Module: Trained model.
    """
    with tqdm(total=num_epochs, desc=f"Training Fold {fold}") as pbar:
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
            pbar.set_postfix({
                'Epoch': f'{epoch+1}/{num_epochs}',
                'Train Loss': f'{epoch_loss/len(train_loader):.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val Acc': f'{val_accuracy:.4f}'
            })
            pbar.update(1)
            wandb.log({
                f"train_loss": epoch_loss/len(train_loader),
                f"val_loss": val_loss,
                f"val_accuracy": val_accuracy
            })
    return model

# Define the function to evaluate the model
def evaluate_model(model, data_loader, criterion):
    """
    Evaluates the model on the validation data.

    Args:
        model (nn.Module): The trained CNN model.
        data_loader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.

    Returns:
        tuple: (Average validation loss, Validation accuracy)
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(data_loader), accuracy

# Define the function to prepare the data loader
def prepare_dataloader(trajectories, labels, batch_size=32, shuffle=True):
    """
    Prepares the data loader for training and validation datasets.

    Args:
        trajectories (numpy.ndarray): Array of trajectories.
        labels (list): List of labels corresponding to the trajectories.
        batch_size (int, optional): Batch size for the data loader. Defaults to 32.
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Define the function to train the CNN model
def train_cnn(train_trajectories, train_labels, val_trajectories, val_labels, fold, model_name, num_epochs=10, batch_size=32):
    """
    Trains the CNN model with the provided data and cross-validation fold.

    Args:
        train_trajectories (numpy.ndarray): Training trajectories.
        train_labels (list): Labels for the training trajectories.
        val_trajectories (numpy.ndarray): Validation trajectories.
        val_labels (list): Labels for the validation trajectories.
        fold (int): Current fold number for cross-validation.
        model_name (str): Name of the model.

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
    
    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

    # Initialize the model, criterion, and optimizer
    model = CNNModel(input_dim, output_dim).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, fold=fold, model_name=model_name)
    
    # Save the trained model
    model_save_path = f"trained_models/{model_name}_fold_{fold}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    print(f"Finished training CNN model for fold {fold}.")
    return model

# Define the function for inference
def inference(model, data_loader):
    """
    Performs inference on the validation data.

    Args:
        model (nn.Module): Trained CNN model.
        data_loader (DataLoader): DataLoader for the validation data.

    Returns:
        list: Predictions for the validation data.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds

def load_model(model_class, model_path, input_dim, output_dim):
    """
    Loads a model from the specified path.

    Args:
        model_class (type): The class of the model to be loaded.
        model_path (str): Path to the saved model state dictionary.
        input_dim (int): Number of input features (dimensions of the trajectory).
        output_dim (int): Number of output classes (intentions).

    Returns:
        nn.Module: Loaded model.
    """
    model = model_class(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model