import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Define custom dataset class
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, labels, task_names):
        self.trajectories = torch.tensor(trajectories, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.task_names = task_names

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return {
            'trajectory': self.trajectories[idx],
            'label': self.labels[idx],
            'task_name': self.task_names[idx]
        }

# Prepare DataLoader function
def prepare_dataloader(trajectories, labels, task_names, batch_size=32, shuffle=True):
    dataset = TrajectoryDataset(trajectories, labels, task_names)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Create example data for task1
num_samples_task1 = 500
seq_len = 100
input_dim = 3
trajectories_task1 = np.random.randn(num_samples_task1, seq_len, input_dim)
labels_task1 = np.random.randint(0, 4, size=num_samples_task1)
task_names_task1 = ['task1'] * num_samples_task1

# Create example data for task2
num_samples_task2 = 500
trajectories_task2 = np.random.randn(num_samples_task2, seq_len, input_dim)
labels_task2 = np.random.randint(0, 3, size=num_samples_task2)
task_names_task2 = ['task2'] * num_samples_task2

# Combine data from both tasks
trajectories = np.concatenate((trajectories_task1, trajectories_task2))
labels = np.concatenate((labels_task1, labels_task2))
task_names = task_names_task1 + task_names_task2

# Shuffle the combined data
combined = list(zip(trajectories, labels, task_names))
np.random.shuffle(combined)
trajectories, labels, task_names = zip(*combined)
trajectories = np.array(trajectories)
labels = np.array(labels)
task_names = list(task_names)

# Split data into training and validation sets
train_size = int(0.8 * len(trajectories))
val_size = len(trajectories) - train_size
train_trajectories = trajectories[:train_size]
val_trajectories = trajectories[train_size:]
train_labels = labels[:train_size]
val_labels = labels[train_size:]
train_task_names = task_names[:train_size]
val_task_names = task_names[train_size:]

# Ensure that labels for each task are within the valid range
for idx, task_name in enumerate(train_task_names):
    if task_name == 'task2':
        assert train_labels[idx] < 3, f"Invalid label {train_labels[idx]} for task2 in training data"

for idx, task_name in enumerate(val_task_names):
    if task_name == 'task2':
        assert val_labels[idx] < 3, f"Invalid label {val_labels[idx]} for task2 in validation data"

# Print debug information
print("Data after shuffling and splitting:")
print("Train trajectories shape:", train_trajectories.shape)
print("Train labels shape:", train_labels.shape)
print("Validation trajectories shape:", val_trajectories.shape)
print("Validation labels shape:", val_labels.shape)

class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=8):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=1)
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

class MultitaskCNNModel(nn.Module):
    def __init__(self, input_dim, task_output_dict, kernel_size=8):
        super(MultitaskCNNModel, self).__init__()
        self.shared_encoder = CNNModel(input_dim, 128, kernel_size)
        self.taskmodels_dict = nn.ModuleDict({
            task_name: nn.Linear(128, output_dim)
            for task_name, output_dim in task_output_dict.items()
        })

    def forward(self, task_name, x):
        shared_output = self.shared_encoder(x)
        task_output = self.taskmodels_dict[task_name](shared_output)
        return task_output

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, fold, task_output_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs = batch['trajectory'].to(device)
            labels = batch['label'].to(device)
            task_name = batch['task_name'][0]

            optimizer.zero_grad()
            outputs = model(task_name, inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, task_output_dict)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

def evaluate_model(model, data_loader, criterion, task_output_dict):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['trajectory'].to(device)
            labels = batch['label'].to(device)
            task_name = batch['task_name'][0]

            outputs = model(task_name, inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(data_loader), accuracy

if __name__ == "__main__":
    # Prepare data loaders
    batch_size = 32
    train_loader = prepare_dataloader(train_trajectories, train_labels, train_task_names, batch_size=batch_size, shuffle=True)
    val_loader = prepare_dataloader(val_trajectories, val_labels, val_task_names, batch_size=batch_size, shuffle=False)

    # Define task output dimensions
    task_output_dict = {
        'task1': 4,
        'task2': 3
    }

    # Initialize model, loss function, and optimizer
    input_dim = train_trajectories.shape[2]
    model = MultitaskCNNModel(input_dim, task_output_dict, kernel_size=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, fold=1, task_output_dict=task_output_dict)
