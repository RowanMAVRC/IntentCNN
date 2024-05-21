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
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, DataCollatorWithPadding, MobileBertForSequenceClassification, TrainingArguments, Trainer
import evaluate
from datasets import Dataset, DatasetDict
import wandb

# File Imports
from data_tools.chart_generator import generate_histogram_and_pie_chart, generate_histogram_and_pie_chart_for_split
from data_tools.normalization import z_score_standardization_all, z_score_standardization_single, mean_removed_all, mean_removed_single, compute_trajectory_stats, normalize
from data_tools.augmentations import flip_trajectories_x, augment_with_jitters

# Define Model Architectures
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, output_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch_size, input_dim, sequence_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Define Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, fold, model_name):
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        wandb.log({f"train_loss_{model_name}": epoch_loss/len(train_loader),
                   f"val_loss_{model_name}": val_loss,
                   f"val_accuracy_{model_name}": val_accuracy})
    return model

# Define Evaluation Function
def evaluate_model(model, data_loader, criterion):
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

# Define Data Loading Function
def load_flight_data(data_path: str, labels_path: str, label_detailed_path: str, augment: bool = False) -> tuple:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory '{data_path}' does not exist.")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file '{labels_path}' does not exist.")
    
    if not os.path.exists(label_detailed_path):
        raise FileNotFoundError(f"Labels file '{label_detailed_path}' does not exist.")
    
    trajectory_files = [data_path] if os.path.isfile(data_path) else [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('.pt')]

    train_trajectories = []
    for file_name in trajectory_files:
        trajectory = torch.load(file_name).numpy()
        train_trajectories.extend(trajectory)

    train_trajectories = np.array(train_trajectories)

    first_points = [trajectory[0].copy() for trajectory in train_trajectories]
    remaining_trajectories = [trajectory[1:] for trajectory in train_trajectories]
    remaining_trajectories = np.array(remaining_trajectories)

    train_labels = [int(trajectory[0][0]) for trajectory in remaining_trajectories]
    remaining_trajectories = np.delete(remaining_trajectories, 0, axis=2)

    normalized_trajectories = normalize(remaining_trajectories)
    mean_removed_trajectories = mean_removed_all(normalized_trajectories)
    
    if augment:
        flipped_trajectories = flip_trajectories_x(mean_removed_trajectories)
        mean_removed_trajectories = np.concatenate((mean_removed_trajectories, flipped_trajectories), axis=0)
        train_labels += train_labels
        first_points += first_points
    
    processed_trajectories = []
    for first_point, mean_removed_trajectory in zip(first_points, mean_removed_trajectories):
        processed_trajectory = np.vstack((first_point[1:], mean_removed_trajectory))
        processed_trajectories.append(processed_trajectory)
    
    train_trajectories = np.array(processed_trajectories)

    with open(labels_path, "r") as stream:
        id2label = yaml.safe_load(stream)
    label2id = {v: k for k, v in id2label.items()}
    
    with open(label_detailed_path, "r") as stream:
        id2label_detailed = yaml.safe_load(stream)
    label_detailed2id = {v: k for k, v in id2label_detailed.items()}

    return train_trajectories, train_labels, id2label, label2id, id2label_detailed

def prepare_dataloader(trajectories, labels, batch_size=32, shuffle=True):
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

def preprocess_function(examples, id2label_detailed, tokenizer):
    def format_trajectory(traj):
        return " ".join(f"({x}, {y}, {z})" for x, y, z in np.array(traj).reshape(-1, 3))

    drone_models = [id2label_detailed[int(traj[0][0])].split('|')[0] for traj in examples["trajectory"]]
    for traj in examples["trajectory"]:
        traj.pop(0)
    
    special_tokens = 2  # [CLS] and [SEP]

    # Construct detailed contextual sentences
    contextual_sentences = [f"This trajectory data is from a {model} drone flight." for model in drone_models]
    
    context_token_lengths = [len(tokenizer.tokenize(sentence)) for sentence in contextual_sentences]
    max_context_length = max(context_token_lengths)

    available_tokens_for_trajectory = 512 - max_context_length - special_tokens
    tokens_per_point = 10  # Estimate more tokens due to descriptive text
    max_trajectory_points = available_tokens_for_trajectory // tokens_per_point
    
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
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

def train_lstm(train_trajectories, train_labels, val_trajectories, val_labels, fold, model_name):
    print(f"Training LSTM model for fold {fold}...")
    train_loader = prepare_dataloader(train_trajectories, train_labels)
    val_loader = prepare_dataloader(val_trajectories, val_labels, shuffle=False)
    input_dim = train_trajectories.shape[2]
    output_dim = len(np.unique(train_labels))

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
    
    model = LSTMModel(input_dim, 128, output_dim, 2).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, fold=fold, model_name=model_name)
    print(f"Finished training LSTM model for fold {fold}.")
    return model

def train_gru(train_trajectories, train_labels, val_trajectories, val_labels, fold, model_name):
    print(f"Training GRU model for fold {fold}...")
    train_loader = prepare_dataloader(train_trajectories, train_labels)
    val_loader = prepare_dataloader(val_trajectories, val_labels, shuffle=False)
    input_dim = train_trajectories.shape[2]
    output_dim = len(np.unique(train_labels))

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
    
    model = GRUModel(input_dim, 128, output_dim, 2).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, fold=fold, model_name=model_name)
    print(f"Finished training GRU model for fold {fold}.")
    return model

def train_transformer(train_trajectories, train_labels, val_trajectories, val_labels, fold, model_name):
    print(f"Training Transformer model for fold {fold}...")
    train_loader = prepare_dataloader(train_trajectories, train_labels)
    val_loader = prepare_dataloader(val_trajectories, val_labels, shuffle=False)
    input_dim = train_trajectories.shape[2]
    hidden_dim = 128
    output_dim = len(np.unique(train_labels))

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

    model = TransformerModel(input_dim, 4, hidden_dim, output_dim, 2).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, fold=fold, model_name=model_name)
    print(f"Finished training Transformer model for fold {fold}.")
    return model

def train_cnn(train_trajectories, train_labels, val_trajectories, val_labels, fold, model_name):
    print(f"Training CNN model for fold {fold}...")
    train_loader = prepare_dataloader(train_trajectories, train_labels)
    val_loader = prepare_dataloader(val_trajectories, val_labels, shuffle=False)
    input_dim = train_trajectories.shape[2]
    output_dim = len(np.unique(train_labels))

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

    model = CNNModel(input_dim, output_dim).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, fold=fold, model_name=model_name)
    print(f"Finished training CNN model for fold {fold}.")
    return model

def train_mobilebert(train_trajectories, train_labels, val_trajectories, val_labels, id2label_detailed, fold, model_name):
    print(f"Training MobileBERT model for fold {fold}...")
    num_labels = len(np.unique(train_labels))
    
    # Convert to Hugging Face datasets
    train_ds = Dataset.from_dict({"trajectory": train_trajectories.tolist(), "labels": train_labels})
    val_ds = Dataset.from_dict({"trajectory": val_trajectories.tolist(), "labels": val_labels})
    data_dict = DatasetDict({"train": train_ds, "test": val_ds})
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    tokenizer.model_max_length = 512
    
    # Apply preprocessing
    tokenized_data = data_dict.map(lambda examples: preprocess_function(examples, id2label_detailed, tokenizer), batched=True)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define model
    model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=num_labels)  

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/mobilebert_fold_{fold}",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
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
    trainer.train()
    print(f"Finished training MobileBERT model for fold {fold}.")
    return model

# Define Inference Function
def inference(model, data_loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds

if __name__ == "__main__":
    data_path = "/data/TGSSE/UpdatedIntentions/XYZ/800pad_66" 
    labels_path = "/data/TGSSE/UpdatedIntentions/labels.yaml"
    label_detailed_path = "/data/TGSSE/UpdatedIntentions/XYZ/800pad_66/trajectory_with_intentions_800_pad_533_min_151221_label_detailed.yaml"
    
    # Load the data
    flight_data, flight_labels, id2label, label2id, id2label_detailed = load_flight_data(data_path, labels_path, label_detailed_path)

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {}
    model_names = ['LSTM', 'GRU', 'Transformer', 'CNN']
    
    print("Starting training of all models with K-fold cross-validation...\n")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(flight_data)):
        print(f"\nFold {fold+1}/{kf.n_splits}\n")
        
        wandb.init(project="alternate_test", name=f"fold_{fold+1}")
        
        train_trajectories, val_trajectories = flight_data[train_idx], flight_data[val_idx]
        train_labels, val_labels = np.array(flight_labels)[train_idx], np.array(flight_labels)[val_idx]
        
        # Train LSTM
        models[f'LSTM_fold_{fold}'] = train_lstm(train_trajectories, train_labels, val_trajectories, val_labels, fold, 'LSTM')
        
        # Train GRU
        models[f'GRU_fold_{fold}'] = train_gru(train_trajectories, train_labels, val_trajectories, val_labels, fold, 'GRU')
        
        # Train Transformer
        models[f'Transformer_fold_{fold}'] = train_transformer(train_trajectories, train_labels, val_trajectories, val_labels, fold, 'Transformer')
        
        # Train CNN
        models[f'CNN_fold_{fold}'] = train_cnn(train_trajectories, train_labels, val_trajectories, val_labels, fold, 'CNN')
        
        # Perform inference and print results for each fold and each model
        for model_name in model_names:
            full_model_name = f"{model_name}_fold_{fold}"
            model = models[full_model_name]
            print(f"\nPerforming inference with {full_model_name} model...")
            preds = inference(model, prepare_dataloader(val_trajectories, val_labels, shuffle=False))
            print(f"Predictions: {preds[:50]}")  # Print first 50 predictions for brevity
            accuracy = np.mean(preds == val_labels)
            print(f"Accuracy of {full_model_name} model: {accuracy:.4f}")
            wandb.log({f"inference_accuracy_{model_name}": accuracy})

        wandb.finish()
    
    print("\nFinished training all models with K-fold cross-validation.")
