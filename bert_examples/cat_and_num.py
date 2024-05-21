import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TransformerWithCoordinates(nn.Module):
    def __init__(self, transformer_model, num_coordinate_features, hidden_size):
        super(TransformerWithCoordinates, self).__init__()
        # Initialize the transformer model (e.g., BERT)
        self.transformer = transformer_model
        
        # Define a feedforward neural network to process the coordinates
        self.coordinate_processor = nn.Sequential(
            nn.Linear(num_coordinate_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Define a classifier that will take the combined features and output a single value
        self.classifier = nn.Linear(hidden_size * 2, 1)  # For binary classification

    def forward(self, input_ids, attention_mask, coordinates):
        # Process the text data through the transformer model
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        
        # Extract the CLS token representation (assumed to be the first token's representation)
        text_features = transformer_outputs.last_hidden_state[:, 0, :]  # CLS token output
        
        # Process the coordinates through the feedforward neural network
        coordinates_processed = self.coordinate_processor(coordinates)
        
        # Print the shapes of the text and coordinate features to understand their dimensions
        print("Text features shape:", text_features.shape)
        print("Processed coordinates shape:", coordinates_processed.shape)
        
        # Concatenate the text and coordinate features
        combined_features = torch.cat((text_features, coordinates_processed), dim=1)
        
        # Pass the combined features through the classifier to get the final logits
        logits = self.classifier(combined_features)
        
        return logits

# Example usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transformer_model = BertModel.from_pretrained('bert-base-uncased')
model = TransformerWithCoordinates(transformer_model, num_coordinate_features=9, hidden_size=768)  # Assuming 3 coordinates (x1, y1, z1, x2, y2, z2, xN, yN, zN)

# Dummy data
text = ["Example sentence for sentiment analysis."]
coordinates = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])  # Example coordinates: x1 y1 z1 x2 y2 z2 xN yN zN

# Tokenize the text input
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Forward pass through the model
logits = model(inputs['input_ids'], inputs['attention_mask'], coordinates)

# Print the final logits to understand the output of the model
print("Logits:", logits)
