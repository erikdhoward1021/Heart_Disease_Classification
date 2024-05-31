import torch
import pandas as pd
from datetime import datetime
from src.models.FFNN import FFNN
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.optim as optim

# Define your custom dataset class for male_data
class ModelDataset(Dataset):
    def __init__(self, train_feature_path, train_target_path):
        # Load the data from the data_path
        self.features = pd.read_csv(train_feature_path).values
        self.target = pd.read_csv(train_target_path).values
        self.target = self.target.reshape(-1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # Return the features and target for the given index
        sample_features = torch.tensor(self.features[idx], dtype=torch.float)
        sample_target = torch.tensor(self.target[idx], dtype=torch.long)

        return sample_features, sample_target

# Set the path to your male_data
male_features_path = 'C:/Users/erihoward/Documents/GitHub/Heart_Disease_Classification/data/processed/male_train_features.csv'
male_target_path = 'C:/Users/erihoward/Documents/GitHub/Heart_Disease_Classification/data/processed/male_train_target.csv'

# Create an instance of your custom dataset
male_dataset = ModelDataset(train_feature_path=male_features_path, train_target_path=male_target_path)

# Create a data loader for your male_dataset
batch_size = 16
data_loader = DataLoader(male_dataset, batch_size=batch_size, shuffle=True)

# Create an instance of your FFNN model
input_size = len(male_dataset.features[0])
hidden_size = 64
output_size = len(set(male_dataset.target.flatten()))
model = FFNN(input_size, hidden_size, output_size)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
lr = 0.005
optimizer = optim.Adam(model.parameters(), lr=lr)

# Set the number of training epochs
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Print the loss after each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

time = datetime.now().strftime("%Y%m%d%H%M%S")
# Save the trained model
torch.save(model.state_dict(), f'C:/Users/erihoward/Documents/GitHub/Heart_Disease_Classification/models/Male_FFNN_{time}.pth')