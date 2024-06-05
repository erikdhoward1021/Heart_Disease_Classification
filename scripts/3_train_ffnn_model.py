import torch
import pandas as pd
from datetime import datetime
from src.models.FFNN import FFNN
from torch.utils.data import DataLoader, Dataset
from config import HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE, X_TRAIN_PATH, Y_TRAIN_PATH

import torch.nn as nn
import torch.optim as optim

# Define your custom dataset class for male_data
class ModelDataset(Dataset):
    def __init__(self, train_feature_path, train_target_path):
        # Load the data from the data_path
        self.features = pd.read_csv(train_feature_path).fillna(0).values
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
features_path = X_TRAIN_PATH
target_path = Y_TRAIN_PATH

# Create an instance of your custom dataset
dataset = ModelDataset(train_feature_path=features_path, train_target_path=target_path)

# Create a data loader for your male_dataset
batch_size = BATCH_SIZE
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create an instance of your FFNN model
input_size = len(dataset.features[0])
hidden_size = HIDDEN_SIZE
output_size = OUTPUT_SIZE
model = FFNN(input_size, hidden_size, output_size)

# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer & number of epochs
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
num_epochs = 750

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in data_loader:
        # Forward pass
        outputs = model(inputs)
        # Reshape the labels based on the output of the model and ensure they are of type float
        labels = labels.view(outputs.shape).float()
        # Calculate the loss
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Zero the gradients
        optimizer.zero_grad()
        # Accumulate the batch loss
        epoch_loss += loss.item()
    # Calculate the average loss for the epoch
    epoch_loss /= len(data_loader)
    # Print the loss after each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

date = datetime.now().strftime("%Y%m%d")
loss = round(epoch_loss, 3)
# Save the trained model
torch.save(model.state_dict(), f'models/FFNN_date_{date}_loss_{loss}.pth')