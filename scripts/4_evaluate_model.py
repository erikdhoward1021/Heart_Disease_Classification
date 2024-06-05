import torch
import pandas as pd
import pickle
from config import MODEL_VERSION, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, X_TEST_PATH, Y_TEST_PATH
from src.data.Dataset import Dataset
from src.models.FFNN import FFNN

# Load the model
input_size = INPUT_SIZE
hidden_size = HIDDEN_SIZE
output_size = OUTPUT_SIZE
model = FFNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(MODEL_VERSION))
model.eval()

# load the test features
test_features_path = X_TEST_PATH
test_features = Dataset(test_features_path, 'inference')

# clean and transform the test features
test_features.basic_clean()
test_transformed = test_features.transform_data()['X_inference']

# convert to tensor
inference_data = torch.tensor(test_transformed.values, dtype=torch.float32)

# load the labels and convert to tensor
labels = pd.read_csv(Y_TEST_PATH).values
labels = torch.tensor(labels).float()

# generate inference
with torch.no_grad():
    predictions = model(inference_data)

# reshape the labels to match the model output
labels = labels.reshape(predictions.shape)

# Using torch built in capabilities, evaluate the model given inference_data and labels
criterion = torch.nn.BCELoss()
loss = criterion(predictions, labels)
print(round(loss.item(), 3))
