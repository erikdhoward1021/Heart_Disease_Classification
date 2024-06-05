# inference_script.py

from flask import Flask, request, jsonify
from src.data.Dataset import Dataset
import pandas as pd
import json
import torch
from src.models.FFNN import FFNN
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, MODEL_VERSION, ENCODER_ARTIFACT, SCALER_ARTIFACT
import pickle

app = Flask(__name__)

# Load the model
input_size = INPUT_SIZE
hidden_size = HIDDEN_SIZE
output_size = OUTPUT_SIZE
model = FFNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(MODEL_VERSION))
model.eval()

# Load the encoder and scaler
encoder = pickle.load(open(ENCODER_ARTIFACT, 'rb'))
scaler = pickle.load(open(SCALER_ARTIFACT, 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the data
        json_data = request.get_json()['data']
        
        # Check the structure of the incoming data prior to transforming it
        # Specifically, check if the data is a nested list or a single list
        if isinstance(json_data[0], list):
            json_data = json_data
        else:
            json_data = [json_data]

        received_data = pd.DataFrame(json_data)

        # Assign to Dataset object
        dataset = Dataset(received_data, 'inference')

        # Clean the data
        dataset.basic_clean()

        # Transform and return the data
        transformed_data = dataset.transform_data()['X_inference']

        # Convert the data to a PyTorch tensor
        tensor_data = torch.tensor(transformed_data.values, dtype=torch.float32)

        # Generate Inference
        with torch.no_grad():
            outputs = model(tensor_data)

        # Post-process outputs back to nested list (since it is serializable)
        predictions_numpy = outputs.numpy() # Extract predictions from outputs
        serializable_list = predictions_numpy.tolist()

        return jsonify({'predictions': serializable_list})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
