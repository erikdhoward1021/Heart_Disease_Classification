# inference_script.py

from flask import Flask, request, jsonify
import json
import torch
from src.models.FFNN import FFNN

app = Flask(__name__)

# Load the model
input_size = 20
hidden_size = 100
output_size = 2
model = FFNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('models/Male_FFNN_date_20240531_loss_0.47533559799194336.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.get_json()
        # Deserialize JSON data to a nested list
        received_serializable_list = json.loads(input_data)['data']

        # Convert the nested list to a PyTorch tensor
        tensor_data = torch.tensor(received_serializable_list, dtype=torch.float32)

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
