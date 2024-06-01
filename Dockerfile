# Dockerfile

# Use an official PyTorch image as a base image
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Copy your project files into the container
COPY app/inference_script.py .
COPY data/model_weights.pth .
COPY model/model_module.py .
COPY requirements.txt .

# Install any additional dependencies
RUN pip install -r requirements.txt

# Expose Port for Streamlit (8501 by Default)
EXPOSE 8000

# Define the command to run your inference script
CMD ["python", "/app/inference_script.py"]
