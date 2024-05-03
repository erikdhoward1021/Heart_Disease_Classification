import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set the Kaggle dataset URL
dataset_url = "mexwell/heart-disease-dataset"

# Set the output directory
output_dir = "data/raw"

# Authenticate with Kaggle API
# In C:\Users\username\.kaggle\ create a file kaggle.json with your Kaggle API credentials
# This file should have the following structure:
# {
#     "username": "your_kaggle_username",
#     "key": "your_kaggle_password"
# }
api = KaggleApi()
api.authenticate()

# Download the dataset
api.dataset_download_files(dataset_url, path=output_dir, unzip=True)

# Print the downloaded files and documentation
print("Downloaded files:")
for file in os.listdir(output_dir):
    print(file)
