from src.data.Dataset import Dataset

# Read the raw data
path = 'data/raw/heart_statlog_cleveland_hungary_final.csv'

# Initialize the Dataset object
dataset = Dataset(path, type='train', target_column='target')

# Clean the data
dataset.basic_clean()

# Transform the data
dataset.transform_data()