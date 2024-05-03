from src.data.basic_clean import basic_clean

# Read the raw data
path = 'data/raw/heart_statlog_cleveland_hungary_final.csv'

# Perform initial basic cleaning
clean_data = basic_clean(path)

# Save the cleaned data
clean_data.to_csv('data/cleaned/clean_data.csv', index=False)