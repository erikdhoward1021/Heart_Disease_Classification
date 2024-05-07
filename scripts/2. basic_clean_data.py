import sys
sys.path.insert(0, 'C:/Users/erihoward/Documents/GitHub/Heart_Disease_Classification/src/data/')
from basic_clean import basic_clean

# Read the raw data
path = 'data/raw/heart_statlog_cleveland_hungary_final.csv'

# Perform initial basic cleaning
male_df, female_df = basic_clean(path)

# Save the cleaned data
male_df.to_csv('data/cleaned/male_data.csv', index=False)
female_df.to_csv('data/cleaned/female_data.csv', index=False)