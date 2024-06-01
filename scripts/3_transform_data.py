from src.features.transform import transform_data

# Transform Male Dataset
male_data_path = 'data/cleaned/male_data.csv'
X_train, X_test, y_train, y_test = transform_data(data=male_data_path, target_column='target', population='male')

# Save the transformed data
X_train.to_csv('data/processed/male_train_features.csv', index=False)
y_train.to_csv('data/processed/male_train_target.csv', index=False)
X_test.to_csv('data/processed/male_test_features.csv', index=False)
y_test.to_csv('data/processed/male_test_target.csv', index=False)

# Transform Female Dataset
# data_path = 'C:/Users/erihoward/Documents/GitHub/Heart_Disease_Classification/data/cleaned/female_data.csv'
female_data_path = 'data/cleaned/female_data.csv'
X_train, X_test, y_train, y_test = transform_data(data=female_data_path, target_column='target', population='female')

# Save the transformed data
X_train.to_csv('data/processed/female_train_features.csv', index=False)
y_train.to_csv('data/processed/female_train_target.csv', index=False)
X_test.to_csv('data/processed/female_test_features.csv', index=False)
y_test.to_csv('data/processed/female_test_target.csv', index=False)