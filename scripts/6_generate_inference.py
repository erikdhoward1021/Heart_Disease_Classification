import torch
import pandas as pd
import pickle

# load the MALE model
model = torch.load('models/Male_FFNN_20240506225136.pth')

# load the MALE test male_test_data
male_test_data = pd.read_csv('data/processed/male_test_features.csv')

# import the encoder and scaler in order to transform the features male_test_data
encoder = pickle.load(open('models/pipelines/male_encoder.pkl', 'rb'))
scaler = pickle.load(open('models/pipelines/male_scaler.pkl', 'rb'))

# set dtypes appropriately
male_test_data['age'] = male_test_data['age'].astype('int32')
male_test_data['chest_pain_type'] = male_test_data['chest_pain_type'].astype('category')
male_test_data['resting_bp'] = male_test_data['resting_bp'].astype('int32')
male_test_data['cholesterol'] = male_test_data['cholesterol'].astype('int32')
male_test_data['fasting_blood_sugar'] = male_test_data['fasting_blood_sugar'].astype('category')
male_test_data['resting_ecg'] = male_test_data['resting_ecg'].astype('category')
male_test_data['max_heart_rate'] = male_test_data['max_heart_rate'].astype('int32')
male_test_data['exercise_angina'] = male_test_data['exercise_angina'].astype('category')
male_test_data['oldpeak'] = male_test_data['oldpeak'].astype('int32')
male_test_data['st_slope'] = male_test_data['st_slope'].astype('category')

# apply encoder appropriately
categorical_cols = male_test_data.select_dtypes(include='category').columns
male_test_data_encoded = pd.DataFrame(encoder.fit_transform(male_test_data[categorical_cols]))
male_test_data_encoded.columns = encoder.get_feature_names_out(categorical_cols)
male_test_data = pd.concat([male_test_data.drop(categorical_cols, axis=1), male_test_data_encoded], axis=1)

# apply scalar appropriately
numerical_cols = male_test_data.select_dtypes(include=['int32', 'int64', 'float64']).columns
male_test_data[numerical_cols] = scaler.fit_transform(male_test_data[numerical_cols])

# convert to tensor
male_inference_data = torch.tensor(male_test_data.values).float()

# generate inference
prediction = model(male_inference_data)

print(prediction)
