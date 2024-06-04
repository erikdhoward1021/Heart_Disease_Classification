from typing import *
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle
from config import ENCODER_VERSION, SCALER_VERSION

class Dataset():
    def __init__(self, data: Union[str, pd.DataFrame], type: str, target_column: str = None):
        """
        Initialize the Dataset object.

        Parameters:
            data (Union[str, pd.DataFrame]): The path to the dataset or a DataFrame.
            type (str): The type of data to be read (e.g. 'train', 'inference').
            target_column (str, optional): The name of the target variable. Defaults to None.
        """
        self.data = data
        self.type = type
        self.target_column = target_column
        self.cleaned_data = None
        self.transformed_data = None

    def basic_clean(self) -> pd.DataFrame:
        """
        This function reads the data from the "data/raw" directory or a DataFrame, assigns column names to the dataset, converts the data types of the columns, and returns the cleaned dataset.

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        # Assign columns to the data
        if self.type == 'train':
            columns = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'st_slope', 'target']
        elif self.type == 'inference':
            columns = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'st_slope']

        # Get input data from the request and convert it to a Pandas DataFrame
        if self.type == 'train':
            dtypes = {'age': 'int32', 'sex': 'category', 'chest_pain_type': 'category', 'resting_bp': 'int32', 'cholesterol': 'int32', 'fasting_blood_sugar': 'category', 'resting_ecg': 'category', 'max_heart_rate': 'int32', 'exercise_angina': 'category', 'oldpeak': 'int32', 'st_slope': 'category', 'target': 'category'}
        elif self.type == 'inference':
            dtypes = {'age': 'int32', 'sex': 'category', 'chest_pain_type': 'category', 'resting_bp': 'int32', 'cholesterol': 'int32', 'fasting_blood_sugar': 'category', 'resting_ecg': 'category', 'max_heart_rate': 'int32', 'exercise_angina': 'category', 'oldpeak': 'int32', 'st_slope': 'category'}
        
        # If the data is a string, read the data from the file
        if isinstance(self.data, str):
            # Check if the first row contains the column names
            first_row = pd.read_csv(self.data, nrows=1)
            # If the first row consists of strings, read the data with header=0
            if all(isinstance(col, str) for col in first_row.columns):
                df = pd.read_csv(self.data, header=0)
                df.columns = columns
                df = df.astype(dtypes)
            # If the first row does not contain the column names, read the data with header=None
            else:
                df = pd.read_csv(self.data, names=columns, dtype=dtypes)
        # If the data is a DataFrame, assign the columns and data types
        else:
            df = self.data
            df.columns = columns
            df = df.astype(dtypes)

        # For now, drop rows with missing values
        self.cleaned_data = df.dropna().reset_index(drop=True)

        return self.cleaned_data
    
    def transform_data(self) -> dict:
        """
        Based on the type of data, this function transforms the data accordingly.

        If the type is 'train', it splits the dataset into train and test, 
        fit/tranforms an encoder on categorical variables & scalar on numerical variables, 
        persists the encoder, scaler, and train/test data, and returns the transformed data.

        If the type is 'inference', it encodes the categorical variables and normalizes the numerical variables
        based on the encoder and scaler fitted on the training data, and returns the transformed data.

        Returns:
            dict: A dictionary containing the transformed data.
        """
        if self.type == 'train':
            # Separate the features and target variable
            X = self.cleaned_data.drop([self.target_column], axis=1)
            y = self.cleaned_data[self.target_column]

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            # Encode X_train categorical variables using OneHotEncoder
            categorical_cols = X_train.select_dtypes(include='category').columns
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))
            X_train_encoded.columns = encoder.get_feature_names_out(categorical_cols)
            X_train = pd.concat([X_train.drop(categorical_cols, axis=1), X_train_encoded], axis=1)

            # Normalize numerical variables
            numerical_cols = X_train.select_dtypes(include=['int32', 'int64', 'float64']).columns
            scaler = MinMaxScaler()
            X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

            # Get a list of all files in the encoder / scaler directory
            files = os.listdir('models/pipelines')

            # Filter the list to only include encoder files
            encoder_files = [f for f in files if 'encoder' in f]

            # If there are no encoder files, start with 1
            if not encoder_files:
                next_encoder_number = 1
            else:
                # Get the current highest number
                highest_number = max(int(re.search(r'\d+', f).group()) for f in encoder_files)

                # Increment the highest number for the new encoder
                next_encoder_number = highest_number + 1

            # Create the new encoder name
            new_encoder_name = f'encoder_v{next_encoder_number}'
            with open(f'models/pipelines/{new_encoder_name}.pkl', 'wb') as f:
                pickle.dump(encoder, f)

            # Filter the list to only include scaler files
            scaler_files = [f for f in files if 'scaler' in f]

            # If there are no scaler files, start with 1
            if not scaler_files:
                next_scaler_number = 1
            else:
                # Get the current highest number
                highest_number = max(int(re.search(r'\d+', f).group()) for f in scaler_files)

                # Increment the highest number for the new scaler
                next_scaler_number = highest_number + 1

            # Create the new scaler name
            new_scaler_name = f'scaler_v{next_scaler_number}'
            with open(f'models/pipelines/{new_scaler_name}.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            # Save the train data
            X_train.to_csv('data/train/train_features.csv', index=False)
            y_train.to_csv('data/train/train_target.csv', index=False)

            # Save the test data
            X_test.to_csv('data/test/test_features.csv', index=False)
            y_test.to_csv('data/test/test_target.csv', index=False)

            self.transformed_data = {'X_train': X_train, 'y_train': y_train}

        elif self.type == 'inference':
            # Load the encoder and scaler
            encoder = pickle.load(open(f'models/pipelines/encoder_v{ENCODER_VERSION}.pkl', 'rb'))
            scaler = pickle.load(open(f'models/pipelines/scaler_v{SCALER_VERSION}.pkl', 'rb'))

            # Encode X_train categorical variables using OneHotEncoder
            categorical_cols = self.cleaned_data.select_dtypes(include='category').columns
            df = pd.DataFrame(encoder.transform(self.cleaned_data[categorical_cols]))
            df.columns = encoder.get_feature_names_out(categorical_cols)
            df = pd.concat([self.cleaned_data.drop(categorical_cols, axis=1), df], axis=1)

            # Normalize numerical variables
            numerical_cols = df.select_dtypes(include=['int32', 'int64', 'float64']).columns
            df[numerical_cols] = scaler.transform(df[numerical_cols])

            self.transformed_data = {'X_inference': df}

        return self.transformed_data
            


