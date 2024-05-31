import pandas as pd
from typing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle    

def transform_data(data_path: str, target_column: str = 'target', population: str = 'male') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    This function reads the data from the "data/cleaned" directory, encodes the categorical variables, normalizes the numerical variables, and splits the data into train and test sets.

    Parameters:
        data_path (str): The path to the dataset
        target_column (str): The name of the target variable
        population (str): The population to consider for the model

    Returns:
        X_train (DataFrame): The features of the training set
        X_test (DataFrame): The features of the test set
        y_train (Series): The target variable of the training set
        y_test (Series): The target variable of the test set
    '''
    data = pd.read_csv(data_path)

        # Convert the data types of the columns
    data['age'] = data['age'].astype('int32')
    data['sex'] = data['sex'].astype('category')
    data['chest_pain_type'] = data['chest_pain_type'].astype('category')
    data['resting_bp'] = data['resting_bp'].astype('int32')
    data['cholesterol'] = data['cholesterol'].astype('int32')
    data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype('category')
    data['resting_ecg'] = data['resting_ecg'].astype('category')
    data['max_heart_rate'] = data['max_heart_rate'].astype('int32')
    data['exercise_angina'] = data['exercise_angina'].astype('category')
    data['oldpeak'] = data['oldpeak'].astype('int32')
    data['st_slope'] = data['st_slope'].astype('category')
    data['target'] = data['target'].astype('category')
    
    # Separate the features and target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Encode categorical variables using OneHotEncoder
    categorical_cols = X.select_dtypes(include='category').columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
    X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    X = pd.concat([X.drop(categorical_cols, axis=1), X_encoded], axis=1)

    # Normalize numerical variables
    numerical_cols = X.select_dtypes(include=['int32', 'int64', 'float64']).columns
    scaler = MinMaxScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Save the encoders and scalers
    with open(f'models/pipelines/{population}_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open(f'models/pipelines/{population}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test