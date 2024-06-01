import pandas as pd
from typing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle    

def transform_data(data: Union[str, pd.DataFrame], target_column: str = 'target', population: str = 'male') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    This function reads the data from the "data/cleaned" directory or DataFrame, encodes the categorical variables, normalizes the numerical variables, and splits the data into train and test sets.

    Parameters:
        data_path (str or DataFrame): The path to the dataset or DataFrame
        target_column (str): The name of the target variable
        population (str): The population to consider for the model

    Returns:
        X_train (DataFrame): The features of the training set
        X_test (DataFrame): The features of the test set
        y_train (Series): The target variable of the training set
        y_test (Series): The target variable of the test set
    '''
    if isinstance(data, str):
        data = pd.read_csv(data)

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
    X = data.drop([target_column, 'sex'], axis=1)
    y = data[target_column]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Encode X_train (to ensure no data leakage) categorical variables using OneHotEncoder
    categorical_cols = X_train.select_dtypes(include='category').columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))
    X_train_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    X_train = pd.concat([X_train.drop(categorical_cols, axis=1), X_train_encoded], axis=1)

    ### DO NOT APPLY THE ENCODER TO X_TEST ###
    # This should be done during validation to ensure that the encoder
    # can be correctly loaded and applied to new data

    # Normalize numerical variables
    numerical_cols = X_train.select_dtypes(include=['int32', 'int64', 'float64']).columns
    scaler = MinMaxScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

    ### DO NOT APPLY THE SCALER TO X_TEST ###
    # This should be done during validation to ensure that the scaler
    # can be correctly loaded and applied to new data

    # Save the encoders and scalers for use during validation &
    with open(f'models/pipelines/{population}_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open(f'models/pipelines/{population}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test