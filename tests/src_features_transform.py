import pytest
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.transform import transform_data

def test_transform():
    """
    Test that the encoder and scaler are correctly applied to new data
    """
    # Define the test data: ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'st_slope', 'target']
    test_data = pd.DataFrame({
        'age': [63, 67, 67, 37, 41, 63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'chest_pain_type': [1, 4, 4, 3, 2, 1, 4, 4, 3, 2],
        'resting_bp': [145, 160, 120, 130, 130, 145, 160, 120, 130, 130],
        'cholesterol': [233, 286, 229, 250, 204, 233, 286, 229, 250, 204],
        'fasting_blood_sugar': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'resting_ecg': [2, 2, 2, 0, 2, 2, 2, 2, 0, 2],
        'max_heart_rate': [150, 108, 129, 187, 172, 150, 108, 129, 187, 172],
        'exercise_angina': [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        'oldpeak': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'st_slope': [2, 2, 2, 1, 2, 2, 2, 2, 1, 2],
        'target': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    })

    # I don't know the distribution of the scaler and encoder, but write a test that checks if the output is as expected
    X_train, X_test, y_train, y_test = transform_data(test_data)
    

    # Create a test dataframe of 10 records that has 5 male and 5 female present
    X_train_expected = pd.DataFrame({
        'age': [63, 67, 67, 37, 41, 63, 67, 67, 37],
        'sex': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'chest_pain_type': [1, 4, 4, 3, 2, 1, 4, 4, 3],
        'resting_bp': [145, 160, 120, 130, 130, 145, 160, 120, 130],
        'cholesterol': [233, 286, 229, 250, 204, 233, 286, 229, 250],
        'fasting_blood_sugar': [1, 0, 0, 0, 0, 1, 0, 0, 0],
        'resting_ecg': [2, 2, 2, 0, 2, 2, 2, 2, 0],
        'max_heart_rate': [150, 108, 129, 187, 172, 150, 108, 129, 187],
        'exercise_angina': [0, 1, 1, 0, 0, 0, 1, 1, 0],
        'oldpeak': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'st_slope': [2, 2, 2, 1, 2, 2, 2, 2, 1]
    })

    X_train_expected = X_train_expected.astype({
        'age': 'int32',
        'sex': 'category',
        'chest_pain_type': 'category',
        'resting_bp': 'int32',
        'cholesterol': 'int32',
        'fasting_blood_sugar': 'category',
        'resting_ecg': 'category',
        'max_heart_rate': 'int32',
        'exercise_angina': 'category',
        'oldpeak': 'int32',
        'st_slope': 'category'
    })

    y_train_expected = pd.Series([1, 0, 0, 1, 0, 1, 0, 0])

    X_test_expected = pd.DataFrame({
        'age': [63, 67],
        'sex': [1, 1],
        'chest_pain_type': [4, 4],
        'resting_bp': [160, 120],
        'cholesterol': [286, 229],
        'fasting_blood_sugar': [0, 0],
        'resting_ecg': [2, 2],
        'max_heart_rate': [108, 129],
        'exercise_angina': [1, 1],
        'oldpeak': [0.0, 0.0],
        'st_slope': [2, 2],
        'target': [0, 0]
    })

    X_test_expected = X_test_expected.astype({
        'age': 'int32',
        'sex': 'category',
        'chest_pain_type': 'category',
        'resting_bp': 'int32',
        'cholesterol': 'int32',
        'fasting_blood_sugar': 'category',
        'resting_ecg': 'category',
        'max_heart_rate': 'int32',
        'exercise_angina': 'category',
        'oldpeak': 'int32',
        'st_slope': 'category'
    })

    y_test_expected = pd.Series([1, 0])

    # Is it possible to test the basic_clean function give it provides a tuple of two dataframes?
    male_test_output, female_test_output = basic_clean(test_data)
    
    # Confirm the output is as expected
    pd.testing.assert_frame_equal(male_expected_output, male_test_output)
    pd.testing.assert_frame_equal(female_expected_output, female_test_output)


                                     