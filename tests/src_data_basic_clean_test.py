import pytest
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.basic_clean import basic_clean

# Test the basic_clean function

def test_basic_clean():
    # Define the expected output given the following columns: ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'st_slope', 'target']
    male_expected_output = pd.DataFrame({
        'age': [63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 1],
        'chest_pain_type': [1, 4, 4, 3, 2],
        'resting_bp': [145, 160, 120, 130, 130],
        'cholesterol': [233, 286, 229, 250, 204],
        'fasting_blood_sugar': [1, 0, 0, 0, 0],
        'resting_ecg': [2, 2, 2, 0, 2],
        'max_heart_rate': [150, 108, 129, 187, 172],
        'exercise_angina': [0, 1, 1, 0, 0],
        'oldpeak': [0.0, 0.0, 0.0, 0.0, 0.0],
        'st_slope': [2, 2, 2, 1, 2],
        'target': [1, 0, 0, 1, 0]
    })
    female_expected_output = pd.DataFrame({
        'age': [63, 67, 67, 37, 41],
        'sex': [0, 0, 0, 0, 0],
        'chest_pain_type': [1, 4, 4, 3, 2],
        'resting_bp': [145, 160, 120, 130, 130],
        'cholesterol': [233, 286, 229, 250, 204],
        'fasting_blood_sugar': [1, 0, 0, 0, 0],
        'resting_ecg': [2, 2, 2, 0, 2],
        'max_heart_rate': [150, 108, 129, 187, 172],
        'exercise_angina': [0, 1, 1, 0, 0],
        'oldpeak': [0.0, 0.0, 0.0, 0.0, 0.0],
        'st_slope': [2, 2, 2, 1, 2],
        'target': [1, 0, 0, 1, 0]
    })
    
    # Create a test dataframe of 10 records that has 5 male and 5 female present
    test_data = pd.DataFrame({
        'age': [63, 67, 67, 37, 41, 63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
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

    # Is it possible to test the basic_clean function give it provides a tuple of two dataframes?
    male_test_output, female_test_output = basic_clean(test_data)
    
    # Confirm the output is as expected
    pd.testing.assert_frame_equal(male_expected_output, male_test_output)
    pd.testing.assert_frame_equal(female_expected_output, female_test_output)


                                     