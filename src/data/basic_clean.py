import pandas as pd
from typing import *
from sklearn.preprocessing import MinMaxScaler

def basic_clean(data: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    This function reads the data from the "data/raw" directory or a DataFrame, assigns column names to the dataset, converts the data types of the columns, and returns two cleaned datasets (male and female).
    
    Parameters:
        path (str or DataFrame): The path to the dataset or a DataFrame

    Returns:
        male_df (DataFrame): The cleaned dataset
        female_df (DataFrame): The cleaned dataset
    
    '''
    # Read the data from the "data/raw" directory
    if isinstance(data, str):
        data = pd.read_csv(data)

    # Assign column names to the dataset
    data.columns = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'st_slope', 'target']

    data.dropna(inplace=True)
    
    male_df = data[data['sex'] == 1].reset_index(drop=True)
    female_df = data[data['sex'] == 0].reset_index(drop=True)

    return male_df, female_df