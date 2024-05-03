import pandas as pd

def basic_clean(path):
    # Read the data from the "data/raw" directory
    data = pd.read_csv()

    # Assign column names to the dataset
    data.columns = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'st_slope', 'target']

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

    return data
