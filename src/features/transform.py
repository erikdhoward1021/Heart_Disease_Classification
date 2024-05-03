from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle    

def transform_data(data):
    # Separate the features and target variable
    X = data.drop('target', axis=1)
    y = data['target']

    # Encode categorical variables using OneHotEncoder
    categorical_cols = X.select_dtypes(include='category').columns
    encoder = OneHotEncoder(sparse=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
    X_encoded.columns = encoder.get_feature_names(categorical_cols)
    X = pd.concat([X.drop(categorical_cols, axis=1), X_encoded], axis=1)

    # Normalize numerical variables
    numerical_cols = X.select_dtypes(include=['int32', 'int64', 'float64']).columns
    scaler = MinMaxScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Save the encoders and scalers
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test