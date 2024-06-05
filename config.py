# MODEL PARAMETERS
INPUT_SIZE = 22
HIDDEN_SIZE = 100
OUTPUT_SIZE = 1
BATCH_SIZE = 16
THRESHOLD = 0.5

# MODEL PATH
MODEL_VERSION = 'artifacts/models/FFNN_date_20240603_loss_0.337.pth'
MODEL_NAME = 'FFNN_date_20240603_loss_0.337'

# PIPELINES PATHS
ENCODER_ARTIFACT = 'artifacts/pipelines/encoder_v2.pkl'
SCALER_ARTIFACT = 'artifacts/pipelines/scaler_v2.pkl'

# DATA PATHS
X_TRAIN_PATH = 'data/train/train_features.csv'
Y_TRAIN_PATH = 'data/train/train_target.csv'
X_TEST_PATH = 'data/test/test_features.csv'
Y_TEST_PATH = 'data/test/test_target.csv'