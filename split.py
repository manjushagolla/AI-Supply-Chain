# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define constants
DATA_PATH = 'newcleaned_medicinedata.csv'
TARGET_COLUMN = 'Transport_Status'
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.5
RANDOM_STATE = 42

def load_data(data_path, target_column):
    """
    Load the dataset from a CSV file and specify the dtype for the target column.

    Parameters:
    data_path (str): Path to the CSV file.
    target_column (str): The target column name.

    Returns:
    pd.DataFrame: Loaded dataset.
    """
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {data_path} was not found.")
    
    # Load data with specified dtype for mixed-type columns
    dtype = {target_column: 'str'}
    return pd.read_csv(data_path, dtype=dtype, low_memory=False)

def preprocess_data(df, target_column):
    """
    Drop rows with NaN values in target column and features.
    
    Parameters:
    df (pd.DataFrame): The loaded dataset.
    target_column (str): The target column name.
    
    Returns:
    pd.DataFrame: Cleaned dataset.
    """
    # Drop rows with NaN values in target column and all features
    df.dropna(subset=[target_column], inplace=True)
    df.dropna(inplace=True)
    return df

def split_data(df, target_column, test_size, validation_size, random_state):
    """
    Perform stratified splitting into training, validation, and test sets.

    Parameters:
    df (pd.DataFrame): The cleaned dataset.
    target_column (str): The target column name.
    test_size (float): Proportion of the data to be used for testing.
    validation_size (float): Proportion of the data to be used for validation.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    X_train, X_val, X_test, y_train, y_val, y_test: Split data and target sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Perform first split into training and temporary set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Perform second split into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=validation_size, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def print_split_shapes(X_train, X_val, X_test, y_train, y_val, y_test):
    """Print shape of each split."""
    print(f"\nTraining set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

def print_target_distributions(y_train, y_val, y_test):
    """Print target distribution for each split."""
    for name, target in zip(['Train', 'Validation', 'Test'], [y_train, y_val, y_test]):
        print(f"\n{name} Target Distribution:")
        print(target.value_counts(normalize=True))

def main():
    # Load data
    df = load_data(DATA_PATH, TARGET_COLUMN)
    
    # Preprocess data (cleaning)
    df = preprocess_data(df, TARGET_COLUMN)
    
    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, TARGET_COLUMN, TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE
    )
    
    # Print split shapes
    print_split_shapes(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Print target distributions
    print_target_distributions(y_train, y_val, y_test)

if __name__ == "__main__":
    main()
