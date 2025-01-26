import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_and_preprocess_data(file_path):
    """
    Load dataset from a CSV file and handle missing values.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        DataFrame: Preprocessed dataset.
        List: List of categorical column names.
    """
    data = pd.read_csv(file_path, low_memory=False)

    # Check for mixed types and fix them in categorical columns
    for col in data.columns:
        if data[col].dtype == 'object':  # Categorical columns
            # Convert mixed types to strings or handle them
            data[col] = data[col].astype(str).fillna('missing')

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns

    # Fill missing values
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    return data, categorical_cols

def encode_categorical_columns(data, categorical_cols):
    """
    Encode categorical columns using LabelEncoder.

    Parameters:
        data (DataFrame): Dataset containing categorical columns.
        categorical_cols (List): List of categorical column names.

    Returns:
        DataFrame: Dataset with encoded categorical columns.
    """
    for col in categorical_cols:
        data[col] = LabelEncoder().fit_transform(data[col])
    return data

def train_and_evaluate_model(data):
    """
    Train a Random Forest Regressor and evaluate the model.

    Parameters:
        data (DataFrame): Dataset with features and target.

    Returns:
        RandomForestRegressor: Trained model.
    """
    X = data.drop(columns=["Risk_Factor"])
    y = data["Risk_Factor"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize and apply PCA
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)

    X_train = pca.fit_transform(scaler.fit_transform(X_train))
    X_test = pca.transform(scaler.transform(X_test))

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse}")
    return model

def generate_low_stock_alerts(data, stock_column='Economic_Factor', factor=0.3):
    """
    Simulate stock levels and identify low stock items.

    Parameters:
        data (DataFrame): Dataset containing stock information.
        stock_column (str): Column used to simulate stock levels.
        factor (float): Threshold factor to identify low stock items.

    Returns:
        DataFrame: Filtered dataset with low stock items.
    """
    # Simulate stock levels
    data['Simulated_Stock_Level'] = data[stock_column] * 100

    # Set threshold for low stock alerts
    threshold = data['Simulated_Stock_Level'].mean() * factor

    # Filter low stock items
    low_stock_items = data[data['Simulated_Stock_Level'] < threshold]
    return low_stock_items

def save_to_csv(data, filename):
    """
    Save DataFrame to CSV file.

    Parameters:
        data (DataFrame): Dataset to be saved.
        filename (str): Path to the output CSV file.
    """
    data.to_csv(filename, index=False)
    print(f"Risk alerts saved to {filename}")

# Main execution
file_path = '/content/drive/MyDrive/data_risk.csv'  # Update this path to your dataset file
output_file = 'low_stock_alerts.csv'

# Load and preprocess data
data, categorical_cols = load_and_preprocess_data(file_path)
data = encode_categorical_columns(data, categorical_cols)

# Train model (optional: if you're only interested in risk alerts, you can skip this step)
model = train_and_evaluate_model(data)

# Generate low stock alerts
low_stock_items = generate_low_stock_alerts(data)

# Save low stock alerts to CSV
save_to_csv(low_stock_items, output_file)
