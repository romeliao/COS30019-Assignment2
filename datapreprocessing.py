import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(spreadsheet_path, output_dir, traffic_data_path):
    """
    Preprocess the SCATS data and combine it with traffic count data.

    Args:
        spreadsheet_path (str): Path to the input Excel file.
        output_dir (str): Directory to save the processed datasets.
        traffic_data_path (str): Path to the traffic count CSV file.

    Returns:
        None
    """
    # Load the first two rows to create the headers
    header_rows = pd.read_excel(spreadsheet_path, sheet_name="Data", nrows=2, header=None)
    headers = header_rows.apply(lambda x: ' '.join(x.dropna().astype(str)).strip(), axis=0).tolist()

    # Load the data with combined headers
    data = pd.read_excel(spreadsheet_path, sheet_name="Data", skiprows=2, header=None, names=headers)

    # Rename 'Start Time Date' to 'Date' if it exists
    if 'Start Time Date' in data.columns:
        data.rename(columns={'Start Time Date': 'Date'}, inplace=True)

    # Ensure 'Date' exists
    if 'Date' not in data.columns:
        raise ValueError("The 'Date' column is missing. Check the combined headers.")

    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Drop unnecessary columns
    columns_to_drop = ['HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc']
    for col in columns_to_drop:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    # Replace inf values and fill NaNs
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # Extract V00 to V95 columns as targets
  # Dynamically detect columns ending with V00 to V95
    v_suffixes = [f"V{str(i).zfill(2)}" for i in range(96)]
    v_columns = [col for col in data.columns if any(col.endswith(v) for v in v_suffixes)]

    # Confirm all expected columns are found
    missing_v_suffixes = [v for v in v_suffixes if not any(col.endswith(v) for col in data.columns)]
    if missing_v_suffixes:
        raise KeyError(f"The following expected V columns are missing: {missing_v_suffixes}")


    y = data[v_columns]
    X = data.drop(columns=v_columns)  # Remove V00-V95 from features

    # Optional: Remove rows where target is invalid (e.g., all 0s or NaN)
    valid_rows = y.notnull().all(axis=1)
    X = X[valid_rows]
    y = y[valid_rows]

    # Normalize numerical features
    numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
    scaler = MinMaxScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=52)

    # Save to output_dir
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"Processed and combined datasets saved in: {output_dir}")
