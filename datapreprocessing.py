import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(spreadsheet_path, output_dir):
    """
    Preprocess the SCATS data from the given spreadsheet and save the processed datasets.

    Args:
        spreadsheet_path (str): Path to the input Excel file.
        output_dir (str): Directory to save the processed datasets.

    Returns:
        None
    """
    # Load the first two rows to create the headers
    header_rows = pd.read_excel(spreadsheet_path, sheet_name="Data", nrows=2, header=None)

    # Combine the two rows into a single header row
    headers = header_rows.apply(lambda x: ' '.join(x.dropna().astype(str)).strip(), axis=0).tolist()

    # Debug: Print the combined headers
    print("Combined Headers:", headers)

    # Load the "Data" sheet from the spreadsheet with the new headers
    data_sheet = pd.read_excel(spreadsheet_path, sheet_name="Data", skiprows=2, header=None, names=headers)

    # Debug: Print the DataFrame columns
    print("DataFrame Columns Before Renaming:", data_sheet.columns)

    # Rename the combined 'Start Time Date' column to 'Date'
    if 'Start Time Date' in data_sheet.columns:
        data_sheet.rename(columns={'Start Time Date': 'Date'}, inplace=True)

    # Debug: Print the DataFrame columns after renaming
    print("DataFrame Columns After Renaming:", data_sheet.columns)

    # Ensure the 'Date' column exists
    if 'Date' not in data_sheet.columns:
        raise ValueError("The 'Date' column is missing. Check the combined headers.")

    # Preprocessing Steps

    # 1. Convert 'Date' column to datetime
    data_sheet['Date'] = pd.to_datetime(data_sheet['Date'], errors='coerce')

    # 2. Drop unnecessary columns
    columns_to_drop = ['HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'CD_MELWAY']
    data_sheet.drop(columns=columns_to_drop, inplace=True)

    # 3. Normalize numerical columns (V00 to V95)
    # Identify numerical columns (columns starting with 'V')
    # Debug: Print all column names
    print("All Columns in DataFrame:", data_sheet.columns)

    # Identify numerical columns based on data type
    numerical_columns = data_sheet.select_dtypes(include=['number']).columns.tolist()

    # Debug: Print the numerical columns
    print("Numerical Columns (by data type):", numerical_columns)

    # Ensure numerical_columns is not empty
    if not numerical_columns:
        raise ValueError("No numerical columns found. Check the column names in the DataFrame.")

    # Normalize numerical columns
    scaler = MinMaxScaler()
    data_sheet[numerical_columns] = scaler.fit_transform(data_sheet[numerical_columns])

    # 4. Include 'Date' in the features
    X = data_sheet[numerical_columns + ['Date']]  # Add 'Date' to features
    y = data_sheet['NB_TYPE_SURVEY']  # Target

    # 5. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a directory to save the processed datasets
    os.makedirs(output_dir, exist_ok=True)

    # Save the datasets as CSV files
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"Processed datasets saved in: {output_dir}")