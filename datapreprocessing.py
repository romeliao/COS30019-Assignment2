import os
import pandas as pd
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
    columns_to_drop = ['HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc']
    data_sheet.drop(columns=columns_to_drop, inplace=True)

    # 3. Normalize numerical columns (V00 to V95)
    # Identify numerical columns (columns starting with 'V')
    print("All Columns in DataFrame:", data_sheet.columns)
    numerical_columns = data_sheet.select_dtypes(include=['number']).columns.tolist()
    print("Numerical Columns (by data type):", numerical_columns)

    if not numerical_columns:
        raise ValueError("No numerical columns found. Check the column names in the DataFrame.")

    # Normalize numerical columns
    scaler = MinMaxScaler()
    data_sheet[numerical_columns] = scaler.fit_transform(data_sheet[numerical_columns])

    # Include 'Date' and 'CD_MELWAY' in the features
    X = data_sheet[numerical_columns + ['Date', 'CD_MELWAY']]
    y = data_sheet['NB_TYPE_SURVEY']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load and clean the traffic count data
    traffic_data = pd.read_csv(traffic_data_path)

    # Keep only the important columns
    traffic_data = traffic_data[['TFM_ID', 'X', 'Y', 'AADT_ALLVE', 'AADT_TRUCK', 'PER_TRUCKS', 'SITE_DESC']]

    # Remove duplicate rows
    traffic_data = traffic_data.drop_duplicates()

    # Drop rows with missing essential values
    traffic_data = traffic_data.dropna(subset=['TFM_ID', 'X', 'Y', 'AADT_ALLVE'])

    # Ensure correct data types
    traffic_data['TFM_ID'] = traffic_data['TFM_ID'].astype(int)
    traffic_data['AADT_ALLVE'] = pd.to_numeric(traffic_data['AADT_ALLVE'], errors='coerce')
    traffic_data['AADT_TRUCK'] = pd.to_numeric(traffic_data['AADT_TRUCK'], errors='coerce')
    traffic_data['PER_TRUCKS'] = pd.to_numeric(traffic_data['PER_TRUCKS'], errors='coerce')

    # Sort by TFM_ID and reset index
    traffic_data = traffic_data.sort_values(by='TFM_ID').reset_index(drop=True)

    # Save the cleaned traffic data
    cleaned_traffic_data_path = os.path.join(output_dir, "Cleaned_Traffic_Count_Locations.csv")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    traffic_data.to_csv(cleaned_traffic_data_path, index=False)
    print(f"[INFO] Traffic data cleaned and saved as '{cleaned_traffic_data_path}'")

    # Reload the cleaned traffic data for merging
    cleaned_traffic_data = pd.read_csv(cleaned_traffic_data_path)

    # Ensure CD_MELWAY is also a string for merging
    data_sheet['CD_MELWAY'] = data_sheet['CD_MELWAY'].astype(str).str.strip()
    cleaned_traffic_data['TFM_ID'] = cleaned_traffic_data['TFM_ID'].astype(str).str.strip()

    # Combine traffic data with train/test datasets
    X_train_combined = X_train.merge(cleaned_traffic_data, how='left', left_on='CD_MELWAY', right_on='TFM_ID')
    X_test_combined = X_test.merge(cleaned_traffic_data, how='left', left_on='CD_MELWAY', right_on='TFM_ID')

    # Debug: Check merge results
    print("Number of Rows in X_train_combined:", X_train_combined.shape[0])
    print("Number of Non-Matching Rows in X_train_combined:", X_train_combined['TFM_ID'].isnull().sum())

    # Remove traffic data columns from X_test_combined
    columns_to_remove = ['TFM_ID', 'X', 'Y', 'AADT_ALLVE', 'AADT_TRUCK', 'PER_TRUCKS', 'SITE_DESC']
    X_test_combined = X_test_combined.drop(columns=columns_to_remove, errors='ignore')

    # Debug: Check the final structure of X_test_combined
    print("Columns in X_test_combined after removing traffic data:", X_test_combined.columns)

    # Save the combined datasets as CSV files
    X_train_combined.to_csv(os.path.join(output_dir, "X_train_combined.csv"), index=False)
    X_test_combined.to_csv(os.path.join(output_dir, "X_test_combined.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"Processed and combined datasets saved in: {output_dir}")