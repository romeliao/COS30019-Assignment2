import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Defining the dataset file path
base_dir = os.path.dirname(os.path.abspath(__file__))
spreadsheet_path = os.path.join(base_dir, "Scats Data October 2006.xls")

# Define the new headers
headers = [
    "SCATS Number", "Location", "CD_MELWAY", "NB_LATITUDE", "NB_LONGITUDE", 
    "HF VicRoads Internal", "VR Internal Stat", "VR Internal Loc", 
    "NB_TYPE_SURVEY", "Date", 
    *[f"V{i:02}" for i in range(96)]  # Generate V00 to V95 dynamically
]

# Load the "Data" sheet from the spreadsheet with the new headers
data_sheet = pd.read_excel(spreadsheet_path, sheet_name="Data", header=1, names=headers)

# Display basic information about the data
print("Data Sheet Information:")
print(data_sheet.info())

# Display the headers (column names) of the data
print("\nHeaders in the Data Sheet:")
print(data_sheet.columns)

# Display the first few rows of the data
print("\nData Sheet Head:")
print(data_sheet.head())

# Preprocessing Steps

# 1. Convert 'Date' column to datetime
data_sheet['Date'] = pd.to_datetime(data_sheet['Date'], errors='coerce')

# 2. Drop unnecessary columns
columns_to_drop = ['HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'CD_MELWAY']
data_sheet.drop(columns=columns_to_drop, inplace=True)

# 3. Normalize numerical columns (V00 to V95)
numerical_columns = [col for col in data_sheet.columns if col.startswith('V')]
scaler = MinMaxScaler()
data_sheet[numerical_columns] = scaler.fit_transform(data_sheet[numerical_columns])

# 4. Split data into features (X) and target (y)
# Assuming 'NB_TYPE_SURVEY' is the target column
X = data_sheet[numerical_columns]  # Features
y = data_sheet['NB_TYPE_SURVEY']  # Target

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a directory to save the processed datasets
processed_dir = os.path.join(base_dir, "processed_dataset")
os.makedirs(processed_dir, exist_ok=True)

# Save the datasets as CSV files
X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

print(f"Processed datasets saved in: {processed_dir}")