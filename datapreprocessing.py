import os
import pandas as pd
from sklearn.model_selection import train_test_split

#Defining the dataset file path 
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "datasets", "Traffic_Count_Locations_with_LONG_LAT.csv")

#loading the dataset 
data = pd.read_csv(file_path)

#display basic information about the dataset 
print("Dataset Information:")
print(data.info())

#handle missing data (Dropping rows with missing values)
data_cleaned = data.dropna()

#Selecting relevant features and target variables 
#Features = Numerical columns except the target variable
features = ['X', 'Y', 'YEAR_SINCE', 'LAST_YEAR', 'AADT_TRUCK', 'PER_TRUCKS']
target = 'AADT_ALLVE'

#extract features X and Target Y from data
X = data_cleaned[features]
y = data_cleaned[target] 

#splitting the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#saving the processed data to CSV files 
output_dir = os.path.join(base_dir, "processed_data")
os.makedirs(output_dir, exist_ok=True) #create directory if it doesn't exist

X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

print("Data preprocessing completed. Processed data saved to:", output_dir)
