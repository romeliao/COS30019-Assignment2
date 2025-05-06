import os
from datapreprocessing import preprocess_data

if __name__ == "__main__":
    # Define the input spreadsheet path, traffic data path, and output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    spreadsheet_path = os.path.join(base_dir, "Scats Data October 2006.xls")
    traffic_data_path = os.path.join(base_dir, "Traffic_Count_Locations_with_LONG_LAT.csv")
    output_dir = os.path.join(base_dir, "processed_dataset")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run the preprocessing function
    preprocess_data(spreadsheet_path, output_dir, traffic_data_path)