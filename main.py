import os
from datapreprocessing import preprocess_data

if __name__ == "__main__":
    # Define the input spreadsheet path and output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    spreadsheet_path = os.path.join(base_dir, "Scats Data October 2006.xls")
    output_dir = os.path.join(base_dir, "processed_dataset")

    # Run the preprocessing function
    preprocess_data(spreadsheet_path, output_dir)