import os
import glob
import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE

def process_file(filename):
    # Load the dataset
    df = pd.read_csv(filename)
    
    # Shuffle the dataset to ensure random ordering
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate features and target
    X = df[['HR', 'BT', 'SpO2', 'Age', 'Gender']]
    y = df['Outcome']
    
    # Apply SMOTE to generate synthetic samples for the minority class
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Create a new DataFrame from the resampled data and shuffle it again
    df_resampled = pd.DataFrame(X_resampled, columns=['HR', 'BT', 'SpO2', 'Age', 'Gender'])
    df_resampled['Outcome'] = y_resampled
    df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the resampled dataset to a new CSV file
    new_filename = filename.replace(".csv", "_resampled.csv")
    df_resampled.to_csv(new_filename, index=False)
    
    print(f"Resampling complete for {filename}.")
    print("Class distribution after SMOTE:")
    print(df_resampled['Outcome'].value_counts())

def main():
    parser = argparse.ArgumentParser(description="Apply SMOTE to client data CSV files and output resampled CSV files.")
    parser.add_argument('--file', type=str, help="Input CSV file to process. If not provided, all 'client_data_*.csv' files in current directory will be processed.")
    args = parser.parse_args()
    
    if args.file:
        process_file(args.file)
    else:
        # Process all files matching the pattern
        files = glob.glob("../data/client_data_*.csv")
        if not files:
            print("No client_data_*.csv files found in the current directory.")
            return
        for file in files:
            process_file(file)

if __name__ == '__main__':
    main()
