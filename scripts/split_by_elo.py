import glob
import os

import pandas as pd
from tqdm import tqdm

input_files = ["./data/csv/raw/lichess_elite_database-older.csv"]
input_files = glob.glob("./data/csv/raw/*.csv")  # This will select all .csv files in the directory

output_dir = "./data/csv/new_split_elo"

file_name_prefix = "elo"

chunksize = 5_000_000  # Adjust based on available memory

print(f"Splitting files: {input_files}")

headers = ["index", "id", "date", "white_elo", "black_elo", "result", "ply", "ply_30s", "piece_uci"]

# Define the bins and labels
elo_bins = [0] + list(range(1000, 4001, 200))
elo_labels = list(range(1000, 4000, 200))
elo_labels = [f"{elo}-{elo+200}" for elo in elo_labels]
elo_labels[-1] = f"{elo_labels[-1]}+"
elo_labels = ["<1000"] + elo_labels

# Create the output directory if it doesn't exist

os.makedirs(output_dir, exist_ok=True)

# Estimate total rows to set up the progress bar
total_rows = sum(1 for _ in open(input_files[0]))


# Process each input file
with tqdm(total=total_rows * len(input_files), desc="Processing", unit="rows") as pbar:
    for input_file in input_files:
        for chunk in pd.read_csv(input_file, delimiter=";", header=None, names=headers, chunksize=chunksize):
            # Calculate average Elo and assign bins
            chunk["avg_elo"] = (chunk["white_elo"] + chunk["black_elo"]) / 2
            chunk["elo_bin"] = pd.cut(chunk["avg_elo"], bins=elo_bins, labels=elo_labels)
            
            # Write each bin to its own file in the split_elo directory with ";" delimiter
            for label in elo_labels:
                bin_chunk = chunk[chunk["elo_bin"] == label][headers]
                if not bin_chunk.empty:
                    bin_chunk.to_csv(f"{output_dir}/{file_name_prefix}_{label}.csv", mode='a', index=False, header=False, sep=';')
            
            # Update the progress bar
            pbar.update(len(chunk))

print(f"Splitting complete! Files are saved in the '{output_dir}' folder.")