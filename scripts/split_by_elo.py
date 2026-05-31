import glob
import os

import pandas as pd
from tqdm import tqdm

# input_files = ["./data/csv/raw/lichess_elite_database-older.csv"]
input_files = list(glob.glob("../data/csv/raw/*.csv"))  # This will select all .csv files in the directory

output_dir = "../data/csv/elo_split_2"

file_name_prefix = "elo"

chunksize = 1_000_000  # Adjust based on available memory

print(f"Splitting files: {input_files}")

headers = ["index", "id", "date", "white_elo", "black_elo", "white_title", "black_title", "result", "ply", "ply_30s", "piece_uci"]

elo_bin_range = 100
elo_bin_min = 0
elo_bin_max = 4000

# Define the bins and labels
elo_bins = list(range(elo_bin_min, elo_bin_max + 1, elo_bin_range))
elo_labels = list(range(elo_bin_min, elo_bin_max, elo_bin_range))
elo_labels = [f"{elo}-{elo+elo_bin_range-1}" for elo in elo_labels]
elo_labels[-1] = f"{elo_labels[-1]}+"

# Create the output directory if it doesn't exist

os.makedirs(output_dir, exist_ok=True)

# Estimate total rows to set up the progress bar
total_rows = sum(1 for _ in open(input_files[0]))


# Process each input file
with tqdm(total=total_rows * len(input_files), desc="Processing", unit="rows") as pbar:
    for input_file in input_files:
        for chunk in pd.read_csv(input_file, chunksize=chunksize):
            
            chunk = chunk[(chunk["white_title"] != "BOT") & (chunk["black_title"] != "BOT")]
            chunk = chunk[(chunk["ply"] >= 4) & ((chunk["ply_30s"] == -1) | (chunk["ply_30s"] >= 4))]
            chunk = chunk[chunk["result"] != "*"]

            # Calculate average Elo and assign bins
            chunk["avg_elo"] = (chunk["white_elo"] + chunk["black_elo"]) // 2
            chunk["elo_bin"] = pd.cut(chunk["avg_elo"], bins=elo_bins, labels=elo_labels, right=False)
            
            # Write each bin to its own file in the split_elo directory
            for label, bin_chunk in chunk.groupby("elo_bin"):
                if not bin_chunk.empty:
                    bin_chunk = bin_chunk.drop(["avg_elo", "elo_bin"], axis=1)
                    output_path = f"{output_dir}/{file_name_prefix}_{label}.csv"

                    bin_chunk.to_csv(
                        output_path,
                        mode='a',
                        index=False,
                        header=not os.path.exists(output_path)
                    )
                                
            # Update the progress bar
            pbar.update(len(chunk))

print(f"Splitting complete! Files are saved in the '{output_dir}' folder.")