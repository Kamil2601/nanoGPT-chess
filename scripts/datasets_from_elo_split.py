import os
import pandas as pd
from pathlib import Path

lowest_elo = 700
highest_elo = 2200

# Lista plików CSV
input_files = [
    f"../data/csv/elo_split_2/elo_{elo}-{elo+99}.csv"
    for elo in range(lowest_elo, highest_elo + 1, 100)
]


# Ile wierszy brać z KAŻDEGO pliku
train_size_per_file = 500_000
val_size_per_file = 5_000
test_size_per_file = 5_000
min_game_ply = 4

# Folder wyjściowy
output_dir = "../data/csv/train_val_test_8kk_elo_700-2299"
os.makedirs(output_dir, exist_ok=True)

# Pliki wynikowe
train_output = os.path.join(output_dir, "train.csv")
val_output = os.path.join(output_dir, "validation.csv")
test_output = os.path.join(output_dir, "test.csv")


def append_to_csv(df: pd.DataFrame, output_path: str):
    file_exists = os.path.exists(output_path)

    df.to_csv(
        output_path,
        mode="a",
        header=not file_exists,
        index=False,
    )


# Usuń stare pliki wynikowe jeśli istnieją
for path in [train_output, val_output, test_output]:
    if os.path.exists(path):
        os.remove(path)

for file_path in input_files:
    print(f"Processing: {file_path}")

    total_needed = train_size_per_file + val_size_per_file + test_size_per_file

    first_take = int(total_needed * 1.2)

    df = pd.read_csv(
        file_path, nrows=first_take, dtype={"white_title": str, "black_title": str}
    )

    df = df[(df["ply_30s"] == -1) | (df["ply_30s"] >= 4)]
    df = df[df["ply"] <= 500]
    df = df[df["result"] != "*"]
    

    if len(df) < total_needed:
        raise ValueError(
            f"Not enough games with at least {min_game_ply} ply in {file_path}: {len(df)} ({total_needed} needed)"
        )
    
    

    # Opcjonalne tasowanie
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Podział
    train_df = df.iloc[:train_size_per_file]

    val_start = train_size_per_file
    val_end = val_start + val_size_per_file
    val_df = df.iloc[val_start:val_end]

    test_start = val_end
    test_end = test_start + test_size_per_file
    test_df = df.iloc[test_start:test_end]

    # Dopisywanie do zbiorów globalnych
    append_to_csv(train_df, train_output)
    append_to_csv(val_df, val_output)
    append_to_csv(test_df, test_output)

    print(
        f"Added: "
        f"train={len(train_df)}, "
        f"val={len(val_df)}, "
        f"test={len(test_df)}"
    )

print("Done.")
