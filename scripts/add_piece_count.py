import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from data_process.utils import (add_piece_count, add_piece_count_to_games,
                                remove_material_tokens)

tqdm.pandas()

input_file_path = "../data/csv/uniform_elo_distribution/train.csv"
output_file_path = "../data/csv/uniform_elo_distribution/train_piece_count.csv"

headers = ["index", "id", "date", "white_elo", "black_elo", "result", "ply", "ply_30s", "piece_uci"]

games_df = pd.read_csv(input_file_path, delimiter=";", header=None, names=headers)

games = remove_material_tokens(games_df.piece_uci)

games = games.progress_apply(add_piece_count)

games_df["piece_uci"] = games

games_df.to_csv(output_file_path, index=False, sep=";", header=False)


