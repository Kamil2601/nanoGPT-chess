import chess
import pandas as pd


def material(board):
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    white_material = (
        chess.popcount(white & board.pawns) +
        3 * chess.popcount(white & board.knights) +
        3 * chess.popcount(white & board.bishops) +
        5 * chess.popcount(white & board.rooks) +
        9 * chess.popcount(white & board.queens)
    )

    black_material = (
        chess.popcount(black & board.pawns) +
        3 * chess.popcount(black & board.knights) +
        3 * chess.popcount(black & board.bishops) +
        5 * chess.popcount(black & board.rooks) +
        9 * chess.popcount(black & board.queens)
    )

    return white_material, black_material


def create_uniform_elo_distribution(
    split_elo_dir,
    output_dir="./data/csv/uniform_elo_distribution",
    n_rows=900_000,
    n_required=820_000,
    n_train=810_000,
):
    headers = [
        "index",
        "id",
        "date",
        "white_elo",
        "black_elo",
        "result",
        "ply",
        "ply_30s",
        "piece_uci",
    ]

    for file in split_elo_dir:
        df = pd.read_csv(file, delimiter=";", header=None, names=headers, nrows=n_rows)
        df = df[(df.ply <= 300) & (df.ply >= 4)]

        if len(df) < n_required:
            print(f"File {file} has less than {n_required} rows. Skipping")
            continue
        else:
            print(f"File {file} has {len(df)} rows.")

        df = df.iloc[:n_required]

        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:]

        train_df.to_csv(
            f"{output_dir}/train.csv", index=False, mode="a", header=False, sep=";"
        )
        test_df.to_csv(
            f"{output_dir}/test.csv", index=False, mode="a", header=False, sep=";"
        )
        print(
            f"Saved {len(train_df)} rows to train.csv and {len(test_df)} rows to test.csv."
        )