import json
import os
import sys

import chess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from data_process import tokenizers
from nanoGPT.model import GPTConfig
from playing.agents import GPTEloAgent, NegaMaxMaterialGPTEloAgent
from playing.utils import board_to_piece_uci_moves
from training import load_models
from training.lightning_training import LightningGPT

# material_checkpoint = "./models/full_training/elo_material_ignore_material_prediction/epoch=9-step=1250000.ckpt"
# material_model = load_models.material_model(material_checkpoint)

# material_pair_checkpoint = "./models/full_training/elo_material_pair_ignore_material_prediction/epoch=9-step=1250000.ckpt"
# material_pair_model = load_models.material_pair_model(material_pair_checkpoint)

piece_count_checkpoint = "./models/full_training/elo_piece_count_ignore_material_prediction/epoch=9-step=1250000.ckpt"
piece_count_model = load_models.piece_count_model(piece_count_checkpoint)

# base_checkpoint = "./models/full_training/masked_elo/epoch=9-step=1250000.ckpt"
# base_model = load_models.base_elo_model(base_checkpoint)

# base_no_mask_checkpoint = "./models/full_training/adaptive_elo/epoch=9-step=1250000.ckpt"
# base_no_mask_model = load_models.base_elo_no_mask_model(base_no_mask_checkpoint)

# no_elo_checkpoint = "./models/full_training/no_elo/epoch=9-step=1250000.ckpt"
# no_elo_model = load_models.no_elo_model(no_elo_checkpoint)


import pandas as pd
from tqdm import tqdm

from data_process.utils import (remove_last_player_material_token,
                                remove_material_tokens)


def move_probabilities(agent, game, white_elo, black_elo):
    if isinstance(game, str):
        game = game.split(" ")
    if isinstance(white_elo, int):
        white_elo = str(white_elo)
    if isinstance(black_elo, int):
        black_elo = str(black_elo)

    board = chess.Board()

    moves_probabilities = []

    for move in game:
        if board.turn == chess.WHITE:
            agent.model_elo = white_elo
        else:
            agent.model_elo = black_elo

        current_move_probabilities = agent.legal_moves_probabilities(board)

        moves_probabilities.append(
            {
                "fen": board.fen(),
                "played_move": move,
                "probabilities": current_move_probabilities,
            }
        )

        board.push_uci(move[1:])

    return moves_probabilities


def process_game(agent, row):
    game_str = row.piece_uci.split(" ")
    game = game_str[::3]
    white_elo = row.white_elo // 100 * 100
    black_elo = row.black_elo // 100 * 100

    game_moves_probabilities = move_probabilities(agent, game, white_elo, black_elo)

    return {
        "game": game,
        "white_elo": row.white_elo,
        "black_elo": row.black_elo,
        "result": row.result,
        "game_moves_probabilities": game_moves_probabilities,
    }


def test_elo_agent(agent, games_df):
    results = []
    for row in tqdm(games_df.itertuples(index=False), total=len(games_df)):
        result = process_game(agent, row)
        results.append(result)
    return results


games_df = pd.read_csv("./data/test_piece_count.csv", delimiter=";")

# games_df = games_df.sample(frac=0.1, random_state=42)
games_df = games_df[
    (games_df["white_elo"] // 100 * 100 == 1500)
    & (games_df["black_elo"] // 100 * 100 == 1500)
]


agent = GPTEloAgent(model=piece_count_model.model, model_elo=1500, notes="piece_count")

probabilities = test_elo_agent(agent, games_df)

with open("move_probabilities.json", "w") as f:
    json.dump(probabilities, f)
