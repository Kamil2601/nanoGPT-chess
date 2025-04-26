import io
import os
import re
from pathlib import Path

import chess.pgn
import pandas as pd


def game_to_piece_uci(game: chess.pgn.Game | str):
    if isinstance(game, str):
        game = chess.pgn.read_game(io.StringIO(game))

    board = game.board()
    res = []
    for move in game.mainline_moves():
        piece = board.piece_at(move.from_square).symbol().upper()
        uci = move.uci()
        res.append(piece + uci)
        board.push(move)

    return " ".join(res)


def extract_pgn_games(path):
    if isinstance(path, str):
        path = Path(path)

    if path.is_dir():
        files = path.glob("*.pgn")
    else:
        files = [path]

    games = []
    for file in files:
        with open(file, "r") as pgn_file:
            pgn_text = pgn_file.read()
            game_texts = re.findall(r"\[Event[^\[]*(?:\[(?!Event)[^\[]*)*", pgn_text)
            games.extend(game_texts)

    return games

def remove_material_tokens(piece_uci: pd.Series):
    return piece_uci.str.replace(r" \d+", "", regex=True)

def join_material_tokens(piece_uci: pd.Series, replace_bigger_values: bool = True):
    if replace_bigger_values:
        piece_uci = piece_uci.str.replace(r'[45678]\d', '40', regex=True)
    piece_uci = piece_uci.str.replace(r'(\S+)\s+(\d+)\s+(\d+)', r'\1 \2|\3', regex=True)

    return piece_uci

def add_elo_token_to_games(piece_uci, white_elo, black_elo):
    elo_piece_uci = (white_elo // 100 * 100).astype(str) + " " + (black_elo // 100 * 100).astype(str) + " " + piece_uci
    return elo_piece_uci
