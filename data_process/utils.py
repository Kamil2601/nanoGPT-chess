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
        piece_uci = piece_uci.str.replace(r'([456789]|10)\d', '40', regex=True)
    piece_uci = piece_uci.str.replace(r'(\S+)\s+(\d+)\s+(\d+)', r'\1 \2|\3', regex=True)

    return piece_uci


def add_elo_token_to_games(piece_uci, white_elo, black_elo):
    elo_piece_uci = (
        (white_elo // 100 * 100).astype(str)
        + " "
        + (black_elo // 100 * 100).astype(str)
        + " "
        + piece_uci
    )
    return elo_piece_uci


def remove_last_player_material_token(piece_uci: pd.Series):
    def filter_func(uci):
        uci = uci.split(" ")
        uci = [token for i, token in enumerate(uci) if i % 6 in [0, 2, 3, 4]]
        uci = " ".join(uci)
        return uci

    piece_uci = piece_uci.apply(filter_func)
    return piece_uci


def piece_count(board):
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    white_material = {
        "Q": chess.popcount(white & board.queens),
        "R": chess.popcount(white & board.rooks),
        "B": chess.popcount(white & board.bishops),
        "N": chess.popcount(white & board.knights),
        "P": chess.popcount(white & board.pawns)
    }

    black_material = {
        "Q": chess.popcount(black & board.queens),
        "R": chess.popcount(black & board.rooks),
        "B": chess.popcount(black & board.bishops),
        "N": chess.popcount(black & board.knights),
        "P": chess.popcount(black & board.pawns)
    }

    return white_material, black_material

def piece_count_tokens(board, max_count={"Q": 2, "R": 3, "B": 3, "N": 3, "P": 8}):
    white_material, black_material = piece_count(board)
    white_tokens = "".join(
        f"{piece}{min(count, max_count[piece])}" for piece, count in white_material.items()
    )
    black_tokens = "".join(
        f"{piece}{min(count, max_count[piece])}" for piece, count in black_material.items()
    )
    return white_tokens, black_tokens

def add_piece_count(game):
    game = game.split(" ")
    game_piece_count = []

    board = chess.Board()

    for move in game:
        board.push(chess.Move.from_uci(move[1:]))
        white_pieces, black_pieces = piece_count_tokens(board)
        
        game_piece_count += [move, white_pieces, black_pieces]

    return " ".join(game_piece_count)

def add_piece_count_to_games(games):
    return games.apply(add_piece_count)


def add_elo_and_piece_count_to_dataset(row):
    uci = row["piece_uci"]
    uci = uci.split(" ")
    uci = [token for i, token in enumerate(uci) if i % 6 in [0, 2, 3, 4]]
    uci = " ".join(uci)

    white_elo = row["white_elo"]
    black_elo = row["black_elo"]
    
    elo_piece_uci = (
        str(white_elo // 100 * 100)
        + " "
        + str(black_elo // 100 * 100)
        + " "
        + uci
    )

    return {"game": elo_piece_uci}

def row_for_base_training(row):
    uci = row["piece_uci"]
    uci = uci.split(" ")
    uci = [token for i, token in enumerate(uci) if i % 3 == 0] # Keep only the moves
    uci = " ".join(uci)

    return {"game": uci}