import io
import os
import re
from pathlib import Path

import chess.pgn


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
