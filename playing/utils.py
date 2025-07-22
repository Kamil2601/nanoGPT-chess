from typing import Literal

import chess


def board_to_piece_uci_moves(board: chess.Board, notes: Literal["none", "material", "piece_count"] = "none"):
    play_board = chess.Board()
    res = []
    for move in board.move_stack:
        piece = play_board.piece_at(move.from_square).symbol().upper()
        uci = move.uci()
        res.append(piece + uci)
        play_board.push(move)

        if notes == "material":
            white_material, black_material = material(play_board)

            if play_board.turn == chess.WHITE:
                res.append(str(white_material))
            else:
                res.append(str(black_material))

        elif notes == "piece_count":
            white_token, black_token = piece_count_token(play_board)

            if play_board.turn == chess.WHITE:
                res.append(white_token)
            else:
                res.append(black_token)

    return res

def piece_uci_to_board(piece_uci_moves: list):
    board = chess.Board()
    for move in piece_uci_moves:
        board.push(chess.Move.from_uci(move[1:]))

    return board

def legal_moves_piece_uci(board: chess.Board):
    res = []
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square).symbol().upper()
        uci = move.uci()
        res.append(piece + uci)

    return res

def moves_piece_uci(board: chess.Board, moves: list):
    res = []
    for move in moves:
        piece = board.piece_at(move.from_square).symbol().upper()
        uci = move.uci()
        res.append(piece + uci)

    return res


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

def piece_count_token(board, max_count={"Q": 2, "R": 3, "B": 3, "N": 3, "P": 8}):
    white_material, black_material = piece_count(board)
    white_token = "".join(
        f"{piece}{min(count, max_count[piece])}" for piece, count in white_material.items()
    )
    black_token = "".join(
        f"{piece}{min(count, max_count[piece])}" for piece, count in black_material.items()
    )
    return white_token, black_token


def material_balance(board):
    white, black = material(board)
    return white - black
