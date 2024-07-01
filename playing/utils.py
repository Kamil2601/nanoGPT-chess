import chess


def board_to_piece_uci_moves(board: chess.Board):
    play_board = chess.Board()
    res = []
    for move in board.move_stack:
        piece = play_board.piece_at(move.from_square).symbol().upper()
        uci = move.uci()
        res.append(piece + uci)
        play_board.push(move)

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