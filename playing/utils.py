import chess


def board_to_piece_uci_moves(board: chess.Board, include_material=False):
    play_board = chess.Board()
    res = []
    for move in board.move_stack:
        piece = play_board.piece_at(move.from_square).symbol().upper()
        uci = move.uci()
        res.append(piece + uci)
        play_board.push(move)

        if include_material:
            white_material, black_material = material(play_board)

            if play_board.turn == chess.WHITE:
                res.append(str(white_material))
            else:
                res.append(str(black_material))

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

def material_balance(board):
    white, black = material(board)
    return white - black