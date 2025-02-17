import chess


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