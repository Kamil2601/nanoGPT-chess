import itertools

import chess

SQUARE_PAIRS = list(itertools.product(chess.SQUARES, chess.SQUARES))
SQUARE_VOCAB = [
    "[PAD]",
    "[BOS]",
    "[EOS]",
    "P",
    "N",
    "R",
    "B",
    "Q",
    "K",
    "c2",
    "c4",
    "e7",
    "e5",
    "g2",
    "g3",
    "b8",
    "c6",
    "f1",
    "g8",
    "f6",
    "b1",
    "c3",
    "f8",
    "b4",
    "d5",
    "e8",
    "e2",
    "e3",
    "e4",
    "a2",
    "a3",
    "d6",
    "d8",
    "a1",
    "e6",
    "b2",
    "b3",
    "c7",
    "d2",
    "d3",
    "f7",
    "f5",
    "a7",
    "a5",
    "g1",
    "d4",
    "g4",
    "c5",
    "e1",
    "d7",
    "d1",
    "h8",
    "c1",
    "a4",
    "h2",
    "h3",
    "a8",
    "f4",
    "g7",
    "g5",
    "b6",
    "b7",
    "c8",
    "h7",
    "h4",
    "f3",
    "h5",
    "h1",
    "f2",
    "h6",
    "g6",
    "b5",
    "a6",
    "q",
    "r",
    "b",
    "n",
]

MATERIAL_VOCAB = [str(val) for val in range(41)]
MATERIAL_VOCAB_EXTENDED = [str(val) for val in range(104)]
MATERIAL_PAIR_VOCAB = [
    f"{white}|{black}"
    for (white, black) in itertools.product(MATERIAL_VOCAB, MATERIAL_VOCAB)
]
ELO_VOCAB = [str(val) for val in range(0, 4001, 100)] + ["[UNK_ELO]"]

queen_count = ["Q" + str(i) for i in range(0, 3)]
rook_count = ["R" + str(i) for i in range(0, 4)]
bishop_count = ["B" + str(i) for i in range(0, 4)]
knight_count = ["N" + str(i) for i in range(0, 4)]
pawn_count = ["P" + str(i) for i in range(0, 9)]

PIECE_COUNT_VOCAB = [
    q + r + b + k + p
    for q in queen_count
    for r in rook_count
    for b in bishop_count
    for k in knight_count
    for p in pawn_count
]


class PieceMove(chess.Move):
    def __init__(self, from_square, to_square, promotion=None, piece_type=None):
        super().__init__(from_square, to_square, promotion)
        self.piece_type = piece_type

    @classmethod
    def from_uci(cls, uci: str):
        piece_type = chess.PIECE_SYMBOLS.index(uci[0].lower())
        from_square = chess.SQUARE_NAMES.index(uci[1:3])
        to_square = chess.SQUARE_NAMES.index(uci[3:5])
        promotion = None

        if len(uci) == 6:
            promotion = chess.PIECE_SYMBOLS.index(uci[5].lower())

        return cls(from_square, to_square, promotion=promotion, piece_type=piece_type)

    def __repr__(self) -> str:
        return f"{str(self)}"

    def __str__(self) -> str:
        return chess.piece_symbol(self.piece_type).upper() + self.uci()


def is_valid_rook_move(start, end):
    start_rank, start_file = start // 8, start % 8
    end_rank, end_file = end // 8, end % 8

    if (start_rank == end_rank or start_file == end_file) and start != end:
        return True
    return False


def is_valid_bishop_move(start, end):
    start_rank, start_file = start // 8, start % 8
    end_rank, end_file = end // 8, end % 8

    if abs(start_rank - end_rank) == abs(start_file - end_file) and start != end:
        return True
    return False


def is_valid_queen_move(start, end):
    return is_valid_rook_move(start, end) or is_valid_bishop_move(start, end)


def is_valid_knight_move(start, end):
    start_rank, start_file = start // 8, start % 8
    end_rank, end_file = end // 8, end % 8

    if (abs(start_rank - end_rank) == 1 and abs(start_file - end_file) == 2) or (
        abs(start_rank - end_rank) == 2 and abs(start_file - end_file) == 1
    ):
        return True
    return False


def is_valid_king_move(start, end):
    start_rank, start_file = start // 8, start % 8
    end_rank, end_file = end // 8, end % 8

    if (
        abs(start_rank - end_rank) <= 1
        and abs(start_file - end_file) <= 1
        and start != end
    ):
        return True
    return False


def is_valid_pawn_move(start, end):
    start_rank, start_file = start // 8, start % 8
    end_rank, end_file = end // 8, end % 8

    if start_rank in [0, 7] or end_rank in [0, 7]:
        return False

    if start_file == end_file and abs(start_rank - end_rank) == 1:
        return True

    if abs(start_file - end_file) == 1 and abs(start_rank - end_rank) == 1:
        return True

    if start_file == end_file and (
        (start_rank == 6 and end_rank == 4) or (start_rank == 1 and end_rank == 3)
    ):
        return True

    return False


def is_valid_promotion_move(start, end):
    start_rank, start_file = start // 8, start % 8
    end_rank, end_file = end // 8, end % 8

    if abs(start_file - end_file) > 1:
        return False

    if start_rank == 6 and end_rank == 7:
        return True
    if start_rank == 1 and end_rank == 0:
        return True

    return False


def legal_rook_moves():
    return [
        PieceMove(src, dest, piece_type=chess.ROOK)
        for src, dest in SQUARE_PAIRS
        if is_valid_rook_move(src, dest)
    ]


def legal_bishop_moves():
    return [
        PieceMove(src, dest, piece_type=chess.BISHOP)
        for src, dest in SQUARE_PAIRS
        if is_valid_bishop_move(src, dest)
    ]


def legal_queen_moves():
    return [
        PieceMove(src, dest, piece_type=chess.QUEEN)
        for src, dest in SQUARE_PAIRS
        if is_valid_queen_move(src, dest)
    ]


def legal_knight_moves():
    return [
        PieceMove(src, dest, piece_type=chess.KNIGHT)
        for src, dest in SQUARE_PAIRS
        if is_valid_knight_move(src, dest)
    ]


def legal_king_moves():
    return [
        PieceMove(src, dest, piece_type=chess.KING)
        for src, dest in SQUARE_PAIRS
        if is_valid_king_move(src, dest)
    ]


def castles():
    return [
        PieceMove(chess.E1, chess.G1, piece_type=chess.KING),
        PieceMove(chess.E1, chess.C1, piece_type=chess.KING),
        PieceMove(chess.E8, chess.G8, piece_type=chess.KING),
        PieceMove(chess.E8, chess.C8, piece_type=chess.KING),
    ]


def legal_pawn_moves():
    return [
        PieceMove(src, dest, piece_type=chess.PAWN)
        for src, dest in SQUARE_PAIRS
        if is_valid_pawn_move(src, dest)
    ]


def legal_promotions():
    return [
        PieceMove(src, dest, piece_type=chess.PAWN, promotion=promotion)
        for src, dest, promotion in itertools.product(
            chess.SQUARES,
            chess.SQUARES,
            [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT],
        )
        if is_valid_promotion_move(src, dest)
    ]


def legal_chess_moves():
    return (
        legal_king_moves()
        + castles()
        + legal_queen_moves()
        + legal_rook_moves()
        + legal_bishop_moves()
        + legal_knight_moves()
        + legal_pawn_moves()
        + legal_promotions()
    )
