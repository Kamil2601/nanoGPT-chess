

from collections import Counter

import chess
import pandas as pd
from IPython.display import display
from tqdm import tqdm

from data_process.vocabulary import PieceMove
from playing.agents import Agent, RandomAgent
from playing.utils import material, material_balance, piece_uci_to_board


def play_game(white: Agent, black: Agent, n_moves=40, verbose=False, board = None):
    if not board:
        board = chess.Board()

    if verbose:
        display(board)

    if board.turn == chess.BLACK:
        move = black.play(board)
        board.push(move)

        if verbose:
            display(board)

    n_moves -= len(board.move_stack)//2

    for i in range(n_moves):
        for agent in [white, black]:
            if board.is_game_over():
                break

            move = agent.play(board)

            board.push(move)

            if verbose:
                # print(board.fen())
                display(board)

    return board

engine = chess.engine.SimpleEngine.popen_uci("stockfish")

def simple_result(board, time=1):
    score = engine.analyse(board, chess.engine.Limit(time=time))["score"].white()

    if score <= chess.engine.Cp(-100):
        return -1
    elif score < chess.engine.Cp(100):
        return 0
    return 1


def test_agent(agent, opponent=None, n_games=50, n_moves=40, time=1, verbose=True):
    if not opponent:
        opponent = RandomAgent()

    white_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board = play_game(agent, opponent, n_moves=n_moves)
        score = simple_result(final_board, time=time)
        white_scores.append(score)

    white_stats = Counter(white_scores)

    if verbose:
        print(white_stats)

    black_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board = play_game(opponent, agent, n_moves=n_moves)
        score = simple_result(final_board, time=time)
        black_scores.append(score)

    black_stats = Counter(black_scores)

    if verbose:
        print(black_stats)

    return {"white": white_stats, "black": black_stats}

def play_game_with_material_count(
    white: Agent, black: Agent, n_moves=40, verbose=False
):
    board = chess.Board()

    if verbose:
        display(board)

    white_materials = []
    black_materials = []

    white_material, black_material = material(board)
    white_materials.append(white_material)
    black_materials.append(black_material)

    for i in range(n_moves):
        if board.is_game_over():
                break
        for agent in [white, black]:
            if board.is_game_over():
                break

            move = agent.play(board)

            board.push(move)

            if verbose:
                display(board)

        white_material, black_material = material(board)
        white_materials.append(white_material)
        black_materials.append(black_material)

    return board, white_materials, black_materials


def test_agent_material_count(agent, opponent=None, n_games=50, n_moves=40, time=1):
    if not opponent:
        opponent = RandomAgent()
    results = []

    agent_materials = []
    opponent_materials = []

    white_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board, white_materials, black_materials = play_game_with_material_count(
            agent, opponent, n_moves=40
        )
        score = simple_result(final_board, time=time)
        white_scores.append(score)

        agent_materials.append(white_materials)
        opponent_materials.append(black_materials)

    white_stats = Counter(white_scores)
    print(white_stats)

    black_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board, white_materials, black_materials = play_game_with_material_count(
            opponent, agent, n_moves=40
        )
        score = simple_result(final_board, time=time)
        black_scores.append(score)

        agent_materials.append(black_materials)
        opponent_materials.append(white_materials)

    black_stats = Counter(black_scores)
    print(black_stats)

    agent_materials = list(pd.DataFrame(agent_materials).mean(axis=0, skipna=True))
    opponent_materials = list(pd.DataFrame(opponent_materials).mean(axis=0, skipna=True))

    return {"white": white_stats, "black": black_stats}, agent_materials, opponent_materials


def test_agent_in_openings(agent, opponent, openings, n_moves=40, time=1, verbose=False):
    white_scores = []
    black_scores = []

    for opening in tqdm(openings):
        board = piece_uci_to_board(opening)
        
        final_board_white = play_game(agent, opponent, board = board.copy(), n_moves=n_moves)
        score = simple_result(final_board_white, time=time)
        white_scores.append(score)

        final_board_black = play_game(opponent, agent, board = board.copy(), n_moves=n_moves)
        score = simple_result(final_board_black, time=time)
        black_scores.append(score)

    white_stats = Counter(white_scores)
    black_stats = Counter(black_scores)


    return {"white": white_stats, "black": black_stats}


def is_legal_move(board: chess.Board, move: PieceMove):
    move_no_piece = chess.Move.from_uci(move.uci())

    if move_no_piece not in board.legal_moves:
        return False
    
    piece_at = board.piece_at(move.from_square)

    if piece_at is None:
        return False

    if move.piece_type == piece_at.piece_type and board.turn == piece_at.color:
        return True
    
    return False



def play_against_agent(agent, color=chess.WHITE):
    board = chess.Board()
    display(board)
    while not board.is_game_over():
        if board.turn != color:
            move = agent.play(board)
            board.push(move)
        else:
            while True:
                try:
                    move = input("Enter move (UCI): ")
                    if move == "q":
                        return
                    move = chess.Move.from_uci(move)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move")
                except:
                    print("Illegal move")
        display(board)
    print("Game over")
    print(board.result())




