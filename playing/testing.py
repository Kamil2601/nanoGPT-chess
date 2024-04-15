

from collections import Counter

import chess
from IPython.display import display
from tqdm import tqdm

from playing.agents import Agent, RandomAgent


def play_game(white: Agent, black: Agent, n_moves=40, verbose=False):
    board = chess.Board()

    if verbose:
        display(board)

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


def test_agent(agent, opponent=None, n_games=50, n_moves=40, time=1):
    if not opponent:
        opponent = RandomAgent()

    white_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board = play_game(agent, opponent, n_moves=40)
        score = simple_result(final_board, time=time)
        white_scores.append(score)

    white_stats = Counter(white_scores)
    print(white_stats)

    black_scores = []
    for game in tqdm(range(n_games // 2)):
        final_board = play_game(opponent, agent, n_moves=40)
        score = simple_result(final_board, time=time)
        black_scores.append(score)

    black_stats = Counter(black_scores)
    print(black_stats)

    return {"white": white_stats, "black": black_stats}


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