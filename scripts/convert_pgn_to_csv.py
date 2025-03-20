import csv
import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.pgn
from tqdm import tqdm
from utils import material

input_file_path = "./data/lichess_db_standard_rated_2024-12.pgn"
output_file_path = "./data/lichess_db_standard_rated_2024-12.csv"
num_proc = 15

add_material_info = True



def read_n_to_last_line(filename, n = 1):
    """Returns the nth before last line of a file (n=1 gives last line)"""
    num_newlines = 0
    with open(filename, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)    
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b'\n':
                    num_newlines += 1
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line


def game_to_csv_row(game: chess.pgn.Game | str):
    if isinstance(game, str):
        game = chess.pgn.read_game(io.StringIO(game))

    if 'Bullet' in game.headers['Event']:
        return None

    game_id = game.headers['Site'].split("/")[-1]
    white_elo = int(game.headers['WhiteElo'])
    black_elo = int(game.headers['BlackElo'])
    result = game.headers['Result']
    date = game.headers['UTCDate']

    board = game.board()
    moves_uci = []
    less_than_30_seconds_move = -1

    mainline = list(game.mainline())
    game_length = len(mainline)

    # print(game_length)

    if game_length == 0:
        # print('EMPTY GAME')
        return None


    for i, node in enumerate(mainline):
        if node.clock() and node.clock() < 30 and less_than_30_seconds_move == -1:
            less_than_30_seconds_move  = i

        move = node.move

        piece = board.piece_at(move.from_square).symbol().upper()
        uci = move.uci()
        moves_uci.append(piece + uci)
        board.push(move)

        if add_material_info:
            material_white, material_black = material(board)
            moves_uci += [str(material_white), str(material_black)]

    uci_str = " ".join(moves_uci)

    return game_id, date, white_elo, black_elo, result, game_length, less_than_30_seconds_move, uci_str


def game_to_csv_row_with_index(game_with_index: tuple[chess.pgn.Game, int]):
    game, index = game_with_index

    try:
        row = game_to_csv_row(game)
        if row:
            row = (index,) + row
        return row
    except:
        return None

    
def read_game(f):
    game = []
    moves_read = False
    for line in f:
        game.append(line)
        if line.startswith("1."):
            moves_read = True

        if line == "\n" and moves_read:
            return "".join(game)
    return None

def convert_file(input_file_path, output_file_path, skip_games = False):
    game_index = 0

    if os.path.exists(output_file_path) and skip_games:
        last_game = read_n_to_last_line(output_file_path, 1)
        if len(last_game) > 2:
            try:
                game_index = int(last_game.split(";")[0])
            except:
                pass


    with open(input_file_path, 'r') as input_file, open(output_file_path, 'a') as output_file:
        writer = csv.writer(output_file, delimiter=';')

        if skip_games:
            for i in tqdm(range(game_index), desc="Skipping games"):
                line = input_file.readline()
                while not line.startswith("1."):
                    line = input_file.readline()

        while True:
            games_to_process = []
            for i in range(100000):
                game = read_game(input_file)

                if game is None:
                    break

                games_to_process.append((str(game), game_index))
                game_index += 1

            with ProcessPoolExecutor(max_workers=num_proc) as executor:
                rows = executor.map(game_to_csv_row_with_index, games_to_process)
                    
            rows = [row for row in rows if row]

            if len(rows) == 0:
                break

            writer.writerows(rows)

            print(game_index)


def main():
    input_files_dir = Path("./data/pgn-elite-older")
    input_files = input_files_dir.glob("*.pgn")

    output_file = Path("./data/csv/raw/lichess_elite_database-older.csv")

    for input_file_path in input_files:
        print(f"Converting {input_file_path} to csv")
        convert_file(input_file_path, output_file, skip_games=False)


if __name__ == "__main__":
    main()