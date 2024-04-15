import random
from collections import Counter

import chess.engine
import torch
from tqdm import tqdm
from transformers import AutoModel, GPT2LMHeadModel

from data_process.tokenizers import FullMoveTokenizerNoEOS, SquareTokenizer
from nanoGPT.model import GPT
from playing.utils import board_to_piece_uci_moves, legal_moves_piece_uci


class Agent:
    def __init__(self) -> None:
        pass

    def play(self, board: chess.Board):
        pass

class RandomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def play(self, board: chess.Board):
        legal_moves = list(board.legal_moves)

        if len(legal_moves) == 0:
            return None

        random_move = random.choice(legal_moves)

        return random_move

class GPTAgent(Agent):
    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model
        self.model.cpu()
        self.tokenizer = FullMoveTokenizerNoEOS()

    def play(self, board: chess.Board):
        game_str = board_to_piece_uci_moves(board)
        game_encoded = self.tokenizer.encode(game_str)

        legal_moves = legal_moves_piece_uci(board)
        legal_moves_encoded = self.tokenizer.encode(legal_moves)[1:]

        game_tensor = torch.tensor(game_encoded).unsqueeze(0)

        model_output, _ = self.model(game_tensor)

        legal_moves_scores = model_output[0, -1, legal_moves_encoded].softmax(-1)
        best_legal_move_index = legal_moves_scores.argmax()
        best_legal_move = legal_moves[best_legal_move_index]

        return chess.Move.from_uci(best_legal_move[1:])
    

class GPTBeamSearchAgent(Agent):
    def __init__(self, model: GPT, beam_width: int = 3, depth: int = 3) -> None:
        super().__init__()
        self.model = model
        self.model.cpu()
        self.tokenizer = FullMoveTokenizerNoEOS()
        self.beam_width = beam_width
        self.depth = depth

    def play(self, board: chess.Board):
        
        initial_candidate = ([], 1.0)  # Tuple of (move sequence, score)
        candidates = [initial_candidate]

        for _ in range(self.depth):
            new_candidates = []
            for candidate_seq, candidate_score in candidates:
                temp_board = board.copy()

                for move in candidate_seq:
                    temp_board.push(chess.Move.from_uci(move[1:]))

                game_str = board_to_piece_uci_moves(temp_board)
                game_encoded = self.tokenizer.encode(game_str)

                game_tensor = torch.tensor(game_encoded).unsqueeze(0)
                
                legal_moves = legal_moves_piece_uci(temp_board)
                legal_moves_encoded = self.tokenizer.encode(legal_moves)[1:]

                model_output, _ = self.model(game_tensor)

                legal_moves_scores = model_output[0, -1, legal_moves_encoded].softmax(-1)

                # Combine the candidate score and legal move scores
                combined_scores = (legal_moves_scores * candidate_score).tolist()

                # Add all legal moves with their combined scores to new_candidates
                for idx, move in enumerate(legal_moves):
                    new_candidate_seq = candidate_seq + [move]
                    new_candidate_score = combined_scores[idx]
                    new_candidates.append((new_candidate_seq, new_candidate_score))

            # Sort new_candidates based on combined scores and keep top beam_width candidates
            if len(new_candidates) == 0:
                break

            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:self.beam_width]

        # Select the move sequence with the highest score
        best_candidate_seq, _ = max(candidates, key=lambda x: x[1])
        best_legal_move = best_candidate_seq[0]

        return chess.Move.from_uci(best_legal_move[1:])
    

def board_to_uci_squares(board: chess.Board):
    res = []
    for move in board.move_stack:
        move_uci = move.uci()
        res.append(move_uci[:2])
        res.append(move_uci[2:4])
        if len(move_uci) == 5:
            res.append(move_uci[4])

    return res


class GPTSquareTokenAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('shtoshni/gpt2-chess-uci')
        self.model.cpu()
        self.tokenizer = SquareTokenizer()

    def play(self, board: chess.Board):
        pred_move_str = ""
        greedy_game_prefix = board_to_uci_squares(board)
        greedy_game_prefix = self.tokenizer.encode(greedy_game_prefix)[:-1]

        for idx in range(3):
            prefix_tens = torch.tensor(greedy_game_prefix).unsqueeze(0)

            logits = self.model(prefix_tens)[0]
            last_token_logit = logits[0, -1, :]

            token_idx = torch.argmax(last_token_logit).item()
            current_token = self.tokenizer.decode_token(token_idx)

            pred_move_str += current_token

            if idx == 0 and current_token == self.tokenizer.eos_token:
                break

            greedy_game_prefix.append(token_idx)

        if len(pred_move_str) == 6:
            pred_move_str = pred_move_str[:4]
        try:
            pred_move = chess.Move.from_uci(pred_move_str)
        except:
            pred_move = None

        if pred_move is None or pred_move not in board.legal_moves:
            pred_move = random.choice(list(board.legal_moves))

        return pred_move
