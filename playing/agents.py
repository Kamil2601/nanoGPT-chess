import random
from collections import Counter

import chess.engine
import torch
from tqdm import tqdm
from transformers import AutoModel, GPT2LMHeadModel

from data_process.tokenizers import (FullMoveEloMaterialTokenizer,
                                     FullMoveTokenizerNoEOS,
                                     FullMoveTokenizerWithElo, SquareTokenizer)
from data_process.vocabulary import PieceMove
from nanoGPT.model import GPT
from playing.utils import (board_to_piece_uci_moves, legal_moves_piece_uci,
                           material, material_balance, moves_piece_uci)


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
        legal_moves_encoded = self.tokenizer.encode(legal_moves, add_special_tokens=False)

        game_tensor = torch.tensor(game_encoded).unsqueeze(0)

        model_output, _ = self.model(game_tensor)

        legal_moves_scores = model_output[0, -1, legal_moves_encoded].softmax(-1)
        best_legal_move_index = legal_moves_scores.argmax()
        best_legal_move = legal_moves[best_legal_move_index]

        return chess.Move.from_uci(best_legal_move[1:])

class GPTEloAgent(Agent):
    def __init__(self, model: GPT, model_elo, use_material_tokens = False) -> None:
        super().__init__()
        self.model = model
        self.model.cpu()
        self.use_material_tokens = use_material_tokens
        self.model_elo = str(model_elo // 100 * 100)
        
        if use_material_tokens:
            self.tokenizer = FullMoveEloMaterialTokenizer()
        else:
            self.tokenizer = FullMoveTokenizerWithElo()

    def play(self, board: chess.Board):
        game_str = board_to_piece_uci_moves(board, include_material=self.use_material_tokens)

        if board.turn == chess.WHITE:
            game_str = [self.model_elo, self.tokenizer.unk_elo_token] + game_str
        else:
            game_str = [self.tokenizer.unk_elo_token, self.model_elo] + game_str

        game_encoded = self.tokenizer.encode(game_str)

        legal_moves = legal_moves_piece_uci(board)
        legal_moves_encoded = self.tokenizer.encode(legal_moves)


        with torch.inference_mode():
            game_tensor = torch.tensor(game_encoded).unsqueeze(0)

            model_output, _ = self.model(game_tensor)

            legal_moves_scores = model_output[0, -1, legal_moves_encoded].softmax(-1)
            best_legal_move_index = legal_moves_scores.argmax().item()
            best_legal_move = legal_moves[best_legal_move_index]

            return chess.Move.from_uci(best_legal_move[1:])
    


class GPTNocheckAgent(Agent):
    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model
        self.model.cpu()
        self.tokenizer = FullMoveTokenizerNoEOS()

    def play(self, board: chess.Board):
        game_str = board_to_piece_uci_moves(board)
        game_encoded = self.tokenizer.encode(game_str)

        game_tensor = torch.tensor(game_encoded).unsqueeze(0)

        model_output, _ = self.model(game_tensor)

        moves_scores = model_output[0, -1, :].softmax(-1)
        best_move_index = moves_scores.argmax().item()
        best_move = self.tokenizer.decode_token(best_move_index)

        return PieceMove.from_uci(best_move)
    

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

class UCIEngineAgent(Agent):
    def __init__(self, engine_path: str, limit: chess.engine.Limit = None, config = None) -> None:
        super().__init__()
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        if config:
            self.engine.configure(config)
        self.limit = limit
        self.engine.play(chess.Board(), limit=self.limit)

    def play(self, board: chess.Board):
        result = self.engine.play(board, limit=self.limit)
        return result.move


class GPTSquareTokenAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('shtoshni/gpt2-chess-uci')
        self.model.cpu()
        self.tokenizer = SquareTokenizer()
        self.random_move_count = 0

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
            self.random_move_count += 1

        return pred_move



### MINMAX AGENTS ###

class NegaMaxAgent(Agent):
    def __init__(self, depth = 3) -> None:
        super().__init__()
        self.depth = depth

    def play(self, board: chess.Board):
        best_score = -float('inf')
        best_moves = []
        alpha = -float('inf')
        beta = float('inf')
        
        for move in self.ordered_moves(board):
            board.push(move)
            score = -self.negamax(board, self.depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
            alpha = max(alpha, score)
        
        return self.choose_from_best_moves(board, best_moves)


    def negamax(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board, alpha, beta)
        
        max_score = -float('inf')
        for move in self.ordered_moves(board):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha > beta:
                break
        return max_score
    

    def ordered_moves(self, board):
        return board.legal_moves
    
    def choose_from_best_moves(self, board, best_moves):
        pass
    
    # Evaluate the board from the perspective of the current player
    def evaluate_board(self, board, alpha = None, beta = None):
        pass


class NegaMaxMaterialAgent(NegaMaxAgent):
    def __init__(self, depth = 3) -> None:
        super().__init__(depth)

    def evaluate_board(self, board):
        color = 1 if board.turn == chess.WHITE else -1
        return color * material_balance(board)
    
    def choose_from_best_moves(self, board, best_moves):
        return random.choice(best_moves)
    

class NegaMaxMaterialGPTAgent(NegaMaxMaterialAgent):
    def __init__(self, model, depth=3) -> None:
        super().__init__(depth)
        self.model = model
        self.model.cpu()
        self.tokenizer = FullMoveTokenizerNoEOS()

    def choose_from_best_moves(self, board, best_moves):
        game_str = board_to_piece_uci_moves(board)
        game_encoded = self.tokenizer.encode(game_str)

        moves = moves_piece_uci(board, best_moves)
        moves_encoded = self.tokenizer.encode(moves, add_special_tokens=False)

        game_tensor = torch.tensor(game_encoded).unsqueeze(0)

        model_output, _ = self.model(game_tensor)

        moves_scores = model_output[0, -1, moves_encoded].softmax(-1)
        best_move_index = moves_scores.argmax()
        best_move = moves[best_move_index]

        return chess.Move.from_uci(best_move[1:])
    

class NegaMaxMaterialGPTEloAgent(NegaMaxMaterialAgent):
    def __init__(self, model, depth=3, model_elo = 1500, use_material_tokens = False) -> None:
        super().__init__(depth)
        self.model = model
        self.model.cpu()
        self.tokenizer = FullMoveTokenizerNoEOS()
        self.model_elo = str(model_elo // 100 * 100)
        self.use_material_tokens = use_material_tokens

        if use_material_tokens:
            self.tokenizer = FullMoveEloMaterialTokenizer()
        else:
            self.tokenizer = FullMoveTokenizerWithElo()

    def evaluate_board(self, board, alpha = None, beta = None):
        return self.quiescent_search(board, alpha, beta)
    
    def quiescent_search(self, board, alpha, beta):
        all_possible_captures = [move for move in board.legal_moves if is_favorable_move(board, move)]

        if len(all_possible_captures) == 0:
            return self.static_evaluate(board)

        max_score = -float('inf')
        for move in all_possible_captures:
            board.push(move)
            score = -self.quiescent_search(board, -beta, -alpha)
            board.pop()
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha > beta:
                break
        return max_score
    
    def static_evaluate(self, board):
        color = 1 if board.turn == chess.WHITE else -1
        return color * material_balance(board)
    

    def choose_from_best_moves(self, board, best_moves):    
        game_str = board_to_piece_uci_moves(board, include_material=self.use_material_tokens)

        if board.turn == chess.WHITE:
            game_str = [self.model_elo, self.tokenizer.unk_elo_token] + game_str
        else:
            game_str = [self.tokenizer.unk_elo_token, self.model_elo] + game_str

        game_encoded = self.tokenizer.encode(game_str)

        moves = moves_piece_uci(board, best_moves)
        moves_encoded = self.tokenizer.encode(moves)

        with torch.inference_mode():
            game_tensor = torch.tensor(game_encoded).unsqueeze(0)

            model_output, _ = self.model(game_tensor)

            moves_scores = model_output[0, -1, moves_encoded].softmax(-1)
            best_move_index = moves_scores.argmax().item()
            best_move = moves[best_move_index]

            return chess.Move.from_uci(best_move[1:])
        

def is_favorable_move(board: chess.Board, move: chess.Move) -> bool:
    if move.promotion is not None:
        return True
    if board.is_capture(move) and not board.is_en_passant(move):
        if get_piece_val(board.piece_type_at(move.from_square)) < get_piece_val(
            board.piece_type_at(move.to_square)
        ) or len(board.attackers(board.turn, move.to_square)) > len(
            board.attackers(not board.turn, move.to_square)
        ):
            return True
    if board.is_en_passant(move):
        return True
    return False


def get_piece_val(piece):
    if(piece == None):
        return 0
    value = 0
    if piece == "P" or piece == "p":
        value = 10
    if piece == "N" or piece == "n":
        value = 30
    if piece == "B" or piece == "b":
        value = 30
    if piece == "R" or piece == "r":
        value = 50
    if piece == "Q" or piece == "q":
        value = 90
    if piece == 'K' or piece == 'k':
        value = 900
    #value = value if (board.piece_at(place)).color else -value
    return value