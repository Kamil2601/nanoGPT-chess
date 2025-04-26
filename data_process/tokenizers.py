import re
from collections import OrderedDict

from .vocabulary import (ELO_VOCAB, MATERIAL_PAIR_VOCAB, MATERIAL_VOCAB,
                         SQUARE_VOCAB, legal_chess_moves)


class Tokenizer:
    def __init__(self) -> None:
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        self.vocab = [self.pad_token, self.bos_token, self.eos_token]

    @property
    def special_tokens(self):
        return [self.pad_token, self.bos_token, self.eos_token]

    @property
    def special_tokens_ids(self):
        return [self.pad_token_id, self.bos_token_id, self.eos_token_id]

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def decode_token(self, token_idx):
        return self.itos[token_idx]

    def tokenize(self, text: str) -> list:
        pass

    def encode(self, text: str, add_special_tokens = True) -> list:
        pass

    def decode(self, tokens: list) -> str:
        pass


class FullMoveTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.vocab += [str(move) for move in legal_chess_moves()]
        self.stoi = {move: i for i, move in enumerate(self.vocab)}
        self.itos = {i: move for i, move in enumerate(self.vocab)}

    def tokenize(self, text: str) -> list:
        return text.split()

    def encode(self, text: str, add_special_tokens = True) -> list:
        main_tokens = [self.stoi[move] for move in self.tokenize(text)]

        if add_special_tokens:
            return [self.bos_token_id] + main_tokens + [self.eos_token_id]
        
        return main_tokens


    def decode(self, tokens: list, keep_special_tokens = False) -> str:
        if keep_special_tokens:
            return " ".join([self.itos[token] for token in tokens])
        else:
            for i, token in enumerate(tokens):
                if token in [self.eos_token_id, self.pad_token_id]:
                    tokens = tokens[:i]
                    break
            return " ".join([self.itos[token] for token in tokens if token not in self.special_tokens_ids])
        

class FullMoveTokenizerNoEOS(FullMoveTokenizer):
    def __init__(self) -> None:
        super().__init__()

    def tokenize(self, moves, cut = -1) -> list:
        if type(moves) == str:
            moves = moves.split()
    
        if cut >= 0:
            return moves[:cut]
        return moves

    def encode(self, text: str, add_special_tokens = True) -> list:
        main_tokens = [self.stoi[move] for move in self.tokenize(text)]

        if add_special_tokens:
            return [self.bos_token_id] + main_tokens
        
        return main_tokens
    
class FullMoveTokenizerWithElo(FullMoveTokenizerNoEOS):
    def __init__(self):
        super().__init__()
        self.vocab += ELO_VOCAB
        self.stoi = {move: i for i, move in enumerate(self.vocab)}
        self.itos = {i: move for i, move in enumerate(self.vocab)}

        self.unk_elo_token = self.vocab[-1]
        self.unk_elo_token_id = self.vocab_size - 1

    def encode(self, text: str, add_special_tokens = True) -> list:
        main_tokens = [self.stoi[move] for move in self.tokenize(text)]

        # if add_special_tokens:
        #     return [self.bos_token_id] + main_tokens
        
        return main_tokens
    
class FullMoveEloMaterialPairTokenizer(FullMoveTokenizerWithElo):
    def __init__(self):
        super().__init__()
        self.vocab += MATERIAL_PAIR_VOCAB
        self.stoi = {move: i for i, move in enumerate(self.vocab)}
        self.itos = {i: move for i, move in enumerate(self.vocab)}

    def encode(self, text: str, add_special_tokens = True) -> list:
        main_tokens = [self.stoi[move] for move in self.tokenize(text)]

        # if add_special_tokens:
        #     return [self.bos_token_id] + main_tokens
        
        return main_tokens


class SquareTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        
        self.vocab = SQUARE_VOCAB
        self.stoi = {move: i for i, move in enumerate(self.vocab)}
        self.itos = self.vocab
    
    def get_vocab(self):
        return self.vocab

    def encode_token(self, token):
        return self.vocab[token]

    def decode_token(self, token_idx):
        return self.itos[token_idx]
    
    def tokenize(self, game_str):
        if isinstance(game_str, list):
            game_moves = game_str
        else:
            game_moves = game_str.split()
        
        res = []

        for move in game_moves:
            move_tokens = [move[i:i+2] for i in range(0, len(move), 2)]
            res.extend(move_tokens)

        return res


    def encode(self, game_str, add_special_tokens=True):
        tokens = self.tokenize(game_str)
        token_ids = [self.vocab.index(token) for token in tokens]

        if add_special_tokens:
            return [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids
        

