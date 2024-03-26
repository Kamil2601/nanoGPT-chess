from .vocabulary import legal_chess_moves


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

    def tokenize(self, text: str) -> list:
        pass

    def encode(self, text: str) -> list:
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

    def encode(self, text: str) -> list:
        return (
            [self.bos_token_id]
            + [self.stoi[move] for move in self.tokenize(text)]
            + [self.eos_token_id]
        )

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

    def tokenize(self, text: str, cut = -1) -> list:
        split = text.split()
        if cut >= 0:
            return split[:cut]
        return split

    def encode(self, text: str, cut = -1) -> list:
        return (
            [self.bos_token_id]
            + [self.stoi[move] for move in self.tokenize(text, cut)]
        )

