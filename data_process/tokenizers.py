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
    def size(self):
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

    def decode(self, tokens: list) -> str:
        return " ".join(
            [
                self.itos[token]
                for token in tokens
                if token not in self.special_tokens_ids
            ]
        )
