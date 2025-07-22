import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from data_process import tokenizers
from nanoGPT.model import GPTConfig

material_pair_tokenizer = tokenizers.FullMoveEloMaterialPairTokenizer()

material_tokenizer = tokenizers.FullMoveEloMaterialTokenizer()

piece_count_tokenizer = tokenizers.FullMoveEloPieceCountTokenizer()

base_tokenizer = tokenizers.FullMoveTokenizerWithElo()

no_elo_tokenizer = tokenizers.FullMoveTokenizerNoEOS()

# Model with elo tokens and material pair "white_material|black_material" token after each move
# Example: "1800 [UNK_ELO] Pe2e4 39|39 Pd7d5 39|39 Pe4d5 39|38"
material_pair_model_config = GPTConfig(
    block_size=604,
    vocab_size=len(material_pair_tokenizer.vocab),
    n_layer=8,
    n_head=8,
    n_embd=512,
    bias=False,
)

# white_elo black_elo white_move black_material black_move white_material
# Example: "1800 [UNK_ELO] Pe2e4 39 Pd7d5 39 Pe4d5 38"
material_model_config = GPTConfig(
    block_size=604,
    vocab_size=material_tokenizer.vocab_size,
    n_layer=8,
    n_head=8,
    n_embd=512,
    bias=False,
)

# white_elo black_elo white_move black_piece_count black_move white_piece_count
# Example: "1800 [UNK_ELO] Pe2e4 Q1R2B2N2P8 Pd7d5 Q1R2B2N2P8 Pe4d5 Q1R2B2N2P7"
piece_count_model_config = GPTConfig(
    block_size=604,
    vocab_size=piece_count_tokenizer.vocab_size,
    n_layer=8,
    n_head=8,
    n_embd=512,
    bias=False,
)

# white_elo black_elo white_move black_move
# Example: "1800 [UNK_ELO] Pe2e4 Pd7d5 Pe4d5"
base_model_config = GPTConfig(
    block_size=302,
    vocab_size=base_tokenizer.vocab_size,
    n_layer=8,
    n_head=8,
    n_embd=512,
    bias=False,
)

# [BOS] white_move black_move
# Example: "[BOS] Pe2e4 Pd7d5 Pe4d5"
no_elo_config = GPTConfig(
    block_size=302,
    vocab_size=no_elo_tokenizer.vocab_size,
    n_layer=8,
    n_head=8,
    n_embd=512,
    bias=False,
)