import gc
import os
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_process.tokenizers import (FullMoveEloMaterialPairTokenizer,
                                     FullMoveEloMaterialTokenizer,
                                     FullMoveEloPieceCountTokenizer,
                                     FullMoveTokenizerNoEOS,
                                     FullMoveTokenizerWithElo)
from data_process.utils import (add_elo_token_to_games, join_material_tokens,
                                remove_last_player_material_token,
                                remove_material_tokens)
from lightning_training import (GamesDataModule, LightningGPT,
                                LightningGPTWeighted, WeightedGamesDataModule,
                                WeightsConfig)
from nanoGPT.model import GPTConfig

### SETTINGS ###

# tokenizer = FullMoveTokenizerWithElo()
# tokenizer = FullMoveEloMaterialPairTokenizer()
# tokenizer = FullMoveEloMaterialTokenizer()
tokenizer = FullMoveEloPieceCountTokenizer()

block_size = 604

model_config_small = GPTConfig(
    block_size=block_size,
    vocab_size=tokenizer.vocab_size,
    n_layer=4,
    n_head=4,
    n_embd=256,
    bias=False,
)

model_config_big = GPTConfig(
    block_size=block_size,
    vocab_size=tokenizer.vocab_size,
    n_layer=8,
    n_head=8,
    n_embd=512,
    bias=False,
)

small_training_size = {
    "train": 1_000_000,
    "val": 10_000,
}

full_training_size = {
    "train": None,
    "val": 100_000,
}

small_training_max_epochs = 5
full_training_max_epochs = 10

train_size = full_training_size["train"]
val_size = full_training_size["val"]
max_epochs = full_training_max_epochs


model_config = model_config_big

learning_rate = 0.0001
batch_size = 64

num_workers = 8

ignore_first_n_targets = 1
training_target_step = 2 # 2 is for ignoring material prediction during loss calculation, otherwise should be 1

data_path = "./data/csv/uniform_elo_distribution/train_piece_count.csv"
# data_path = "./data/test.csv"

max_game_length = block_size

tensorboard_logger_version = 0 # SET TO NONE FOR FUTURE TRAININGS


tensorboard_logger_name = "elo_piece_count_ignore_material_prediction"
checkpoint_path = f"./models/full_training/{tensorboard_logger_name}"
tensorboard_dir = f"./lightning_logs/full_training/"

mask_elo_token = True

checkpoint = None
# checkpoint = "./models/full_training/elo_material_ignore_material_prediction/epoch=1-step=250000.ckpt"


##################


headers = [
    #"index",
    #"id",
    #"date",
    "white_elo",
    "black_elo",
    #"result",
    #"ply",
    "ply_30s",
    "piece_uci",
]

if train_size is not None:
    games_df = pd.read_csv(data_path, delimiter=";", usecols=[3,4,7,8], header=None, names=headers, nrows=train_size + val_size)
else:
    games_df = pd.read_csv(data_path, delimiter=";", usecols=[3,4,7,8], header=None, names=headers)


# games_df = games_df[["result", "white_elo", "black_elo", "piece_uci", "ply_30s"]]

# games = remove_material_tokens(games_df.piece_uci)
# games = join_material_tokens(games_df.piece_uci, replace_bigger_values=True)
games = remove_last_player_material_token(games_df.piece_uci)

games = add_elo_token_to_games(games, games_df.white_elo, games_df.black_elo)
# games = games.apply(lambda x: x.split(" "))

games = list(games)
# cuts = list(games_df.ply_30s)

del games_df
gc.collect()

# print(games[0])


data_module = data_module = GamesDataModule(
    games,
    # cuts=cuts,
    batch_size=batch_size,
    validation_size=val_size,
    num_workers=num_workers,
    tokenizer=tokenizer,
    mask_elo_token=mask_elo_token,
    max_game_length=max_game_length,
)

if checkpoint is None:
    pl_model = LightningGPT(
        model_config,
        learning_rate=learning_rate,
        training_ignore_first_n_targets=ignore_first_n_targets,
        trainig_target_step=training_target_step
    )
else:
    pl_model = LightningGPT.load_from_checkpoint(
        checkpoint,
        config=model_config,
        learning_rate=learning_rate,
        training_ignore_first_n_targets=ignore_first_n_targets,
        training_target_step=training_target_step
    )


checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path, save_top_k=-1, every_n_epochs=1
)

tensorboard_logger = pl.loggers.TensorBoardLogger(
    save_dir=tensorboard_dir, name=tensorboard_logger_name, version=tensorboard_logger_version
)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=max_epochs,
    callbacks=[pl.callbacks.RichProgressBar(), checkpoint_callback],
    logger=tensorboard_logger,
    precision="bf16-mixed",
    # default_root_dir=
    # fast_dev_run=True,
)

torch.set_float32_matmul_precision("high")

trainer.fit(
    model=pl_model,
    datamodule=data_module,
    ckpt_path=checkpoint,
)