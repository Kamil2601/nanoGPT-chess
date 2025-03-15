import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_process.tokenizers import (FullMoveTokenizerNoEOS,
                                     FullMoveTokenizerWithElo)
from data_process.utils import add_elo_token_to_games, remove_material_tokens
from lightning_training import (GamesDataModule, LightningGPT,
                                LightningGPTWeighted, WeightedGamesDataModule,
                                WeightsConfig)
from nanoGPT.model import GPTConfig

tokenizer = FullMoveTokenizerWithElo()

model_config_small = GPTConfig(
    block_size=302,
    vocab_size=tokenizer.vocab_size,
    n_layer=4,
    n_head=4,
    n_embd=256,
    bias=False,
)

model_config_big = GPTConfig(
    block_size=302,
    vocab_size=tokenizer.vocab_size,
    n_layer=8,
    n_head=8,
    n_embd=512,
    bias=False,
)


### SETTINGS ###

val_size = 0.1
model_config = model_config_big

learning_rate = 0.0001
batch_size = 64

num_workers = 8

ignore_first_n_targets = 1

data_path = "./data/csv/train.csv"


tensorboard_logger_name = "elo_training_2"
checkpoint_path = "./models/elo_training_2/"

# checkpoint = "./models/standard_small_normal/epoch=3-step=374992.ckpt"
checkpoint = None

##################


# games_df = pd.read_csv(data_path)

headers = [
    "index",
    "id",
    "date",
    "white_elo",
    "black_elo",
    "result",
    "ply",
    "ply_30s",
    "piece_uci",
]

games_df = pd.read_csv(data_path, delimiter=";", header=None, names=headers)
games_df = games_df[["result", "white_elo", "black_elo", "piece_uci"]]

games = remove_material_tokens(games_df.piece_uci)
games = add_elo_token_to_games(games, games_df.white_elo, games_df.black_elo)
games = list(games)


data_module = data_module = GamesDataModule(
    games,
    batch_size=batch_size,
    test_size=val_size,
    num_workers=num_workers,
    tokenizer=tokenizer,
)

if checkpoint is None:
    pl_model = LightningGPT(model_config, learning_rate=learning_rate, ignore_first_n_targets=ignore_first_n_targets)
else:
    pl_model = LightningGPT.load_from_checkpoint(
        checkpoint, config=model_config, learning_rate=learning_rate, ignore_first_n_targets=ignore_first_n_targets
    )


checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path, save_top_k=-1, every_n_epochs=1
)

tensorboard_logger = pl.loggers.TensorBoardLogger(
    save_dir="./lightning_logs/elo_training_2", name=tensorboard_logger_name
)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=5,
    callbacks=[pl.callbacks.RichProgressBar(), checkpoint_callback],
    logger=tensorboard_logger,
    precision="bf16-mixed",
    # default_root_dir=
    fast_dev_run=True,
)

torch.set_float32_matmul_precision("high")

trainer.fit(
    model=pl_model,
    datamodule=data_module,
    ckpt_path=checkpoint,
    # ckpt_path=ckpt_path
)

# ################# NO ELO TRAINING ###############################

# tensorboard_logger_name = "no_elo_training"
# checkpoint_path = "./models/no_elo_training/"

# # checkpoint = "./models/standard_small_normal/epoch=3-step=374992.ckpt"
# checkpoint = None

# # ##################


# # games_df = pd.read_csv(data_path)

# # headers = ["index", "id", "date", "white_elo", "black_elo", "result", "ply", "ply_30s", "piece_uci"]

# # games_df = pd.read_csv(data_path, delimiter=";", header=None, names=headers)
# # games_df = games_df[['result', 'white_elo', 'black_elo', 'piece_uci']]

# games = remove_material_tokens(games_df.piece_uci)
# games = tokenizer.unk_elo_token + " " + tokenizer.unk_elo_token + " " + games
# games = list(games)


# data_module = data_module = GamesDataModule(games, batch_size=batch_size, test_size=val_size, num_workers=num_workers, tokenizer=tokenizer)
# if checkpoint is None:
#     pl_model = LightningGPT(model_config, learning_rate=learning_rate)
# else:
#     pl_model = LightningGPT.load_from_checkpoint(checkpoint, config=model_config, learning_rate=learning_rate)


# checkpoint_callback = ModelCheckpoint(
#     dirpath=checkpoint_path,
#     save_top_k=-1,
#     every_n_epochs=1
# )

# tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="./lightning_logs/no_elo_training", name=tensorboard_logger_name)

# trainer = pl.Trainer(
#     accelerator="gpu",
#     max_epochs=5,
#     callbacks=[pl.callbacks.RichProgressBar(), checkpoint_callback],
#     logger=tensorboard_logger,
#     precision="bf16-mixed",
#     # default_root_dir=
#     # fast_dev_run=True
# )

# torch.set_float32_matmul_precision("high")

# trainer.fit(
#     model=pl_model,
#     datamodule=data_module,
#     ckpt_path=checkpoint
#     # ckpt_path=ckpt_path
# )
