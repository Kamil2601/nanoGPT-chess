import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

from data_process.tokenizers import FullMoveTokenizerNoEOS
from lightning_training import (GamesDataModule, LightningGPT,
                                LightningGPTWeighted, WeightedGamesDataModule,
                                WeightsConfig)
from nanoGPT.model import GPTConfig

tokenizer = FullMoveTokenizerNoEOS()

model_config_small = GPTConfig(
    block_size=301,
    vocab_size=tokenizer.vocab_size,
    n_layer=4,
    n_head=4,
    n_embd=256,
    bias=False,
)

model_config_big = GPTConfig(
    block_size=301,
    vocab_size=tokenizer.vocab_size,
    n_layer=8,
    n_head=8,
    n_embd=512,
    bias=False,
)


### SETTINGS ###

val_size = 100000
model_config = model_config_big

learning_rate = 0.0001
batch_size = 64

num_workers = 8


weighted_training = False

use_elo = True
loss_weight = 0.5
draw_weight = 0.75
win_weight = 1

data_path = "./data/strong_play/standard.csv"


tensorboard_logger_name = "standard_big_normal"
checkpoint_path = "./models/standard_big_normal/"

# checkpoint = "./models/standard_small_normal/epoch=3-step=374992.ckpt"
checkpoint = None

##################



# games_df = pd.read_csv(data_path)

headers = ["index", "id", "date", "white_elo", "black_elo", "result", "ply", "ply_30s", "piece_uci"]

games_df = pd.read_csv("./data/csv/train.csv", delimiter=";", header=None, names=headers)
games_df = games_df[['result', 'white_elo', 'black_elo', 'piece_uci']]


if weighted_training:
    print("Training with weights")
    config = WeightsConfig(use_elo=True, loss_weight=loss_weight, draw_weight=draw_weight, win_weight=win_weight)
    data_module = WeightedGamesDataModule(games_df, weights_config=config, batch_size=batch_size, test_size=val_size, num_workers=num_workers)
    if checkpoint is None:
        pl_model = LightningGPTWeighted(model_config, learning_rate=learning_rate)
    else:
        pl_model = LightningGPTWeighted.load_from_checkpoint(checkpoint, config=model_config, learning_rate=learning_rate)
else:
    # training without weights
    print("Training without weights")
    games = list(games_df.piece_uci)
    data_module = data_module = GamesDataModule(games, batch_size=batch_size, test_size=val_size, num_workers=num_workers)
    if checkpoint is None:
        pl_model = LightningGPT(model_config, learning_rate=learning_rate)
    else:
        pl_model = LightningGPT.load_from_checkpoint(checkpoint, config=model_config, learning_rate=learning_rate)


checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path,
    save_top_k=-1,
    every_n_epochs=1
)

tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="./lightning_logs/strong_play_2/", name=tensorboard_logger_name)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=5,
    callbacks=[pl.callbacks.RichProgressBar(), checkpoint_callback],
    logger=tensorboard_logger,
    precision="bf16-mixed",
    # default_root_dir=
    # fast_dev_run=True
)

torch.set_float32_matmul_precision("high")

trainer.fit(
    model=pl_model,
    datamodule=data_module,
    ckpt_path=checkpoint
    # ckpt_path=ckpt_path
)



# ########### PRZEKLEJONY SKRYPT JESZCZE RAZ ###########

# pl_model.cpu()

# torch.cuda.empty_cache()


# ### SETTINGS ###

# val_size = 100000
# model_config = model_config_big

# learning_rate = 0.0001
# batch_size = 64

# num_workers = 8


# weighted_training = True

# use_elo = True
# loss_weight = 0.5
# draw_weight = 0.75
# win_weight = 1

# data_path = "./data/strong_play/standard.csv"


# tensorboard_logger_name = "standard_big_L05_D075_W1"
# checkpoint_path = "./models/standard_big_L05_D075_W1/"

# # checkpoint = "./models/standard_small_normal/epoch=3-step=374992.ckpt"
# checkpoint = None

# ##################



# if weighted_training:
#     print("Training with weights")
#     config = WeightsConfig(use_elo=True, loss_weight=loss_weight, draw_weight=draw_weight, win_weight=win_weight)
#     data_module = WeightedGamesDataModule(games_df, weights_config=config, batch_size=batch_size, test_size=val_size, num_workers=num_workers)
#     if checkpoint is None:
#         pl_model = LightningGPTWeighted(model_config, learning_rate=learning_rate)
#     else:
#         pl_model = LightningGPTWeighted.load_from_checkpoint(checkpoint, config=model_config, learning_rate=learning_rate)
# else:
#     # training without weights
#     print("Training without weights")
#     games = list(games_df.piece_uci)
#     data_module = data_module = GamesDataModule(games, batch_size=batch_size, test_size=val_size, num_workers=num_workers)
#     if checkpoint is None:
#         pl_model = LightningGPT(model_config, learning_rate=learning_rate)
#     else:
#         pl_model = LightningGPT.load_from_checkpoint(checkpoint, config=model_config, learning_rate=learning_rate)


# checkpoint_callback = ModelCheckpoint(
#     dirpath=checkpoint_path,
#     save_top_k=-1,
#     every_n_epochs=1
# )

# tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="./lightning_logs/strong_play_2/", name=tensorboard_logger_name)

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