import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset

from data_process.tokenizers import FullMoveTokenizerNoEOS
from lightning_training import (LightningGPTWeighted, WeightedGamesDataModule,
                                WeightsConfig)
from nanoGPT.model import GPTConfig

n_rows = 6 * 10**6

standard = pd.read_csv("../data/lichess_database/lichess_db_standard_rated_2024-02.csv", nrows=n_rows * 1.2, delimiter=';')
elite = pd.read_csv("../data/lichess_elite_database.csv", nrows=n_rows, delimiter=';')

standard = standard[((standard.result == '1-0') & (standard.white_elo >= 1200)) |
                    ((standard.result == '0-1') & (standard.black_elo >= 1200)) |
                    ((standard.result == '1/2-1/2') & (standard.white_elo >= 1200) & (standard.white_elo >= 1200))]


games_df = pd.concat([standard, elite], ignore_index=True)
games_df = games_df.dropna()
games_df = games_df[['result', 'white_elo', 'black_elo', 'piece_uci']]


config = WeightsConfig(use_elo=True)
data_module = WeightedGamesDataModule(games_df, weights_config=config, batch_size=64, test_size=0.01, num_workers=8)


model_config = GPTConfig(
    block_size=301,
    vocab_size=len(data_module.tokenizer.vocab),
    n_layer=8,
    n_head=8,
    n_embd=512,
    bias=False,
)

pl_model = LightningGPTWeighted(model_config)

tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="./lightning_logs/strong_play/", name=f"W1-D0.5-L0")

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=10,
    callbacks=[pl.callbacks.RichProgressBar()],
    logger=tensorboard_logger,
    precision="bf16-mixed",
    # default_root_dir=
    # fast_dev_run=True
)

torch.set_float32_matmul_precision("high")

trainer.fit(
    model=pl_model,
    datamodule=data_module,
    # ckpt_path=ckpt_path
)