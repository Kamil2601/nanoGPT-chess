import gc
import os
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_process.tokenizers import (FullMoveEloMaterialPairTokenizer,
                                     FullMoveEloMaterialTokenizer,
                                     FullMoveEloPieceCountTokenizer,
                                     FullMoveTokenizerNoEOS,
                                     FullMoveTokenizerWithElo)
from data_process.utils import (add_elo_and_piece_count_to_dataset,
                                join_material_tokens,
                                remove_last_player_material_token,
                                remove_material_tokens, row_for_base_training)
from nanoGPT.model import GPTConfig
from training.lightning_training import (GamesDataModule, GamesDataset, LightningGPT,
                                         LightningGPTWeighted,
                                         WeightedGamesDataModule,
                                         WeightsConfig)

### SETTINGS ###

# tokenizer = FullMoveTokenizerWithElo()
# tokenizer = FullMoveEloMaterialPairTokenizer()
# tokenizer = FullMoveEloMaterialTokenizer()
tokenizer = FullMoveEloPieceCountTokenizer()
# tokenizer = FullMoveTokenizerNoEOS()

block_size = 1024

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

learning_rate = 0.0002
batch_size = 32

num_workers = 14

ignore_first_n_targets = 1
training_target_step = 2 # 2 is for ignoring material prediction during loss calculation, otherwise should be 1


tensorboard_logger_version = None # SET TO NONE FOR FUTURE TRAININGS


tensorboard_logger_name = "piece_uci_4kk_train"
checkpoint_path = f"./models/{tensorboard_logger_name}"
tensorboard_dir = f"./lightning_logs/"

mask_elo_token = True

checkpoint = None
# checkpoint = "./models/full_training/elo_material_ignore_material_prediction/epoch=1-step=250000.ckpt"


##################

columns_to_load = [
    #"index",
    #"id",
    #"date",
    "white_elo",
    "black_elo",
    #"result",
    "ply",
    "ply_30s",
    "piece_uci"
]


# Define file paths and column names
data_files = {
    "train": "./data/csv/train_val_test/train.csv",
    "validation": "./data/csv/train_val_test/validation.csv",
}

def main():
    # Load both splits
    datasets = load_dataset(
        "csv",
        data_files=data_files,
        delimiter=",",
        usecols=columns_to_load,
        # num_proc=1,
    )

    print(datasets)

    columns_to_remove = [
        #"index",
        #"id",
        #"date",
        "white_elo",
        "black_elo",
        #"result",
        # "ply",
        "ply_30s",
        "piece_uci"
    ]

    datasets = datasets.map(add_elo_and_piece_count_to_dataset, num_proc=6, remove_columns=columns_to_remove)
    # datasets = datasets.map(row_for_base_training, num_proc=6, remove_columns=columns_to_remove)

    # cuts = list(games_df.ply_30s)

    # print(games[0])


    # print(datasets["train"][0])

    # exit()

    def encode_batch(batch):
        return {
            "input_ids": [tokenizer.encode(x)[:-1] for x in batch["game"]],
            "target_ids": [tokenizer.encode(x)[1:] for x in batch["game"]],
        }

    datasets = datasets.map(
        encode_batch,
        batched=True,          # ← VERY IMPORTANT (huge speedup)
        num_proc=6,
        remove_columns=["game"]
    )

    print(datasets["train"][0]["input_ids"])


    # datasets = datasets.filter(lambda x: len(x["input_ids"]) > 5, num_proc=num_workers)
    datasets = datasets.filter(lambda x: len(x["input_ids"]) <= block_size, num_proc=num_workers)


    data_module = data_module = GamesDataModule(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
        mask_elo_token=mask_elo_token,
    )

    print(next(iter(data_module.train_dataloader())))

    if checkpoint is None:
        pl_model = LightningGPT(
            model_config,
            learning_rate=learning_rate,
            training_ignore_first_n_targets=ignore_first_n_targets,
            training_target_step=training_target_step
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
        # precision="16",
        # default_root_dir=
        # fast_dev_run=True,
    )

    torch.set_float32_matmul_precision("medium")

    trainer.fit(
        model=pl_model,
        datamodule=data_module,
        ckpt_path=checkpoint,
    )


if __name__ == "__main__":
    main()