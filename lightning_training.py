from typing import Any

import pyarrow as pa
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from data_process.tokenizers import FullMoveTokenizer
from nanoGPT.model import GPT, GPTConfig

### PL MODELS ###


class LightningGPT(pl.LightningModule):
    def __init__(self, config: GPTConfig, learning_rate=1e-3, weight_decay=0.0):
        super().__init__()
        self.model = GPT(config)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.model(x, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.model(x, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer


### DATASETS ###


class GamesDataset(Dataset):
    def __init__(self, games, tokenizer, max_game_length=300):
        self.games = games
        self.encoded_games = [tokenizer.encode(game) for game in games]
        self.encoded_games = [
            game for game in self.encoded_games if len(game) <= max_game_length + 2
        ]
        self.encoded_games = pa.array(self.encoded_games)
        self.max_game_length = max_game_length

    def __len__(self):
        return len(self.encoded_games)

    @property
    def block_size(self):
        return self.max_game_length + 1

    def __getitem__(self, idx):
        encoded_game = self.encoded_games[idx].as_py()
        encoded_game = torch.tensor(encoded_game, dtype=torch.long)
        x = encoded_game[:-1]
        y = encoded_game[1:]
        return x, y


### PL DATA MODULES ###


def collate_fn(data):
    inputs = [d[0] for d in data]
    targets = [d[1] for d in data]

    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=0
    )
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )

    return padded_inputs, padded_targets


class GamesDataModule(pl.LightningDataModule):
    def __init__(
        self, games, tokenizer=None, batch_size=64, num_workers=8, max_game_length=300
    ) -> None:
        super().__init__()
        self.games = games
        self.train_games, self.val_games = train_test_split(games, test_size=0.1)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_game_length = max_game_length

        if tokenizer == None:
            tokenizer = FullMoveTokenizer()

        self.tokenizer = tokenizer

        self.train_dataset = GamesDataset(self.train_games, tokenizer=tokenizer)
        self.val_dataset = GamesDataset(self.val_games, tokenizer=tokenizer)

    @property
    def block_size(self):
        return self.max_game_length + 1

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
