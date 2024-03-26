from typing import Any

import numpy as np
import pyarrow as pa
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics.classification import MulticlassAccuracy

from data_process.tokenizers import FullMoveTokenizer, FullMoveTokenizerNoEOS
from nanoGPT.model import GPT, GPTConfig

### PL MODELS ###


class LightningGPT(pl.LightningModule):
    def __init__(self, config: GPTConfig, learning_rate=1e-3, weight_decay=0.0, tokenizer=None):
        super().__init__()
        self.model = GPT(config)

        if tokenizer == None:
            tokenizer = FullMoveTokenizer()

        self.tokenizer = tokenizer

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.test_accuracy = MulticlassAccuracy(tokenizer.vocab_size, ignore_index=0, average="micro")

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.model(x, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.model(x, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.model(x, y)
        self.log("test_loss", loss, prog_bar=True)
        y_pred = torch.argmax(output, dim=-1)

        self.test_accuracy.update(y_pred[:,10:], y[:,10:])
        # self.log("test_accuracy", accuracy, prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_accuracy.compute())
        self.test_accuracy.reset()
        

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
        # return encoded_game


class CutGamesDataset(Dataset):
    def __init__(self, games, cuts, max_game_length=300):
        self.tokenizer = FullMoveTokenizerNoEOS()
        self.games = games
        self.cuts = cuts
        self.games_cuts = zip(games, cuts)

        self.encoded_games = [
            self.tokenizer.encode(game, cut) for game, cut in self.games_cuts
        ]
        self.encoded_games = [
            game for game in self.encoded_games if len(game) <= max_game_length + 2
        ]

        self.encoded_games = pa.array(self.encoded_games)
        self.max_game_length = max_game_length

    def __len__(self):
        return len(self.encoded_games)
    
    def __getitem__(self, idx):
        encoded_game = self.encoded_games[idx].as_py()
        encoded_game = torch.tensor(encoded_game, dtype=torch.long)
        x = encoded_game[:-1]
        y = encoded_game[1:]
        return x, y


### SAMPLERS ###


class SimilarLengthSequenceBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.index_length = [(i, len(data_source[i])) for i in range(len(data_source))]

    def __iter__(self):
        np.random.shuffle(self.index_length)
        self.index_length.sort(key=lambda x: x[1])
        batch_indices = list(range(len(self)))
        np.random.shuffle(batch_indices)

        for batch_ind in batch_indices:
            batch_start = batch_ind * self.batch_size
            batch_end = min((batch_ind + 1) * self.batch_size, len(self.data_source))
            yield [self.index_length[i][0] for i in range(batch_start, batch_end)]

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


### PL DATA MODULES ###


def pad_sequence(sequences, padding_value=0):
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = [
        torch.nn.functional.pad(
            seq, (0, max_length - len(seq)), mode="constant", value=padding_value
        )
        for seq in sequences
    ]
    padded_sequences = torch.stack(padded_sequences)

    return padded_sequences


def collate_fn(data):
    inputs = [d[0] for d in data]
    targets = [d[1] for d in data]

    padded_inputs = pad_sequence(inputs, padding_value=0)
    padded_targets = pad_sequence(targets, padding_value=0)

    return padded_inputs, padded_targets


class GamesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        games,
        tokenizer=None,
        batch_size=64,
        num_workers=8,
        max_game_length=300,
        test_size=0.05,
    ) -> None:
        super().__init__()
        self.games = games
        self.train_games, self.val_games = train_test_split(
            games, test_size=test_size, random_state=42
        )

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
        batch_sampler = SimilarLengthSequenceBatchSampler(
            self.train_dataset.encoded_games,
            batch_size=self.batch_size,
            drop_last=False,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
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


class DataModuleMaiaTraining(pl.LightningDataModule):
    def __init__(
        self, dataset_dict, batch_size=64, num_workers=8, max_game_length=300
    ) -> None:
        super().__init__()

        self.dataset_dict = dataset_dict

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_game_length = max_game_length


        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def setup(self, stage):
        if stage == "fit":
            if self.train_dataset is None:
                self.train_dataset = CutGamesDataset(
                    self.dataset_dict["train"]["piece_uci"],
                    self.dataset_dict["train"]["ply_30s"],
                    max_game_length=self.max_game_length,
                )
            if self.val_dataset is None:
                self.val_dataset = CutGamesDataset(
                    self.dataset_dict["valid"]["piece_uci"],
                    self.dataset_dict["valid"]["ply_30s"],
                    max_game_length=self.max_game_length,
                )

        if stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = CutGamesDataset(
                    self.dataset_dict["valid"]["piece_uci"],
                    self.dataset_dict["valid"]["ply_30s"],
                    max_game_length=self.max_game_length,
                )

        if stage == "test":
            if self.test_dataset is None:
                self.test_dataset = CutGamesDataset(
                    self.dataset_dict["test"]["piece_uci"],
                    self.dataset_dict["test"]["ply_30s"],
                    max_game_length=self.max_game_length,
                )


    @property
    def block_size(self):
        return self.max_game_length + 1

    def train_dataloader(self) -> Any:
        batch_sampler = SimilarLengthSequenceBatchSampler(
            self.train_dataset.encoded_games,
            batch_size=self.batch_size,
            drop_last=False,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
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

    def test_dataloader(self) -> Any:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
