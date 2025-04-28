from dataclasses import dataclass
from typing import Any, Optional

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

@dataclass
class WeightsConfig:
    loss_weight: float = 0.0
    win_weight: float = 1.0
    draw_weight: float = 0.5

    use_elo: bool = False
    elo_bias: float = -1000
    elo_ratio: float = 1 / 1000

    def compute_weight(self, elo, result):
        if self.use_elo:
            elo_weight = (elo + self.elo_bias) * self.elo_ratio
            elo_weight = max(0, elo_weight)
        else:
            elo_weight = 1

        if result == 1:
            return self.win_weight * elo_weight
        elif result == 0:
            return self.loss_weight * elo_weight
        else:
            return self.draw_weight * elo_weight

    def __str__(self) -> str:
        return str(self.__dict__)


class LightningGPT(pl.LightningModule):
    def __init__(
        self,
        config: GPTConfig,
        learning_rate=1e-3,
        weight_decay=0.0,
        tokenizer=None,
        test_acc_timesteps=False,
        acc_n_bins=30,
        acc_bin_range=10,
        test_start_token=10,
        test_token_step=1,
        training_ignore_first_n_targets=0,
        trainig_target_step=1,
    ):
        super().__init__()
        self.model = GPT(config)

        if tokenizer == None:
            tokenizer = FullMoveTokenizer()

        self.tokenizer = tokenizer

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_start_token = test_start_token
        self.test_token_step = test_token_step
        self.training_ignore_first_n_targets = training_ignore_first_n_targets
        self.trainig_target_step = trainig_target_step

        self.test_accuracy = MulticlassAccuracy(
            tokenizer.vocab_size, ignore_index=0, average="micro"
        )

        self.test_acc_timesteps = test_acc_timesteps

        if self.test_acc_timesteps:
            self.test_acc_bins = [
                MulticlassAccuracy(
                    tokenizer.vocab_size, ignore_index=0, average="micro"
                ).cpu()
                for _ in range(acc_n_bins)
            ]

        self.acc_n_bins = acc_n_bins
        self.acc_bin_range = acc_bin_range

    def forward(self, x, y=None):
        return self.model(x, y)

    def forward_batch(self, batch):
        x, y = batch
        return self.model(x, y)
    
    def forward_batch_training_validation(self, batch):
        x, y = batch
        output, loss = self.model(
            x, y, ignore_first_n_targets=self.training_ignore_first_n_targets, target_step=self.trainig_target_step
        )
        return output, loss

    def training_step(self, batch, batch_idx):
        output, loss = self.forward_batch_training_validation(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output, loss = self.forward_batch_training_validation(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.forward_batch_training_validation(batch)

        self.log("test_loss", loss, prog_bar=True)
        y_pred = torch.argmax(output, dim=-1)

        self.test_accuracy.update(
            y_pred[:, self.test_start_token::self.test_token_step, ...], y[:, self.test_start_token::self.test_token_step, ...]
        )

        y = y.cpu()
        y_pred = y_pred.cpu()

        if self.test_acc_timesteps:
            for i in range(self.acc_n_bins):
                start = i * self.acc_bin_range
                end = (i + 1) * self.acc_bin_range
                self.test_acc_bins[i].update(y_pred[:, start:end], y[:, start:end])

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_accuracy.compute())
        self.test_accuracy.reset()

        if self.test_acc_timesteps:
            for i in range(self.acc_n_bins):
                self.log(
                    f"test_acc_ply_{i * self.acc_bin_range+1}-{(i+1)*self.acc_bin_range}",
                    self.test_acc_bins[i].compute(),
                )
                self.test_acc_bins[i].reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer


class LightningGPTWeighted(LightningGPT):
    def forward_batch(self, batch):
        x, y, weights = batch
        return self.model(x, y, weights)


### DATASETS ###


class GamesDataset(Dataset):
    def __init__(self, games, tokenizer, max_game_length=300, mask_elo_token=False):
        self.games = games
        self.encoded_games = [tokenizer.encode(game) for game in games]
        self.encoded_games = [
            game for game in self.encoded_games if len(game) <= max_game_length + 2
        ]
        self.encoded_games = pa.array(self.encoded_games)
        self.max_game_length = max_game_length
        self.tokenizer = tokenizer
        self.mask_elo_token = mask_elo_token

    def __len__(self):
        return len(self.encoded_games)

    @property
    def block_size(self):
        return self.max_game_length + 1

    def __getitem__(self, idx):
        encoded_game = self.encoded_games[idx].as_py()
        encoded_game = torch.tensor(encoded_game, dtype=torch.long)

        if self.mask_elo_token:
            index_to_mask = np.random.choice([0, 1])
            encoded_game[index_to_mask] = self.tokenizer.unk_elo_token_id

        x = encoded_game[:-1]
        y = encoded_game[1:]
        return x, y


class CutGamesDataset(Dataset):
    def __init__(
        self, games, cuts, tokenizer=None, max_game_length=300, mask_elo_token=False
    ):
        if tokenizer == None:
            self.tokenizer = FullMoveTokenizerNoEOS()
        else:
            self.tokenizer = tokenizer

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
        self.mask_elo_token = mask_elo_token

    def __len__(self):
        return len(self.encoded_games)

    def __getitem__(self, idx):
        encoded_game = self.encoded_games[idx].as_py()
        encoded_game = torch.tensor(encoded_game, dtype=torch.long)

        if self.mask_elo_token:
            index_to_mask = np.random.choice([0, 1])
            encoded_game[index_to_mask] = self.tokenizer.unk_elo_token_id

        x = encoded_game[:-1]
        y = encoded_game[1:]
        return x, y


class WeightedGamesDataset(Dataset):
    def __init__(
        self, games_df, weights_config=None, tokenizer=None, max_game_length=300
    ):
        if tokenizer == None:
            tokenizer = FullMoveTokenizerNoEOS()
        if weights_config == None:
            weights_config = WeightsConfig()

        games_df = games_df[games_df["result"].isin(["1-0", "0-1", "1/2-1/2"])]

        self.games_df = games_df
        self.weights_config = weights_config

        self.encoded_games = [tokenizer.encode(game) for game in games_df["piece_uci"]]

        if max_game_length is not None:
            self.encoded_games = [
                game for game in self.encoded_games if len(game) <= max_game_length + 2
            ]

        self.encoded_games = pa.array(self.encoded_games)
        self.white_elo = torch.tensor(list(games_df["white_elo"]), dtype=torch.float32)
        self.black_elo = torch.tensor(list(games_df["black_elo"]), dtype=torch.float32)

        result_to_num = {"1-0": 1, "0-1": 0, "1/2-1/2": 0.5}
        self.result = [result_to_num[result] for result in games_df["result"]]
        self.result = torch.tensor(self.result, dtype=torch.float32)

    def __len__(self):
        return len(self.encoded_games)

    def __getitem__(self, idx):
        encoded_game = self.encoded_games[idx].as_py()
        encoded_game = torch.tensor(encoded_game, dtype=torch.long)
        x = encoded_game[:-1]
        y = encoded_game[1:]

        weights = torch.zeros_like(y, dtype=torch.float32)

        result = self.result[idx].item()
        white_elo = self.white_elo[idx].item()
        black_elo = self.black_elo[idx].item()

        weights[::2] = self.weights_config.compute_weight(white_elo, result)
        weights[1::2] = self.weights_config.compute_weight(black_elo, 1 - result)

        return x, y, weights


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


def collate_fn_with_weights(data):
    inputs = [d[0] for d in data]
    targets = [d[1] for d in data]
    weights = [d[2] for d in data]

    padded_inputs = pad_sequence(inputs, padding_value=0)
    padded_targets = pad_sequence(targets, padding_value=0)
    weights = pad_sequence(weights, padding_value=0)

    return padded_inputs, padded_targets, weights


def collate_fn_with_info(data):
    inputs = [d[0] for d in data]
    targets = [d[1] for d in data]
    white_elo = [d[2] for d in data]
    black_elo = [d[3] for d in data]
    result = [d[4] for d in data]

    padded_inputs = pad_sequence(inputs, padding_value=0)
    padded_targets = pad_sequence(targets, padding_value=0)
    white_elo = torch.stack(white_elo)
    black_elo = torch.stack(black_elo)
    result = torch.stack(result)

    return padded_inputs, padded_targets, white_elo, black_elo, result


class GamesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        games=None,
        cuts=None,
        test_games=None,
        tokenizer=None,
        batch_size=64,
        num_workers=12,
        max_game_length=300,
        validation_size=0.05,
        collate_fn=collate_fn,
        mask_elo_token=False,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_game_length = max_game_length
        self.mask_elo_token = mask_elo_token

        if tokenizer == None:
            tokenizer = FullMoveTokenizerNoEOS()

        self.tokenizer = tokenizer
        self.collate_fn = collate_fn

        self.games = games
        self.test_games = test_games

        if cuts is None:
            if games:
                self.train_games, self.val_games = train_test_split(
                    games, test_size=validation_size, random_state=42
                )

                self.train_dataset = GamesDataset(
                    self.train_games,
                    tokenizer=tokenizer,
                    max_game_length=max_game_length,
                    mask_elo_token=mask_elo_token,
                )
                self.val_dataset = GamesDataset(
                    self.val_games,
                    tokenizer=tokenizer,
                    max_game_length=max_game_length,
                    mask_elo_token=mask_elo_token,
                )

            if test_games:
                self.test_dataset = GamesDataset(
                    test_games,
                    tokenizer=tokenizer,
                    max_game_length=max_game_length,
                    mask_elo_token=mask_elo_token,
                )

        else:
            if games:
                self.train_games, self.val_games = train_test_split(
                    games, test_size=validation_size, random_state=42
                )

                self.train_dataset = CutGamesDataset(
                    self.train_games,
                    cuts,
                    tokenizer=tokenizer,
                    max_game_length=max_game_length,
                    mask_elo_token=mask_elo_token,
                )

                self.val_dataset = CutGamesDataset(
                    self.val_games,
                    cuts,
                    tokenizer=tokenizer,
                    max_game_length=max_game_length,
                    mask_elo_token=mask_elo_token,
                )

            if test_games:
                self.test_dataset = CutGamesDataset(
                    test_games,
                    cuts,
                    tokenizer=tokenizer,
                    max_game_length=max_game_length,
                    mask_elo_token=mask_elo_token,
                )

    @property
    def block_size(self):
        return self.max_game_length + 1

    def train_dataloader(self) -> Any:
        if not self.games:
            return None

        batch_sampler = SimilarLengthSequenceBatchSampler(
            self.train_dataset.encoded_games,
            batch_size=self.batch_size,
            drop_last=False,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Any:
        if not self.games:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Any:
        if not self.test_games:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class WeightedGamesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        games,
        tokenizer=None,
        weights_config=None,
        batch_size=64,
        num_workers=12,
        max_game_length=300,
        test_size=0.05,
        collate_fn=collate_fn_with_weights,
    ) -> None:
        super().__init__()

        train_games, val_games = train_test_split(
            games, test_size=test_size, random_state=42
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_game_length = max_game_length

        if tokenizer == None:
            tokenizer = FullMoveTokenizerNoEOS()

        self.tokenizer = tokenizer
        self.collate_fn = collate_fn

        self.train_dataset = WeightedGamesDataset(
            train_games, tokenizer=tokenizer, weights_config=weights_config
        )
        self.val_dataset = WeightedGamesDataset(
            val_games, tokenizer=tokenizer, weights_config=weights_config
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
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
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
