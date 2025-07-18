from dataclasses import dataclass
from typing import Any, Optional

import datasets
import numpy as np
import pyarrow as pa
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity

from data_process.tokenizers import (FullMoveTokenizer, FullMoveTokenizerNoEOS,
                                     Tokenizer)
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
        acc_n_bins=30,
        acc_bin_range=10,
        test_start_token=10,
        test_token_step=1,
        training_ignore_first_n_targets=0,
        training_target_step=1,
        masked_elo_test=False,
        use_elo=True
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
        self.training_target_step = training_target_step
        self.masked_elo_test = masked_elo_test

        self.test_accuracy = MulticlassAccuracy(
            tokenizer.vocab_size, ignore_index=self.tokenizer.pad_token_id, average="micro"
        )

        self.use_elo = use_elo

        self.accuracy_per_elo = {}

        self.test_perplexity = Perplexity(ignore_index=self.tokenizer.pad_token_id)


        self.test_acc_bins = [
            MulticlassAccuracy(
                tokenizer.vocab_size, ignore_index=self.tokenizer.pad_token_id, average="micro"
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
            x,
            y,
            ignore_first_n_targets=self.training_ignore_first_n_targets,
            target_step=self.trainig_target_step,
        )
        return output, loss

    def forward_batch_test(self, batch):
        x, y = batch
        output, loss = self.model(
            x,
            y,
            ignore_first_n_targets=self.test_start_token,
            target_step=self.test_token_step,
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
        batch_dict = False
        if isinstance(batch, dict):
            # If the batch is a dictionary, extract the input_ids and target_ids
            inputs = batch["input_ids"]
            targets = batch["target_ids"]
            white_elos = batch["white_elo"]
            black_elos = batch["black_elo"]
            batch = (inputs, targets)
            batch_dict = True

        if self.masked_elo_test:
            logits, target_moves = self.logits_and_targets_for_masked_elo(batch)
        else:
            logits, target_moves = self.logits_and_targets_no_mask(batch)

        predicted_moves = torch.argmax(logits, dim=-1)

        self.test_accuracy.update(
            predicted_moves,
            target_moves,
        )

        self.test_perplexity.update(
            logits,
            target_moves
        )

        predicted_moves = predicted_moves.to(self.device)
        target_moves = target_moves.to(self.device)

        for i in range(self.acc_n_bins):
            start = i * self.acc_bin_range
            end = (i + 1) * self.acc_bin_range
            self.test_acc_bins[i].update(predicted_moves[:, start:end].cpu(), target_moves[:, start:end].cpu())

        
        target_white_moves = target_moves[:, ::2, ...].to(self.device)
        target_black_moves = target_moves[:, 1::2, ...].to(self.device)

        predicted_white_moves = predicted_moves[:, ::2, ...].to(self.device)
        predicted_black_moves = predicted_moves[:, 1::2, ...].to(self.device)

        inputs = batch[0].to(self.device)

        for i in range(len(batch)):
            if batch_dict:
                white_elo = white_elos[i]
                black_elo = black_elos[i]
            else:
                white_elo = self.tokenizer.decode_token(inputs[i][0].item())
                black_elo = self.tokenizer.decode_token(inputs[i][1].item())

            white_elo_key = f"{white_elo}"
            black_elo_key = f"{black_elo}"
            white_black_key = f"{white_elo_key}_{black_elo_key}"
            black_white_key = f"{black_elo_key}_{white_elo_key}"

            for key in [white_elo_key, black_elo_key, white_black_key, black_white_key]:
                if key not in self.accuracy_per_elo:
                    self.accuracy_per_elo[key] = MulticlassAccuracy(
                        self.tokenizer.vocab_size, ignore_index=self.tokenizer.pad_token_id, average="micro"
                    ).to(self.device)


            self.accuracy_per_elo[white_elo_key].update(
                predicted_white_moves[i],
                target_white_moves[i],
            )
            self.accuracy_per_elo[black_elo_key].update(
                predicted_black_moves[i],
                target_black_moves[i],
            )

            self.accuracy_per_elo[white_black_key].update(
                predicted_white_moves[i],
                target_white_moves[i],
            )

            self.accuracy_per_elo[black_white_key].update(
                predicted_black_moves[i],
                target_black_moves[i],
            )


    

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_accuracy.compute())
        self.test_accuracy.reset()

        self.log("test_perplexity", self.test_perplexity.compute())
        self.test_perplexity.reset()

        for elo_key, accuracy in self.accuracy_per_elo.items():
            self.log(f"test_acc_elo_{elo_key}", accuracy.compute())
            accuracy.reset()

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
    
    def standard_test_step(self, batch):
        x, y = batch

        output, loss = self.forward_batch_test(batch)

        self.log("test_loss", loss, prog_bar=True)
        y_pred = torch.argmax(output, dim=-1)

        self.test_accuracy.update(
            y_pred[:, self.test_start_token :: self.test_token_step, ...],
            y[:, self.test_start_token :: self.test_token_step, ...],
        )

    def logits_and_targets_for_masked_elo(self, batch):
        x, y = batch
        x_black_elo_masked = x.clone()
        x_black_elo_masked[:, 1, ...] = self.tokenizer.unk_elo_token_id

        x_white_elo_masked = x.clone()
        x_white_elo_masked[:, 0, ...] = self.tokenizer.unk_elo_token_id

        batch_black_elo_masked = (x_black_elo_masked, y)
        batch_white_elo_masked = (x_white_elo_masked, y)

        output_black_elo_masked, _  = self.forward_batch_test(batch_black_elo_masked)
        output_white_elo_masked, _  = self.forward_batch_test(batch_white_elo_masked)

        output_white_moves = output_black_elo_masked[:, self.test_start_token :: self.test_token_step, ...][:, ::2, ...]
        output_black_moves = output_white_elo_masked[:, self.test_start_token :: self.test_token_step, ...][:, 1::2, ...]

        min_len = min(output_white_moves.size(1), output_black_moves.size(1))

        output_white_moves_cut = output_white_moves[:, :min_len, :]
        output_black_moves_cut = output_black_moves[:, :min_len, :]

        stacked = torch.stack((output_white_moves_cut, output_black_moves_cut), dim=2)

        output_all_moves = stacked.reshape(output_white_moves.size(0), -1, output_white_moves.size(2))

        if output_white_moves.size(1) > output_black_moves.size(1):
            last_white = output_white_moves[:, -1:, :]  # (batch, 1, vocab)
            output_all_moves = torch.cat((output_all_moves, last_white), dim=1)  # (batch, 2*min_len + 1, vocab)

        target_moves = y[:, self.test_start_token :: self.test_token_step, ...]

        return output_all_moves, target_moves
    
    def logits_and_targets_no_mask(self, batch):
        x, y = batch
        output, _ = self.forward_batch_test(batch)

        output_all_moves = output[:, self.test_start_token :: self.test_token_step, ...]
        target_moves = y[:, self.test_start_token :: self.test_token_step, ...]

        return output_all_moves, target_moves



class LightningGPTWeighted(LightningGPT):
    def forward_batch(self, batch):
        x, y, weights = batch
        return self.model(x, y, weights)


### DATASETS ###


class GamesDataset(Dataset):
    def __init__(self, games, tokenizer, mask_elo_token=False):
        self.games = games
        self.encoded_games = games.map(lambda row: {"game": tokenizer.encode(row["game"])}, num_proc=6)
        self.tokenizer = tokenizer
        self.mask_elo_token = mask_elo_token

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        encoded_game = self.encoded_games[idx]["game"]
        encoded_game = torch.tensor(encoded_game, dtype=torch.long)

        if self.mask_elo_token:
            index_to_mask = np.random.choice([0, 1])
            encoded_game[index_to_mask] = self.tokenizer.unk_elo_token_id

        x = encoded_game[:-1]
        y = encoded_game[1:]
        return x, y
    

class TestGamesDataset(Dataset):
    def __init__(self, games, tokenizer, mask_elo_token=False):
        self.games = games
        self.encoded_games = games.map(lambda row: {"game": tokenizer.encode(row["game"])}, num_proc=6)
        self.tokenizer = tokenizer
        self.mask_elo_token = mask_elo_token

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        row = self.games[idx]
        encoded_game = self.encoded_games[idx]["game"]
        encoded_game = torch.tensor(encoded_game, dtype=torch.long)

        if self.mask_elo_token:
            index_to_mask = np.random.choice([0, 1])
            encoded_game[index_to_mask] = self.tokenizer.unk_elo_token_id

        x = encoded_game[:-1]
        y = encoded_game[1:]

        white_elo = int(row["white_elo"]) // 100 * 100
        black_elo = int(row["black_elo"]) // 100 * 100

        return {
            "input_ids": x,
            "target_ids": y,
            "white_elo": white_elo,
            "black_elo": black_elo
        }


class WeightedGamesDataset(Dataset):
    def __init__(
        self, games_df, weights_config=None, tokenizer=None
    ):
        if tokenizer == None:
            tokenizer = FullMoveTokenizerNoEOS()
        if weights_config == None:
            weights_config = WeightsConfig()

        games_df = games_df[games_df["result"].isin(["1-0", "0-1", "1/2-1/2"])]

        self.games_df = games_df
        self.weights_config = weights_config

        self.encoded_games = [tokenizer.encode(game) for game in games_df["piece_uci"]]

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
        if isinstance(data_source[0], int):
            self.index_length = [(i, data_source[i]) for i in range(len(data_source))]
        else:
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
    inputs = [d["input_ids"] for d in data]
    targets = [d["target_ids"] for d in data]
    white_elo = [int(d["white_elo"]) for d in data]
    black_elo = [int(d["black_elo"]) for d in data]
    # # result = [d["result"] for d in data]

    padded_inputs = pad_sequence(inputs, padding_value=0)
    padded_targets = pad_sequence(targets, padding_value=0)
    # result = torch.stack(result)

    # return inputs, targets

    return {
        "input_ids": padded_inputs,
        "target_ids": padded_targets,
        "white_elo": white_elo,
        "black_elo": black_elo,
        # "result": result
    }

def test_dataloader(games, tokenizer, batch_size=32, num_workers=8):
    dataset = TestGamesDataset(games, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_info, num_workers=num_workers)

class GamesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets = None,
        tokenizer=None,
        batch_size=64,
        num_workers=12,
        collate_fn=collate_fn,
        mask_elo_token=False,
    ) -> None:
        super().__init__()
        
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mask_elo_token = mask_elo_token

        if tokenizer == None:
            tokenizer = FullMoveTokenizerNoEOS()

        self.tokenizer = tokenizer
        self.collate_fn = collate_fn

        # self.games = games
        # self.test_games = test_games
        
        if "train" in datasets:
            self.train_dataset = GamesDataset(
                datasets["train"],
                tokenizer=tokenizer,
                mask_elo_token=mask_elo_token,
            )

        if "validation" in datasets:
            self.val_dataset = GamesDataset(
                datasets["validation"],
                tokenizer=tokenizer,
                mask_elo_token=mask_elo_token,
            )

        if "test" in datasets:
            self.test_dataset = GamesDataset(
                datasets["test"],
                tokenizer=tokenizer,
                mask_elo_token=mask_elo_token,
            )

    def train_dataloader(self) -> Any:
        batch_sampler = SimilarLengthSequenceBatchSampler(
            self.datasets["train"]["ply"],
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

    def test_dataloader(self) -> Any:
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
        test_size=0.05,
        collate_fn=collate_fn_with_weights,
    ) -> None:
        super().__init__()

        train_games, val_games = train_test_split(
            games, test_size=test_size, random_state=42
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

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
