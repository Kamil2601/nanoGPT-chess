from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

csv_dir = "./data/lichess_database/"
save_dir = "./data/huggingface_datasets/elo_bins/"
num_proc = 15

train_size = 6 * 10**6
valid_size = 10**5
test_size = 10**5


def sample_train_valid_test(dataset, train_size, valid_size, test_size):
    sample = dataset.shuffle(seed=42).select(range(train_size + valid_size + test_size))
    train_valid_test = sample.train_test_split(
        test_size=test_size + valid_size, train_size=train_size, seed=42
    )
    test_valid = train_valid_test["test"].train_test_split(test_size=test_size, seed=42)
    train_test_valid_dataset = {
        "train": train_valid_test["train"],
        "valid": test_valid["train"],
        "test": test_valid["test"],
    }
    return DatasetDict(train_test_valid_dataset)


def main():
    print("Creating ELO bins datasets")

    games_dir = Path(csv_dir)
    games_files = games_dir.glob("*.csv")
    games_files = [str(f) for f in games_files]

    games_dataset = load_dataset(
        "csv",
        split="train",
        name="lichess_db_standard_rated",
        data_files=games_files,
        delimiter=";",
        num_proc=num_proc,
    )
    games_dataset = games_dataset.filter(lambda x: x["ply"] > 10, num_proc=num_proc)

    print(games_dataset)

    elo_bins = list(range(1100, 2000, 100))
    elo_bins_datasets = {
        f"elo_{elo}": games_dataset.filter(
            lambda x: elo <= x["white_elo"] < elo + 100
            and elo <= x["black_elo"] < elo + 100,
            num_proc=num_proc,
        )
        for elo in elo_bins
    }

    elo_dataset_dict = DatasetDict(elo_bins_datasets)

    print(elo_dataset_dict)

    elo_datasets = {
        name: sample_train_valid_test(dataset, train_size, valid_size, test_size)
        for name, dataset in elo_dataset_dict.items()
    }
    elo_datasets = DatasetDict(elo_datasets)

    elo_datasets.save_to_disk(save_dir, num_proc=num_proc)


if __name__ == "__main__":
    main()
