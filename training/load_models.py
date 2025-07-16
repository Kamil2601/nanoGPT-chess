

from training.lightning_training import LightningGPT

from . import model_configs


# [BOS] white_move black_move
# Example: "[BOS] Pe2e4 Pd7d5 Pe4d5"
def no_elo_model(checkpoint_path: str):
    no_elo_model = LightningGPT.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=model_configs.no_elo_config,
        test_start_token=0,
        test_token_step=1,
        training_ignore_first_n_targets=0,
        training_target_step=1,
        tokenizer=model_configs.no_elo_tokenizer,
        masked_elo_test=False,
    )

    return no_elo_model

# white_elo black_elo white_move black_move
# Example: "1800 [UNK_ELO] Pe2e4 Pd7d5 Pe4d5"
def base_elo_model(checkpoint_path: str):
    base_model = LightningGPT.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=model_configs.base_model_config,
        test_start_token=10,
        test_token_step=1,
        training_ignore_first_n_targets=1,
        training_target_step=1,
        tokenizer=model_configs.base_tokenizer,
        masked_elo_test=True,
    )

    return base_model


# white_elo black_elo white_move black_move (without elo mask)
# Example: "1800 1700 Pe2e4 Pd7d5 Pe4d5"
def base_elo_no_mask_model(checkpoint_path: str):
    base_no_mask_model = LightningGPT.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=model_configs.base_model_config,
        test_start_token=10,
        test_token_step=1,
        training_ignore_first_n_targets=1,
        training_target_step=1,
        tokenizer=model_configs.base_tokenizer,
        masked_elo_test=False,
    )

    return base_no_mask_model

# white_elo black_elo white_move black_material black_move white_material
# Example: "1800 [UNK_ELO] Pe2e4 39 Pd7d5 39 Pe4d5 38"
def material_model(checkpoint_path: str):
    material_model = LightningGPT.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=model_configs.material_model_config,
        test_start_token=1,
        test_token_step=2,
        training_ignore_first_n_targets=1,
        training_target_step=2,
        tokenizer=model_configs.material_tokenizer,
        masked_elo_test=True,
    )

    return material_model

# Model with elo tokens and material pair "white_material|black_material" token after each move
# Example: "1800 [UNK_ELO] Pe2e4 39|39 Pd7d5 39|39 Pe4d5 39|38"
def material_pair_model(checkpoint_path: str):
    material_pair_model = LightningGPT.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=model_configs.material_pair_model_config,
        test_start_token=1,
        test_token_step=2,
        training_ignore_first_n_targets=1,
        training_target_step=2,
        tokenizer=model_configs.material_pair_tokenizer,
        masked_elo_test=True,
    )

    return material_pair_model

# white_elo black_elo white_move black_piece_count black_move white_piece_count
# Example: "1800 [UNK_ELO] Pe2e4 Q1R2B2N2P8 Pd7d5 Q1R2B2N2P8 Pe4d5 Q1R2B2N2P7"
def piece_count_model(checkpoint_path: str):
    piece_count_model = LightningGPT.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=model_configs.piece_count_model_config,
        test_start_token=1,
        test_token_step=2,
        training_ignore_first_n_targets=1,
        training_target_step=2,
        tokenizer=model_configs.piece_count_tokenizer,
        masked_elo_test=True,
    )

    return piece_count_model
