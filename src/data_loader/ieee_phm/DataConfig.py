from attr import dataclass

@dataclass
class DataConfig:
    """Configuration for dataset construction."""
    seq_len: int = 30           # Sequence length (window size)
    stride: int = 1             # Sliding window stride
    train_ratio: float = 0.7    # Train split ratio (from training bearings)
    val_ratio: float = 0.15     # Validation split ratio
    test_ratio: float = 0.15    # Test split ratio
    batch_size: int = 64
    scaler_type: str = 'minmax' # 'minmax' or 'standard'
    target_col: str = 'RUL'     # Target column
    random_seed: int = 42
    num_workers: int = 2

# config = DataConfig()
# print("Data Configuration:")
# for field_name, field_val in vars(config).items():
#     print(f"  {field_name}: {field_val}")