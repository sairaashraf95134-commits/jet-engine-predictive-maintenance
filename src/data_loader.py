from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw C-MAPSS directory
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "C-MAPSS"

# C-MAPSS FD001 column names
COLUMN_NAMES = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)]
]


def load_train_data(dataset="FD001"):
    """
    Load the C-MAPSS training dataset.
    """
    file_path = DATA_DIR / f"train_{dataset}.txt"

    if not file_path.exists():
        raise FileNotFoundError(
        f"Dataset file not found: {file_path}"
    )

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )


def load_test_data(dataset="FD001"):
    """
    Load the C-MAPSS test dataset.
    """
    file_path = DATA_DIR / f"test_{dataset}.txt"

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )


def load_rul_data(dataset="FD001"):
    """
    Load the true RUL values for the test engines.
    """
    file_path = DATA_DIR / f"RUL_{dataset}.txt"

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )


def load_dataset(dataset="FD001"):
    """
    Load the complete C-MAPSS dataset.

    Returns:
        train_df
        test_df
        rul_df
    """
    train_df = load_train_data(dataset)
    test_df = load_test_data(dataset)
    rul_df = load_rul_data(dataset)

    return train_df, test_df, rul_df