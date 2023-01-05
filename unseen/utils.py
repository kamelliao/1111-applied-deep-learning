from pathlib import Path
import pandas as pd


def load_datasets(data_path: str):
    dataset_dir = Path(data_path)
    datasets = {data_path.stem: pd.read_csv(data_path) for data_path in dataset_dir.iterdir()}

    return datasets