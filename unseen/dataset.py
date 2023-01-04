from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from vocab import USER_FEATS, COURSE_FEATS

class HahowDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()


def collate_fn_multiclass(data: List[Tuple[Dict, int]]) -> Dict[str, Union[KeyedJaggedTensor, torch.FloatTensor]]:
    batch = {k: [x[k] for x in data] for k in data[0].keys()}

    feature_names = USER_FEATS
    lengths = []
    values = []
    for feat in feature_names:
        if isinstance(batch[feat][0], list):
            lengths.extend([len(item) for item in batch[feat]])
            values.extend([item for sublist in batch[feat] for item in sublist])
        else:
            lengths.extend([1 for _ in batch[feat]])
            values.extend(batch[feat])

    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        keys=feature_names,
        values=torch.tensor(values),
        lengths=torch.tensor(lengths)
    )

    return {
        'sparse_features': sparse_features,
        'labels_courseid': torch.tensor(batch['label_course_id'], dtype=torch.float),
        'labels_subgroup': torch.tensor(batch['label_sub_groups'], dtype=torch.float),
    }

def collate_fn_multiclass_inf(data: List[Tuple[Dict, int]]) -> Dict[str, Union[KeyedJaggedTensor, torch.FloatTensor]]:
    batch = {k: [x[k] for x in data] for k in data[0].keys()}

    feature_names = USER_FEATS
    lengths = []
    values = []
    for feat in feature_names:
        if isinstance(batch[feat][0], list):
            lengths.extend([len(item) for item in batch[feat]])
            values.extend([item for sublist in batch[feat] for item in sublist])
        else:
            lengths.extend([1 for _ in batch[feat]])
            values.extend(batch[feat])

    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        keys=feature_names,
        values=torch.tensor(values),
        lengths=torch.tensor(lengths)
    )

    return {
        'sparse_features': sparse_features,
    }
