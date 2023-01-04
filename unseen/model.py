from typing import List

import numpy as np
import torch
import torch.nn as nn
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.deepfm import FactorizationMachine, DeepFM
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.models.deepfm import DenseArch, SparseArch, OverArch


class FMInteractionArch(nn.Module):
    def __init__(
        self,
        fm_in_features: int,
        sparse_feature_names: List[str],
        deep_fm_dimension: int,
    ) -> None:
        super().__init__()
        self.sparse_feature_names: List[str] = sparse_feature_names
        self.deep_fm = DeepFM(
            dense_module=nn.Sequential(
                nn.Linear(fm_in_features, deep_fm_dimension),
                nn.ReLU(),
            )
        )
        self.fm = FactorizationMachine()

    def forward(
        self,
        sparse_features: KeyedTensor,
    ) -> torch.Tensor:
        tensor_list = []
        for feature_name in self.sparse_feature_names:
            tensor_list.append(sparse_features[feature_name])

        deep_interaction = self.deep_fm(tensor_list)
        fm_interaction = self.fm(tensor_list)

        return torch.cat([deep_interaction, fm_interaction], dim=1)


class HahowDeepFM(nn.Module):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        deep_fm_dimension: int,
    ) -> None:
        super().__init__()
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[0].embedding_dim

        # process sparse features
        self.sparse_arch = SparseArch(
            embedding_bag_collection
        )

        # fm & dnn
        feature_names = []
        fm_in_features = 0
        for conf in embedding_bag_collection.embedding_bag_configs():
            for feat in conf.feature_names:
                feature_names.append(feat)
                fm_in_features += conf.embedding_dim

        self.inter_arch = FMInteractionArch(
            fm_in_features=fm_in_features,
            sparse_feature_names=feature_names,
            deep_fm_dimension=deep_fm_dimension,
        )

        # activation
        fm_dimension = 1
        over_in_features = deep_fm_dimension + fm_dimension

        self.out_courseid = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(over_in_features, 728),
            nn.Sigmoid(),
        )
        self.out_subgroup = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(over_in_features, 92),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
        dense_features: torch.Tensor = None
    ) -> torch.Tensor:
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(
            sparse_features=embedded_sparse
        )
       
        logits_courseid = self.out_courseid(concatenated_dense)
        logits_subgroup = self.out_subgroup(concatenated_dense)
        return logits_courseid, logits_subgroup
