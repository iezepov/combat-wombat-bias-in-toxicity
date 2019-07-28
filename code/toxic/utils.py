import os
import random
import hashlib
from typing import Dict
import numpy as np
import torch

from .metrics import IDENTITY_COLUMNS


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def hash_fold(salted: str, n_folds: int) -> int:
    return int(hashlib.md5(salted.encode()).hexdigest(), 16) % n_folds


def perfect_bias(p):
    """
        The best constant init for a bias layer
    """
    return np.log(p / (1 - p))


def clip_to_max_len(batch):
    X, y, lengths = map(torch.stack, zip(*batch))
    max_len = torch.max(lengths).item()
    return X[:, :max_len], y


def should_decay(name):
    return not any(n in name for n in ("bias", "LayerNorm.bias", "LayerNorm.weight"))


def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col_name in ["target"] + IDENTITY_COLUMNS:
        bool_df[col_name] = df[col_name].fillna(0) >= 0.5
    return bool_df


def build_matrix_fasttext(word2index, model):
    embedding_matrix = np.zeros(
        (max(word2index.values()) + 1, model.get_dimension()), dtype=np.float32
    )
    for word, index in word2index.items():
        embedding_matrix[index] = model.get_word_vector(word)
    return embedding_matrix


class RunninMetrics:
    """
    A handy addition for the tqdm progress bar. Use as follows:

    ```
    running_metrics.update({
        'main_loss': main_loss.item(),
        'type_loss': type_loss.item(),
    })

    bar.set_postfix(running_metrics.store)
    ```
    """

    def __init__(self, decay=0.9):
        self.store = {}
        self.decay = decay

    def update(self, metrics):
        for k, v in metrics.items():
            if k in self.store:
                self.store[k] = self.store[k] * self.decay + v * (1 - self.decay)
            else:
                self.store[k] = v


class TensorboardAggregator:
    """
        Log average value periodically instead of logging on every batch
    """

    def __init__(self, writer, every=5):
        self.writer = writer
        self.every = every

        self.step = 0
        self.scalars = None

    def log(self, scalars: Dict[str, float]):
        self.step += 1

        if self.scalars is None:
            self.scalars = scalars.copy()
        else:
            for k, v in scalars.items():
                self.scalars[k] += v

        if self.step % self.every == 0:
            for k, v in self.scalars.items():
                self.writer.add_scalar(k, v / self.every, self.step)
            self.scalars = None

