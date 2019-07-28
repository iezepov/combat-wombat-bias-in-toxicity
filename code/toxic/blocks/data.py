import numpy as np

import torch
from torch.utils import data
from keras.preprocessing.sequence import pad_sequences


class BasicSampler(data.Sampler):
    """
        Just itarte over a list of ids.
    """

    def __init__(self, ids):
        self.ids = ids

    def __iter__(self):
        return self.ids.__iter__()

    def __len__(self):
        return self.ids.__len__()


class TextDataset(data.Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return [self.X[idx], self.y[idx]]
        return self.X[idx]


class PercentilePaddingCollator:
    """
        Pad sequences on the fly!
    """

    def __init__(self, test=False, percentile=100):
        self.test = test
        self.percentile = percentile

    def __call__(self, batch):
        if not self.test:
            sequences = [item[0] for item in batch]
            target = [item[1] for item in batch]
        else:
            sequences = batch
        lens = [len(x) for x in sequences]
        max_len = np.percentile(lens, self.percentile)
        sequences = pad_sequences(sequences, maxlen=int(max_len))
        sequences = torch.tensor(sequences).long()
        if not self.test:
            target = torch.tensor(target).float()
            return [sequences, target]
        return [sequences]


class SequenceBucketCollator:
    def __init__(
        self, choose_length, sequence_index, length_index, maxlen, label_index=None
    ):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.maxlen = maxlen
        self.label_index = label_index

    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]

        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]

        length = self.choose_length(lengths)
        mask = torch.arange(start=self.maxlen, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]

        batch[self.sequence_index] = padded_sequences

        if self.label_index is not None:
            return (
                [x for i, x in enumerate(batch) if i != self.label_index],
                batch[self.label_index],
            )

        return batch
