import numpy as np


class RandomBatchFeeder():

    def __init__(self, data, labels, random_seed):
        assert data.shape[0] == labels.shape[0]
        self._data = data
        self._labels = labels
        self._rand_gen = np.random.RandomState()
        self._rand_gen.seed(random_seed)
        self._indices = np.arange(data.shape[0], dtype=np.uint32)

    def next_batch(self, batch_size):
        rand_indices = self._rand_gen.choice(self._indices, batch_size)
        return self._data[rand_indices, :], self._labels[rand_indices, :]