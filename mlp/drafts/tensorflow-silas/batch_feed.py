import numpy as np


class RandomBatchFeeder():

    def __init__(self, data, labels, random_seed):
        """
        Container for data and labels that provides random batch creation on-demand.

        :param data:        np.ndarray containing the data to be stored in this container,
                            with data points (e.g. feature vectors) as rows
        :param labels:      the labels corresponding to the data, each row i should be the label for data row i.
                            Number of labels must match number of data points.
        :param random_seed: any integer, will always yield the same random batches for a given random seed
        """
        assert data.shape[0] == labels.shape[0]
        self._data = data
        self._labels = labels
        self._rand_gen = np.random.RandomState()
        self._rand_gen.seed(random_seed)
        self._indices = np.arange(data.shape[0], dtype=np.uint32)

    def next_batch(self, batch_size):
        """
        Creates a random batch of data, of size batch_size, and the corresponding labels.
        Note: random selection happens in term of the previously defined random_seed

        :param batch_size:  desired size of the random batch, must not be larger than number of data points in
                            this container (i.e. batch_size <= data.shape[0])
        :return:    np.ndarray with dimensions
                    [batch_size, data.shape[1]] containing the randomly selected data, and np.ndarray with dimensions
                    [batch_size, labels.shape[1]] containing the labels corresponding to the selected data.
        """
        assert batch_size <= self._data.shape[0]
        rand_indices = self._rand_gen.choice(self._indices, batch_size)
        return self._data[rand_indices, :], self._labels[rand_indices, :]