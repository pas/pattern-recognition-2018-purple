import csv
import numpy as np


def load_data(path, normalize=True):
    """
    Loads data from specified path, returns matrix of data points (in row-orientation) and matrix of the corresponding
    10-class one-hot encoded labels. Data is normalized by default (assuming original scale 0-255).

    :param path:        data path (e.g. 'data/train.csv')
    :param normalize:   normalize data (bool)
    :return:    np.ndarray with dimensions [num_data_rows, len_feature_vector] containing (normalized) data,
                and np.ndarray with dimensions [num_data_rows, 10] containing one-hot encoded labels corresponding
                to data.
    """
    print("loading data: {}".format(path))
    with open(path, "r") as file:
        data = []
        labels = []
        csvfile = csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for line in csvfile:
            labels.append(line[0])
            data.append(line[1:])
    data = np.array(data, dtype=np.float32)
    if normalize:
        data = data / 255
    return data, _labels_to_one_hot(labels)

def load_train_data(path, normalize=True):
    """
    Loads data from specified path, returns matrix of data points (in row-orientation) and matrix of the corresponding
    10-class one-hot encoded labels. Data is normalized by default (assuming original scale 0-255).

    :param path:        data path (e.g. 'data/train.csv')
    :param normalize:   normalize data (bool)
    :return:    np.ndarray with dimensions [num_data_rows, len_feature_vector] containing (normalized) data,
                and np.ndarray with dimensions [num_data_rows, 10] containing one-hot encoded labels corresponding
                to data.
    """
    with open(path, "r") as file:
        data = []
        labels = []
        csvfile = csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for line in csvfile:
            labels.append(line[0]) # TODO: add method for no label and mabye identifier here.
            data.append(line[1:])
    data = np.array(data, dtype=np.float32)
    if normalize:
        data = data / 255
    return data, _labels_to_one_hot(labels)

def load_test_data(path, normalize=True):
    with open(path, "r") as file:
        data = []
        csvfile = csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for line in csvfile:
            data.append(line[0:])
    data = np.array(data, dtype=np.float32)
    if normalize:
        data = data / 255
    return data

def _labels_to_one_hot(labels):
    one_hot_labels = []
    for label in labels:
        one_hot_vector = np.zeros(10)
        one_hot_vector[int(label)] = 1
        one_hot_labels.append(one_hot_vector)
    return np.array(one_hot_labels)
