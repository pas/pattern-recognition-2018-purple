import csv
import numpy as np


def load_data(path, normalize=True):
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
    return data, labels_to_one_hot(labels)


def labels_to_one_hot(labels):
    one_hot_labels = []
    for label in labels:
        one_hot_vector = np.zeros(10)
        one_hot_vector[int(label)] = 1
        one_hot_labels.append(one_hot_vector)
    return np.array(one_hot_labels)