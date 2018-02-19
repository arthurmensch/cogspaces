import numpy as np


def accuracy(y_pred, y_true):
    accuracies = {}
    for study in y_true:
        accuracies[study] = np.mean(y_pred[study] == y_true[study])
    return accuracies