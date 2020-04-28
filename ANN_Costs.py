# Script for cost functions for ANN_Class
import numpy as np


class MSE(object):

    @staticmethod
    def func(prediction, actual):
        return np.sum((actual - prediction) ** 2) / (actual.shape[1])

    @staticmethod
    def grad(prediction, actual):
        return -2 * (actual - prediction)/(actual.shape[1])
