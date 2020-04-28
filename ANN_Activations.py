# Script for activation functions for ANN_Class
import numpy as np


class Leaky_Relu(object):

    @staticmethod
    def func(x):
        return (x > 0) * x + (x < 0) * 0.01*x

    @staticmethod
    def grad(x):
        return (x > 0) + (x < 0) * 0.01


class Linear(object):

    @staticmethod
    def func(x):
        return x

    @staticmethod
    def grad(x):
        return 1
