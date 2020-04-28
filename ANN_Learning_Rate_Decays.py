# Functions for a variety of learning rate decay methods
import numpy as np

class Constant(object):

    decay_rate = 0

    def __init__(self, decay_rate):
        self.decay_rate = decay_rate

    def decay(self, learning_rate, iteration):
        return learning_rate


class Exponential(object):

    decay_rate = 0

    def __init__(self, decay_rate):
        self.decay_rate = decay_rate

    def decay(self, learning_rate, iteration):
        return np.power(self.decay_rate, iteration) * learning_rate

