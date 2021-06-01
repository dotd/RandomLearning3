import numpy as np


class PerceptronVanilla:
    """
    self.w is a column vector so batch is [num_vectors * dim of w]
    """

    def __init__(self, dim, random, alphabet=(-1, 1), eta=0.1):
        self.dim = dim
        self.random = random
        self.alphabet = alphabet
        self.w = self.random.choice(self.alphabet, size=(self.dim, 1))
        self.eta = eta

    def classify(self, x):
        """
        vec is (num_samples x dim) so output is (num_samples x dim) x (dim x 1) = num_samples x 1
        :param vec:
        :return: classification
        """
        return np.sign(np.dot(x, self.w))

    def compute_error(self, x,d):
        error_vec = self.compute_error_vec(x, d)
        return np.sum(np.abs(np.sign(error_vec))) / (np.prod(error_vec.shape))

    def compute_error_vec(self, x, d):
        # d = 1, y = -1: 1 - (-1) = 2
        # d[-1, 1], y = [1, -1]: [-1,1] - [1, -1] = [-2, 2]
        y = self.classify(x)
        if len(d.shape) == 1:
            d = d.reshape([-1, 1])
        return d - y

    def train(self, x, d):
        '''
        :param x: batch of vectors in
        :param d:
        :return:
        '''
        error = self.compute_error_vec(x, d)
        # Now we compute (d - y) * x which is equal to x * error
        delta_mat = x * error
        delta = np.average(delta_mat, axis=0).reshape(-1, 1)
        self.w = self.w + self.eta * delta
        return np.sum(np.abs(np.sign(error))) / (self.dim * x.shape[0])


class Peceptron2(PerceptronVanilla):

    def train(self, x, d):
        '''
        :param x: batch of vectors in
        :param d:
        :return:
        '''
        error = self.compute_error_vec(x, d)
        # Now we compute (d - y) * x which is equal to x * error
        delta_mat = x * error
        delta = np.average(delta_mat, axis=0).reshape(-1, 1)
        self.w = np.sign(self.w + self.eta * np.sign(delta))
        return np.sum(np.abs(np.sign(error))) / (self.dim * x.shape[0])

