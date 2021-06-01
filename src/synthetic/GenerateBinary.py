import numpy as np


class GenerateBinary:

    def __init__(self,
                 num_centers,
                 dim,
                 num_samples,
                 random,
                 p_noise,
                 alphabet=(-1, 1)):
        self.num_centers = num_centers
        self.dim = dim
        self.num_samples = num_samples
        self.random = random
        self.p_noise = p_noise
        self.alphabet = alphabet
        self.data = None
        self.label = None
        self.noise = None

        self.centers = self.random.choice(self.alphabet, size=(self.num_centers, self.dim))
        self.labels = self.random.choice(range(self.num_centers), size=(self.num_samples,))
        self.labels1hot = np.zeros(shape=(self.num_samples, self.num_centers))
        self.labels1hot[np.arange(self.num_samples), self.labels] = 1
        self.labels_pm = self.labels * 2 - 1
        self.data = np.dot(self.labels1hot, self.centers)
        self.noise = self.random.choice([-1, 1], size=self.data.shape, p=[p_noise, 1 - p_noise])
        self.data = self.data * self.noise