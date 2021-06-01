import numpy as np
from src.synthetic.GenerateBinary import GenerateBinary
from src.learning.LearnerPM import PerceptronVanilla, Peceptron2
import matplotlib.pyplot as plt


def tst_GenerateBinary():
    num_centers = 2
    dim = 50
    num_samples = 10000
    random = np.random.RandomState(1)
    p_noise = 0.5
    eta = 1
    episodes = 10

    g = GenerateBinary(num_centers, dim, num_samples, random, p_noise)
    c = Peceptron2(dim, random, eta=eta)
    print(g.centers)
    print(g.labels)
    print(g.labels1hot)
    print(g.noise)
    print(g.data)
    error_vec = list()
    error_initial = c.compute_error(x=g.data, d=g.labels_pm)
    print(f"error_initial={error_initial}")
    error_vec.append(error_initial)
    for i in range(episodes):
        error = c.train(x=g.data, d=g.labels_pm)
        error_vec.append(error)
    print(f"final error={error}")
    plt.figure()
    plt.plot(error_vec)
    plt.show(block=True)


if __name__ == "__main__":
    tst_GenerateBinary()
