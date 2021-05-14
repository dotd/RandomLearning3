import numpy as np
from src.synthetic.GenerateBinary import GenerateBinary


def tst_GenerateBinary():
    g = GenerateBinary(2, 7, 10, np.random.RandomState(0), p_noise=0.1)
    print(g.centers)
    print(g.labels)
    print(g.labels1hot)
    print(g.noise)
    print(g.data)


if __name__ == "__main__":
    tst_GenerateBinary()
