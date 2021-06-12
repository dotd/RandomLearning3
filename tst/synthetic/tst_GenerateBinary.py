import numpy as np
import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.synthetic.GenerateBinary import GenerateBinary
from src.learning.LearnerPM import PerceptronVanilla, Peceptron2
from definitions_random_learning3 import ROOT_DIR


def tst_generate_binary():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_writer = SummaryWriter(log_dir=ROOT_DIR + "/tensorboard/runs/tst_BitFlippingEnv_DQN_" + run_id)

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
        tb_writer.add_scalar("error", error, i)
    print(f"final error={error}")


if __name__ == "__main__":
    tst_generate_binary()
