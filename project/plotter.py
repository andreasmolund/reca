from matplotlib import pyplot as pl
import numpy as np


def plot_temporal(x, x_n_steps, x_step, y_n_steps, y_step):
    pl.ion()
    sample_i = 22
    sample = np.array(x[sample_i:sample_i + y_n_steps])
    sample = sample.reshape(y_n_steps * y_step, x_n_steps * x_step)
    pl.pcolormesh(sample, vmin=0, vmax=1, cmap=pl.cm.binary)
    pl.gca().set_yticks(np.arange(0, y_step * y_n_steps, y_step))
    pl.gca().set_xticks(np.arange(0, x_step * x_n_steps, x_step))
    pl.gca().invert_yaxis()
    pl.title("Timesteps for one sample")
    pl.show(block=True)

