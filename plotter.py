import numpy as np
from matplotlib import pyplot as pl


def plot_temporal(x, x_n_steps, x_step, y_n_steps, y_step, sample_nr=12):
    pl.ion()
    sample_i = sample_nr * y_n_steps
    sample = np.array(x[sample_i:sample_i + y_n_steps])
    sample = sample.reshape(y_n_steps * y_step, x_n_steps * x_step)
    pl.pcolormesh(sample, vmin=0, vmax=1, cmap=pl.cm.binary)

    pl.gca().set_yticks(np.arange(0, y_step * y_n_steps, y_step))
    pl.gca().set_xticks(np.arange(0, x_step * x_n_steps, x_step))
    pl.tight_layout()
    pl.ylim([0, y_step*y_n_steps])
    pl.xlim([0, x_step*x_n_steps])
    pl.gca().invert_yaxis()

    pl.title("Timesteps for sample %d" % sample_nr)
    pl.show(block=True)

