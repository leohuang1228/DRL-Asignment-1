import numpy as np


def obs_reshape(obs):
    obs_new = np.array(obs, dtype=float)

    grid_size = obs[5]
    obs_new[0:10] = obs_new[0:10] / grid_size
    return tuple(obs_new)
