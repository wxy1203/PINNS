import scipy.io
import numpy as np


def load_data(path, keys=None):
    if keys is None:
        # default: 2D
        keys = ['x', 'tt', 'uu']
    data = scipy.io.loadmat(path)
    xi = [np.real(data[k]).reshape((-1, 1)) for k in keys[:-1]]
    raw = np.real(data[keys[-1]]).T
    u = np.real(data[keys[-1]]).T.reshape((-1, 1))
    x = np.concatenate([xx.reshape((-1, 1)) for xx in np.meshgrid(*xi)], axis=1)
    data = np.concatenate([x, u], axis=1)
    domain = [[np.min(x[:, i]), np.max(x[:, i])] for i in range(x.shape[-1])]
    return data, domain, raw
