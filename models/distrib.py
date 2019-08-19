import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from math import sin, cos




def sample(x, y, label, n_labels):
        shift = 1.4
        if label >=n_labels:
            label =  np.random.randint(0, n_labels)
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        
        return np.array([new_x, new_y]).reshape((2,))

def normal_mixture(label, batch_size = None, n_dim=2, n_labels = 10, x_var=0.5, y_var=0.2):
    # borrow from:
    # https://github.com/nicklhy/AdversarialAutoEncoder/blob/master/data_factory.py#L40
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")
    x = np.random.normal(0, x_var, size = (batch_size, n_dim // 2))
    y = np.random.normal(0, y_var, size = (batch_size, n_dim // 2))
    z = np.empty((batch_size, n_dim), dtype=np.float32)

    for num, batch in enumerate(np.nonzero(label)[1]):
        for zi in range(n_dim // 2):
            #if label_indices is not None:
                z[num, zi*2:zi*2+2] = sample(x[num, zi], y[num, zi], batch, n_labels)
            #else:
    if np.any(np.isnan(z)):
            raise Exception("Fuck!")         #   z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)
    return z

def normal(batch_size, n_dim = 2):
    z = np.random.normal(0, 0.5, size = (batch_size, n_dim)).astype(np.float32)
    return z