import numpy as np

def faireSplitting(x, y):
    ln = (np.random.rand(x.shape[0]) < 0.8)
    return 0, 0, 0, 0;
