import numpy as np

def faireSplitting(x, y, taille): # return xTrain, xTest, yTrain, yTest
    ln = (np.random.rand(x.shape[0]) < taille)
    return x[ln], x[~ln], y[ln], y[~ln];
