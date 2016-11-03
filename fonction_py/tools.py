import numpy as np
import math

def faireSplitting(x, y, taille): # return xTrain, xTest, yTrain, yTest
    ln = (np.random.rand(x.shape[0]) < taille)
    return x[ln], x[~ln], y[ln], y[~ln];


def check(yEmpirique, yTest):

    alpha=-0.1

    if(yTest.shape[0] != yEmpirique.shape[0]):
        print("Erreur sur la taille de la prédiction")
        return 0
    print("accuracy :")
    print(sum(yEmpirique==yTest)*100/yEmpirique.shape[0]) # pourcentage de bonne prediction

    linex = 0
    diff = (yTest-yEmpirique).values

    for i in range(len(diff)):
        linex = linex + math.exp(alpha * diff[i]) - alpha*diff[i]-1

    print("linEx :")
    print(linex)
