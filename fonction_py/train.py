from numpy import *

def train(xTrain, yTrain):
    W = dot(inv(dot(xTrain.T,xTrain)),dot(xTrain.T,yTrain))
    return(W)