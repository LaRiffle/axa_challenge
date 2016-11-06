from fonction_py.tools import *
from sklearn import linear_model
import matplotlib.pyplot as plt


def robin(x, y):
    xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8)  # rajoute les features
    print("ok")
    model = linear_model.LinearRegression()
    model.fit(xTrain, yTrain)
    model.score(xTrain, yTrain)

    pred =  model.predict(xTest)
    print("DecisionTreeRegressor")
    check(pred, yTest)