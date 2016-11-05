from fonction_py.tools import *
from sklearn import linear_model
import matplotlib.pyplot as plt


def robin(x, y):
    xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8)  # rajoute les features
    print("ok")
    lin = linear_model.LinearRegression()
    lin.fit(xTrain, yTrain)
    lin.score(xTrain, yTrain)
    yEmp = lin.predict(xTest)
    xp = zeros(len(yTest))
    for i in range(len(xp)):
        xp[i]=i
    print(sum(y!=0))
    print(sum(y!=0)/y.shape[0])
    plt.hist(y[y!=0])
    plt.show()