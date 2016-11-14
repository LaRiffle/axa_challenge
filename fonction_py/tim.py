from fonction_py.tools import *
from scipy.optimize import minimize
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import decomposition

# predire que Gestion renault = 0 depuis fevrier/2011
#
#

def fun_to_min(x,xTrain,yTrain):
    a=x[:-1]
    b=x[-1]
    return LinExp(np.dot(xTrain,np.transpose(a))+b,yTrain)


def linearLinexpMinimization(x, y):
    xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8)  # rajoute les features
    del x
    del y    
    print("ok")

    print("AVEC")
    pca = decomposition.PCA(n_components=65)#65)
    pca.fit(xTrain)
    PCAxTrain = pca.transform(xTrain)
    nbLines,nbFeatures = PCAxTrain.shape
    res = minimize(fun_to_min,np.zeros(nbFeatures+1),args=(PCAxTrain,yTrain.values))
    PCAxTest = pca.transform(xTest)
    x = res.x
    a=x[:-1]
    b=x[-1]
    print("a : \n",a)
    print("b : \n",b)
    pred = np.dot(PCAxTest,np.transpose(a))+b
    pred = np.round(pred)
    check(pred, yTest)
    bins = np.linspace(-10, 10, 40)
    plt.hist(pred-yTest, bins, normed=1)
    
    
    
#    print("PCA")
#    pos = [1,3,5,10,20,30,40,50, 60,62, 65, 70, 75,78, 80, 90]
#    resAcc = []
#    resLin = []
#    for i in pos:
#        pca = decomposition.PCA(n_components=i)
#        pca.fit(xTrain)
#        PCAxTrain = pca.transform(xTrain)
#        model = linear_model.LinearRegression()
#        model.fit(PCAxTrain, yTrain)
#        model.score(PCAxTrain, yTrain)
#        pred = model.predict(pca.transform(xTest))
#        pred =np.floor(np.round(pred))
#        resAcc.append(Accuracy(pred, yTest.values))
#        resLin.append(LinExp(pred, yTest.values))
#    print(resAcc)    
#    print(resLin)    
#    plt.plot(pos, resLin)
#    plt.show()
#    plt.plot(pos, resAcc)
#    plt.show()    
#######################################################################   
    #best accuracy en % :
#52.5071805097
#linEx :
#23987.9825805
## del x['TPER_TEAM']
#    x['YEAR'] = x['DATE'].str[0:4]
#    x['MONTH'] = x['DATE'].str[5:7]
#   # x['DAY'] = x['DATE'].str[8:10]
#    #x['HOUR'] = x['DATE'].str[-12:-10].astype(int)
#    x['HOUR'] = x['HOUR']+ ':'+((x['DATE'].str[-9:-8].astype(int)==3)*0.5).astype(str)
#    x['HOUR'] = x['DATE'].str[-12:-8]
#    del x['DATE']
#
#    x=pd.get_dummies(x)