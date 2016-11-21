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
    pca = decomposition.PCA(n_components=65)
    pca.fit(xTrain)
    PCAxTrain = pca.transform(xTrain)
    nbLines,nbFeatures = PCAxTrain.shape
    res = minimize(fun_to_min,np.zeros(nbFeatures+1),args=(PCAxTrain,yTrain))
    PCAxTest = pca.transform(xTest)
    x = res.x #x est de longeur nbFeatures+1
    a=x[:-1] #a.T est une colonne de longueur nbFeatures
    b=x[-1] #b est un scalaire
    print("a : \n",a)
    print("b : \n",b)
    pred = np.dot(PCAxTest,np.transpose(a))+b
    pred = np.round(pred)
    check(pred, yTest)
    bins = np.linspace(-10, 10, 40)
    plt.hist(pred-yTest, bins, normed=1)
    
def telephoniePred(x,y,xTest):
    pca = decomposition.PCA(n_components=65)#65)
    pca.fit(x)
    PCAxTrain = pca.transform(x)
    nbLines,nbFeatures = PCAxTrain.shape
    res = minimize(fun_to_min,np.zeros(nbFeatures+1),args=(PCAxTrain,y))
    PCAxTest = pca.transform(xTest)
    x = res.x
    a=x[:-1]
    b=x[-1]
    pred = np.dot(PCAxTest,np.transpose(a))+b
    pred = np.round(pred)
    return pred
    
    
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

#a : 
# [ -4.51187364e+00   3.38242455e+00  -1.21221693e+01   2.09881551e-02
#  -4.41768572e-01   2.14374943e+00  -2.44087620e-01  -7.36365590e-01
#   6.20131421e+00   3.97019347e+00   2.61540336e+00   5.07557175e+00
#  -2.36070480e+00   7.83770186e-01  -2.85907549e+00   5.98147782e-01
#  -3.71732393e+00   1.13077517e+00   4.20373369e+00   4.36745759e-01
#  -2.68431728e+00  -4.43218981e-01   5.81247338e-01   6.17176255e-01
#  -9.80813609e-01  -1.09948028e+00  -1.15989481e+00  -4.12295756e-01
#  -2.66160200e-01   8.91540111e-02  -2.95623735e-01  -1.33402628e+00
#   9.39698099e-01  -1.45401109e-01   5.44124934e-02   7.90849079e-01
#   2.25779615e+00   4.16458818e-01   7.55368925e-01  -1.39918346e-01
#  -2.60203698e-01  -2.70301801e-01   6.98076852e-01   4.32647696e-01
#   5.83255563e-01  -3.72538379e-01  -1.04008611e+00   8.46517742e-01
#   2.66915968e-01   6.04541556e-01   1.01594631e+01   6.73908394e+00
#   7.47238332e+00   1.28802616e+01  -8.33981596e+00  -8.77078860e+00
#   5.06659923e-01   5.82519167e+00  -2.19151804e+01  -4.65302708e+00
#  -1.10262775e+01   1.10886327e+01   3.36040552e+01  -2.00647693e+01
#   1.15964000e+01]
#b : 
# 13.4367650654