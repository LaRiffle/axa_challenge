from fonction_py.tools import *
from fonction_py.preprocess import *
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
import time

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
    
def submit():
    
    start_time = time.time()
    
    fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
    data=pd.read_csv("data/trainPure.csv", sep=";", usecols=fields) # LECTURE
    resultat = pd.read_csv("data/submission.txt", sep="\t") # LECTURE
    categoryList = ['CAT','CMS','Crises','Domicile','Gestion','Gestion - Accueil Telephonique','Gestion Assurances','Gestion Clients','Gestion DZ','Gestion Relation Clienteles','Gestion Renault','Japon','Manager','Mécanicien','Médical','Nuit','Prestataires','RENAULT','RTC','Regulation Medicale','SAP','Services','Tech. Axa','Tech. Inter','Tech. Total','Téléphonie']
    for category in categoryList :
        start_time = time.time()
        print(category)
        xTrain,yTrain = preprocess(data.copy(), category) # rajoute les features
        xTest,xTrain,souvenir=preprocessFINAL(xTrain,category)
        prediction = telephoniePred(xTrain,yTrain,xTest)    
        prediction =np.round(prediction).astype(int)
        souvenir['prediction']=prediction
        end_time = time.time()
        print('prediction\'s length : ',len(prediction))
        print('Time : ',end_time - start_time)
        resultat=pd.merge(resultat, souvenir, how='left',on=['DATE', 'ASS_ASSIGNMENT'])
    print('DONE')
    resultat=resultat.fillna(0)
    resultat['prediction'] = resultat['prediction_x']+resultat['prediction_y']
    del resultat['prediction_x']
    del resultat['prediction_y']
    pd.DataFrame(res).to_csv("reslist.csv", sep=";", decimal=",")
    resultat.to_csv("vraipred.txt", sep="\t", index =False)    
    return resultat

    