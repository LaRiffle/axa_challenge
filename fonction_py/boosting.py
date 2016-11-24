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

from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor(loss='huber', alpha=0.9,
                                n_estimators=100, max_depth=3,
                                learning_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)

fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
data=pd.read_csv("data/trainPure.csv", sep=";", usecols=fields) # LECTURE
resultat = pd.read_csv("data/submission.txt", sep="\t") # LECTURE
xTrain,yTrain = preprocess(data.copy(), 'Téléphonie') # rajoute les features
xTest,xTrain,souvenir=preprocessFINAL(xTrain,'Téléphonie')
        
clf.fit(xTrain,yTrain)

yPred=clf.predict(xTest)

resultat=pd.merge(resultat, souvenir, how='left',on=['DATE', 'ASS_ASSIGNMENT'])

resultat=resultat.fillna(0)
resultat['prediction'] = resultat['prediction_x']+resultat['prediction_y']
del resultat['prediction_x']
del resultat['prediction_y']
pd.DataFrame(res).to_csv("reslist.csv", sep=";", decimal=",")
resultat.to_csv("boosting.txt", sep="\t", index =False)    