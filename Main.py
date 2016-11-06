import pandas as pd
from fonction_py.preprocess import *
from fonction_py.train import *
from fonction_py.tools import *
from fonction_py.robin import *
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

import time


start_time = time.time()
print("go")

fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS','TPER_TEAM', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
selectAss = 'Téléphonie' # quel type de ASS_ASSIGNMENT on travaille


data=pd.read_csv("data/train_2011_2012_2013.csv", sep=";", usecols=fields, nrows=100000000) # LECTURE

x = data[fields[0:-2]] # Data sans les received calls
y = data[fields[-1]] # label = received calls
ass = data[fields[-2]] # ass assignment = differentes categories a predire

x = x[ass==selectAss]
y = y[ass==selectAss]

print("preprocessing...")
x = preprocess(x) # rajoute les features

######################################################################TEST DE ROBIN
robin(x, y)
print("--- %s seconds ---" % (time.time() - start_time))

######################################################################


print("splitting....")
xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8) # rajoute les features
print("train....")
W = train(xTrain, yTrain) # On créé un produit de l'apprentissage
print("Check....")
#yEmpirique = test(A_DEFINIR, xTest) # rajoute les features
yEmpirique = yTest
#check(yEmpirique, yTest)


print("--- %s seconds ---" % (time.time() - start_time))



