import pandas as pd
from fonction_py.preprocess import *
from fonction_py.train import *
from fonction_py.tools import *
import numpy as np
import matplotlib.pyplot as plt

import time


start_time = time.time()
print("go")

fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS','TPER_TEAM', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire



data=pd.read_csv("data/train_2011_2012_2013.csv", sep=";", usecols=fields, nrows=1000) # LECTURE

x = data[fields[0:-2]] # Data sans les received calls
y = data[fields[-1]] # label = received calls
ass = data[fields[-2]] # ass assignment = differentes categories a predire

#Test sur Crises
x = x[ass=='Crises']
y = y[ass=='Crises']

x = preprocess(x) # rajoute les features
xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8) # rajoute les features

W = train(xTrain, yTrain) # On créé un produit de l'apprentissage
#yEmpirique = test(A_DEFINIR, xTest) # rajoute les features
yEmpirique = yTest
#check(yEmpirique, yTest)
print(x)
#x = pd.concat([x,x['DATE']], axis=1) # test

print("--- %s seconds ---" % (time.time() - start_time))



