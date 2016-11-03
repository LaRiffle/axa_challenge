import pandas as pd
<<<<<<< HEAD
#from fonction_py.preprocess import *
=======
from fonction_py.preprocess import *
from fonction_py.split import *
>>>>>>> origin/master
import numpy as np
import matplotlib.pyplot as plt

import time


start_time = time.time()
print("go")

fields = ['DATE', 'DAY_OFF', 'DAY_DS', 'WEEK_END', 'DAY_WE_DS','TPER_TEAM', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire



data=pd.read_csv("data/train_2011_2012_2013.csv", sep=";", usecols=fields, nrows=100) # LECTURE

x = data[fields[0:-2]] # Data sans les received calls
y = data[fields[-1]] # label = received calls
ass = data[fields[-2]] # ass assignment = differentes categories a predire

#Test sur Crises
<<<<<<< HEAD
print(x[ass=='Crises'])

=======
x = x[ass=='Crises']
y = y[ass=='Crises']
>>>>>>> origin/master


x = preprocess(x) # rajoute les features
xTrain, xTest, yTrain, yTest = faireSplitting(x) # rajoute les features
#A_DEFINIR = train(xTrain, yTrain) # rajoute les features
#yEmpirique = test(A_DEFINIR, xTest) # rajoute les features
#check(yEmpirique, yTest)



#x = pd.concat([x,x['DATE']], axis=1) # test
#print(x)

print("--- %s seconds ---" % (time.time() - start_time))



