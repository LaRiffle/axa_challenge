import pandas as pd
from fonction_py.preprocess import *
from fonction_py.train import *
from fonction_py.tools import *
from fonction_py.robin import *
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

import time

#A FAIRE :)
#preprocess : Preprocessing : choisir les colonnes et créer les nouvelles,
#preprocessFINAL : Preprocessing de submission.txt pour qu'il soit exactement comme la sortie du preprocessing
#
#1 modele :
#    option : PCA -> réduire et choisir la dimension la meilleur,
#    prevision : regression lineaire (tout pourri)
#                gradient descent
#                tree
#2eme modele :
#    etude en fonction de ASS_ASSIGNMENT : voir lesquels servent a qqc l'impact de l'annee    
#    




start_time = time.time()
print("go")

fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS','TPER_TEAM', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
selectAss = 'Services' # quel type de ASS_ASSIGNMENT on travaille


x=pd.read_csv("data/train_2011_2012_2013.csv", sep=";", usecols=fields, nrows=10000) # LECTURE

y = x[fields[-1]] # label = received calls
ass = x[fields[-2]] # ass assignment = differentes categories a predire
x = x[fields[0:-2]] # Data sans les received calls

x = x[ass==selectAss]
y = y[ass==selectAss]

print("preprocessing...")
x = preprocess(x) # rajoute les features

######################################################################TEST DE ROBIN
robin(x, y)
print("--- %s seconds ---" % (time.time() - start_time))
x.columns.values



###########################################################

print("--- %s seconds ---" % (time.time() - start_time))



