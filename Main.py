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

fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
selectAss = 'Gestion Renault' # quel type de ASS_ASSIGNMENT on travaille
c = pd.DataFrame()
listass= ['CAT', 'CMS', 'Crises', 'Domicile', 'Evenements', 'Gestion', 'Gestion - Accueil Telephonique', 'Gestion Amex', 'Gestion Assurances', 'Gestion Clients', 'Gestion DZ', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Manager', 'Mécanicien', 'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC', 'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'Téléphonie']
tot = pd.DataFrame()
for selectAss in listass:
    x=pd.read_csv("data/trainPure.csv", sep=";", usecols=fields) # LECTURE
    print("preprocessing...")
    x,y = preprocess(x,selectAss) # rajoute les features
    tot=pd.concat([tot,faire(x,y,preprocessFINAL(x,selectAss))])
    ######################################################################TEST DE ROBIN
   
c=robin(x, y)


xTrain,yTrain = preprocess(x,'CAT')
print("--- %s seconds ---" % (time.time() - start_time))
x.columns.values



###########################################################

print("--- %s seconds ---" % (time.time() - start_time))



