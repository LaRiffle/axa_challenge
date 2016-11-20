import pandas as pd
from fonction_py.preprocess import *
from fonction_py.train import *
from fonction_py.tools import *
from fonction_py.robin import *
from fonction_py.tim import *
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
selectAss = 'Téléphonie' # quel type de ASS_ASSIGNMENT on travaille
c = pd.DataFrame()
listass= ['CAT', 'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique','Gestion Assurances', 'Gestion Clients', 'Gestion DZ', 'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Manager', 'Mécanicien', 'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC', 'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'Téléphonie']
#'Evenements',  'Gestion Amex'
#setFields = set(pd.read_csv("data/fields.txt", sep=";")['0'].values)
resultat = pd.read_csv("data/submission.txt", sep="\t")
resultat['fait'] = False
i=0
res = []
start_time = time.time()
for selectAss in listass:
    i = i+1
    print(selectAss+' ' +str(np.round(i*100/len(listass))))
    
    x=pd.read_csv("data/trainPure.csv", sep=";", usecols=fields) # LECTURE
<<<<<<< HEAD
    #x,y = preprocess(x,selectAss) # rajoute les features
    x,y = preprocessTel(x)
    res.append(robinTel(x,y))
=======
    x,y = preprocess(x,selectAss) # rajoute les features
    #res.append(robin(x,y))
    res.append(tim(x,y))
>>>>>>> origin/master
    
    
print("--- %s seconds ---" % str((time.time() - start_time)))
res = pd.DataFrame(res, index=listass)
#res.columns = listModel
res.to_csv("restestTel.csv", sep=";", decimal=",")
####################################################### ecriture final
#    xTest, souvenir = preprocessFINAL(x,selectAss)
#    souvenir['prediction']= faire(x,y,xTest)
#    ######################################################################TEST DE ROBIN
#    resultat=pd.merge(resultat, souvenir, how='left',on=['DATE', 'ASS_ASSIGNMENT'])
#    resultat['fait'] = ~pd.isnull(resultat['prediction_y']) | resultat['fait']
#    resultat=resultat.fillna(0)
#    resultat['prediction'] = resultat['prediction_x']+resultat['prediction_y']
#    del resultat['prediction_x']
#    del resultat['prediction_y']

#del resultat['fait']
#resultat.to_csv("vraipred.txt", sep="\t", index =False)
###########################################################

print("--- %s seconds ---" % (time.time() - start_time))



