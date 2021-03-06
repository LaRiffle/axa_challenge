from fonction_py.tools import *
from fonction_py.preprocess import *
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn import decomposition
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import uniform as sp_randint
from sklearn import datasets
from sklearn.linear_model import Ridge
from fonction_py.tim import *

import time

def faireTout():
    fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
    c = pd.DataFrame()
<<<<<<< HEAD
    listmodel = faireListModel()#recupere le nom et les modeles de chaque truc

    data=pd.read_csv("data/trainPure.csv", sep=";", usecols=fields) # LECTURE du fichier de train,
    resultat = pd.read_csv("data/submission.txt", sep="\t") # LECTURE dufichier de test
    res=[] 
    model = listmodel[0]
    for model in listmodel:
        print(model[0]) #affiche le ass assignment
        (xTest, x, souvenir, y)=preprocessTOTAL(model[0]) # ajuste le nombre et le nom de feature pour que xTest et x aient les memes
        mod= GradientBoostingRegressor(loss='huber', alpha=0.9,n_estimators=100, max_depth=3,learning_rate=.1, min_samples_leaf=9,min_samples_split=9)
        mod.fit(x, y) #s'entraine
        pred = mod.predict(xTest) # predit
        pred[pred>max(y)*1.05]=max(y)*1.05 # pour pas predire trop grand
        pred[pred<0]=0 # pas de negatif
        pred =np.round(pred).astype(int) # to int
        souvenir['prediction']=pred # on l'ajoute a souvenir qui garde le format standard et la date pour qu'on remette tout a la bonne place a la fin
        resultat=pd.merge(resultat, souvenir, how='left',on=['DATE', 'ASS_ASSIGNMENT']) # on remet chaque prediction à la bonne ligne -> il cree prediction_x et prediction_y car l'ancienne prediction et la nouvelle colonne de prediction
        resultat=resultat.fillna(0) # on remplit les endroits ou on a pas predit avec des 0
        resultat['prediction'] = resultat['prediction_x']+resultat['prediction_y'] # merge les deux colonnes
        del resultat['prediction_x']
        del resultat['prediction_y']
   
=======
    listmodel = faireListModel()
    #'Evenements',  'Gestion Amex'
    #setFields = set(pd.read_csv("data/fields.txt", sep=";")['0'].values)
#    resultat = pd.read_csv("data/submission.txt", sep="\t")
    
    i=0
#    res = []
    start_time = time.time()
    model = listmodel[24]
    data=pd.read_csv("data/trainPure.csv", sep=";", usecols=fields) # LECTURE
    
    resultat = pd.read_csv("data/submission.txt", sep="\t") # LECTURE
    res=[]
    for model in listmodel:
        i = i+1
        print(model[0])
        x,y = preprocess(data.copy(), model[0]) # rajoute les features
        model[1].fit(x, y)
        #model.score(xTrain, yTrain)
        (xTest, souvenir)=preprocessFINAL(x,model[0])
        pred = model[1].predict(xTest)
        pred[pred>max(y)*1.05]=max(y)*1.05
        pred[pred<0]=0
        pred =np.round(pred)
        souvenir['prediction']=int(pred)
        resultat=pd.merge(resultat, souvenir, how='left',on=['DATE', 'ASS_ASSIGNMENT'])
        resultat=resultat.fillna(0)
        resultat['prediction'] = resultat['prediction_x']+resultat['prediction_y']
        del resultat['prediction_x']
        del resultat['prediction_y']
    x,y = preprocess(data.copy(), 'Téléphonie') # rajoute les features
    #model.score(xTrain, yTrain)
    (xTest, souvenir)=preprocessFINAL(x,'Téléphonie')
    pred=telephoniePred(x,y,xTest)
    pred[pred>max(y)*1.05]=max(y)*1.05
    pred[pred<0]=0
    pred =np.round(pred)
    souvenir['prediction']=int(pred)
    resultat=pd.merge(resultat, souvenir, how='left',on=['DATE', 'ASS_ASSIGNMENT'])
    resultat=resultat.fillna(0)
    resultat['prediction'] = resultat['prediction_x']+resultat['prediction_y']
    del resultat['prediction_x']
    del resultat['prediction_y']
<<<<<<< HEAD
    pd.DataFrame(res).to_csv("reslist.csv", sep=";", decimal=",")
    resultat.to_csv("vraipred.txt", sep="\t", index =False)    
=======
>>>>>>> origin/master
    resultat['prediction']=resultat['prediction'].astype(int)
    resultat.to_csv("pouranalyse.txt", sep="\t", index =False, encoding='utf-8')  
    
>>>>>>> origin/master
    return resultat
    
    
def faireListModel():
    return [('CAT',  linear_model.LinearRegression()), 
    ('CMS', RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=5,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Crises',linear_model.LinearRegression()),
    ('Domicile', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=90, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Gestion',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Gestion - Accueil Telephonique',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=20,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=70, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Gestion Assurances',RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=20,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=20, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Gestion Clients', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
           max_features=90, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=50, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Gestion DZ', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Gestion Relation Clienteles',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
           max_features=90, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=110, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Gestion Renault', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features=50, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Japon',RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=10,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Manager',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Mécanicien',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=20,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Médical',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Nuit', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=20,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Prestataires',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=20,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('RENAULT',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=80,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('RTC',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=20,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Regulation Medicale',linear_model.LinearRegression()), 
    ('SAP',RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=20,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Services',RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=30,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Tech. Axa',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=20,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)), 
    ('Tech. Inter',RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=30,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Tech. Total',RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=70,
           max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)),
    ('Téléphonie',GradientBoostingRegressor(loss='huber', alpha=0.9,n_estimators=100, max_depth=3,learning_rate=.1, min_samples_leaf=9,min_samples_split=9) )]