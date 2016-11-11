from numpy import *
import pandas as pd

def remove_duplicated():
    fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
    x=pd.read_csv("data/train_2011_2012_2013.csv", sep=";", usecols=fields) # LECTURE
    pd.DataFrame(x.groupby(('ASS_ASSIGNMENT', 'DATE', 'WEEK_END', 'DAY_WE_DS'), squeeze =False).sum())to_csv("data/trainPure.csv", sep=';')

    

def preprocess(x):
    # Ajoute les champs utiles et supprime ceux qui servent à rien
    #del x['DAY_WE_DS']
    #del x['TPER_TEAM']
    d = pd.DataFrame(x.groupby(('ASS_ASSIGNMENT', 'DATE', 'WEEK_END', 'DAY_WE_DS'), squeeze =False).sum())
    d.to_csv("tttt.csv", sep=';')
    sum((x[['DATE', 'ASS_ASSIGNMENT']]).duplicated()==False)
    x[pd.DataFrame.duplicated(x)].shape
    x[pd.DataFrame.duplicated(x)].shape
    x['YEAR'] = x['DATE'].str[0:4]
    x['MONTH'] = x['DATE'].str[5:7]
    x['DAY'] = x['DATE'].str[8:10]
    #x['HOUR'] = x['DATE'].str[-12:-10].astype(int)
   # x['HOUR'] = x['HOUR']+ ':'+((x['DATE'].str[-9:-8].astype(int)==3)*0.5).astype(str)
    x['HOUR'] = x['DATE'].str[-12:-8]
    del x['DATE']
    del x['WEEK_END']
    pd.DataFrame.duplicated(x)


    x=pd.get_dummies(x)
    return(x)
    
def preprocessFINAL(x):
    
    return(x)