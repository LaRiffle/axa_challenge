from numpy import *
import pandas as pd



def preprocess(x):
    # Ajoute les champs utiles et supprime ceux qui servent à rien
    #del x['DAY_WE_DS']
    #del x['TPER_TEAM']
    x['YEAR'] = x['DATE'].str[0:4]
    x['MONTH'] = x['DATE'].str[5:7]
    x['DAY'] = x['DATE'].str[8:10]
    #x['HOUR'] = x['DATE'].str[-12:-10].astype(int)
   # x['HOUR'] = x['HOUR']+ ':'+((x['DATE'].str[-9:-8].astype(int)==3)*0.5).astype(str)
    x['HOUR'] = x['DATE'].str[-12:-8]
    del x['DATE']
    del x['WEEK_END']
    x=pd.get_dummies(x)
    return(x)
    
def preprocessFINAL(x):
    
    return(x)