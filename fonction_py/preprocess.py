from numpy import *
import pandas as pd



def preprocess(x):
    # Ajoute les champs utiles et supprime ceux qui servent à rien
    del x['DAY_WE_DS']
    del x['TPER_TEAM']
    x['YEAR'] = x['DATE'].str[0:4]
    x['MONTH'] = x['DATE'].str[5:7]
    x['DAY'] = x['DATE'].str[8:10]
    x['HOUR'] = x['DATE'].str[-12:-10].astype(int)
    x['HOUR'] = x['HOUR']+ (x['DATE'].str[-9:-8].astype(int)==3)*0.5
    del x['DATE']

    x=pd.get_dummies(x)
    # Transformation en champs numériques

    
    # Transformation en champs numériques TODO
    # Normalise (mean 0, std 1) les champs our elsquels cela a un sens
    #X = x['HOUR']
    #m = mean(X,axis=0)
    #s = std(X,axis=0)
    #X = (X - m) / s
    #x['HOUR'] = X
    
    # Ajoute des champs d'ordre deux pour une régression polynomiale TODO
    
    from fonction_py.tools import poly_exp
    #x = poly_exp(x,2)
    #x = column_stack([ones(len(x)), x])
    
    return(x)