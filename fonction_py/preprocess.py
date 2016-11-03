from numpy import *

def preprocess(x):
    # Ajoute les champs utiles et supprime ceux qui servent à rien
    del x['DAY_WE_DS']
    del x['TPER_TEAM']
    x['HOUR'] = x['DATE'].str[-12:-10].astype(int)
    
    # Transformation en champs numériques
    
    # Normalise (mean 0, std 1)
    X = x['HOUR']
    m = mean(X,axis=0)
    s = std(X,axis=0)
    X = (X - m) / s
    x['HOUR'] = X
    
    # Ajoute des champs d'ordre deux pour une régression polynomiale
    from fonction_py.tools import poly_exp
    #x = poly_exp(x,2)
    #x = column_stack([ones(len(x)), x])
    
    return(x)