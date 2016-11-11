from numpy import *
import pandas as pd
import datetime

def sum_duplicated():
    fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
    x=pd.read_csv("data/train_2011_2012_2013.csv", sep=";", usecols=fields) # LECTURE
    pd.DataFrame(x.groupby(('ASS_ASSIGNMENT', 'DATE', 'WEEK_END', 'DAY_WE_DS'), squeeze =False).sum()).to_csv("data/trainPure.csv", sep=';', encoding='utf_8')

    

def preprocess(x, selectAss):
    # Ajoute les champs utiles et supprime ceux qui servent à rien
    #del x['DAY_WE_DS']
    #del x['TPER_TEAM']
    x['YEAR'] = x['DATE'].str[0:4]
    x['MONTH'] = x['DATE'].str[5:7]
    x['DAY'] = x['DATE'].str[8:10]
    #x['HOUR'] = x['DATE'].str[-12:-10].astype(int)
   # x['HOUR'] = x['HOUR']+ ':'+((x['DATE'].str[-9:-8].astype(int)==3)*0.5).astype(str)
    x['HOUR'] = x['DATE'].str[-12:-8]
    x['DATE'] = x['DAY']+'/'+x['MONTH']+'/'+x['YEAR']
    
    file = ['joursFeries', 'vacances']
    for f in file:
        jf =pd.read_csv("data/"+f+".csv", sep=";")
        for n in list(jf):
            x[n]= x['DATE'].isin(jf[n])
    if(selectAss != False):
        x = x[x['ASS_ASSIGNMENT'] == selectAss]
    y = x['CSPL_RECEIVED_CALLS']
    del x['CSPL_RECEIVED_CALLS']
    del x['DATE']
    x=pd.get_dummies(x)
    return(x, y)
    
def preprocessFINAL(x, selectAss):
    xTest=pd.read_csv("data/submission.txt", sep="\t") # LECTURE
    del xTest['prediction']
    xTest['YEAR'] = xTest['DATE'].str[0:4]
    xTest['MONTH'] = xTest['DATE'].str[5:7]
    xTest['DAY'] = xTest['DATE'].str[8:10]
    xTest['HOUR'] = xTest['DATE'].str[-12:-8]
    xTest['DATE'] = xTest['DAY']+'/'+xTest['MONTH']+'/'+xTest['YEAR']
    
    file = ['joursFeries', 'vacances']
    for f in file:
        jf =pd.read_csv("data/"+f+".csv", sep=";")
        for n in list(jf):
            xTest[n]= xTest['DATE'].isin(jf[n])
    if(selectAss != False):
        xTest = xTest[xTest['ASS_ASSIGNMENT'] == selectAss]
    xTest['tmp']=pd.to_datetime(xTest['DATE']).dt.dayofweek
    jour = pd.DataFrame(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    jour.columns = ['DAY_WE_DS']
    jour['tmp']=[1,2,3,4,5,6,7]
    xTest=pd.merge(jour,xTest)
    xTest['WEEK_END'] = xTest['DAY_WE_DS'].isin(['Samedi', 'Dimanche'])
    del xTest['DATE']
    xTest=pd.get_dummies(xTest)
    s=set(list(x))
    ss=set(list(xTest))        
    for tmp in s.difference(ss):
        xTest[tmp]=0
    return(xTest)
    
    
def print_sub(xTest, yTest):
    xTest