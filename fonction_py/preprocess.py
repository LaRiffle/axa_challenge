from numpy import *
import pandas as pd
import datetime
from datetime import timedelta

def sum_duplicated():
    fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
    x=pd.read_csv("data/train_2011_2012_2013.csv", sep=";", usecols=fields) # LECTURE
    pd.DataFrame(x.groupby(('ASS_ASSIGNMENT', 'DATE', 'WEEK_END', 'DAY_WE_DS'), squeeze =False).sum()).to_csv("data/trainPure.csv", sep=';', encoding='utf_8')

   
    
def preprocessTOTAL(selectAss):
    fields = ['DATE', 'DAY_OFF', 'WEEK_END', 'DAY_WE_DS', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ] # selectionne les colonnes à lire
    x=pd.read_csv("data/trainPure.csv", sep=";", usecols=fields) # LECTURE du fichier de train,
    
    
    #################################################" Pour X
    if(selectAss != False):#selection
        x = x[x['ASS_ASSIGNMENT'] == selectAss]
        del x['ASS_ASSIGNMENT']
    x['YEAR'] = x['DATE'].str[0:4]
    x['MONTH'] = x['DATE'].str[5:7]
    x['DAY'] = x['DATE'].str[8:10]
    x['HOUR'] = x['DATE'].str[-12:-8]
    x['DATE'] = pd.to_datetime(x['DAY']+'/'+x['MONTH']+'/'+x['YEAR'])    
    
    ##############pour avoir le call de 7jours avant en 's7'
    tmp = pd.DataFrame()
    tmp['HOUR'] = x['HOUR']
    tmp['DATE'] = x['DATE']- timedelta(days=7)
    #tmp.join(x[['DATE','HOUR', 'CSPL_RECEIVED_CALLS' ]], on=['DATE','HOUR'], how='left')
    tmp[['DATE','HOUR', 's7' ]]=pd.merge(tmp[['DATE','HOUR']],x[['DATE','HOUR', 'CSPL_RECEIVED_CALLS' ]], on=['HOUR', 'DATE'], how='left')
    x=pd.concat([x, tmp['s7']], axis=1)
    x['s7'][pd.isnull(x['s7'])]=x['CSPL_RECEIVED_CALLS'][pd.isnull(x['s7'])]
    
    file = ['joursFeries', 'vacances']
    for f in file:
        jf =pd.read_csv("data/"+f+".csv", sep=";")
        for n in list(jf):
            x[n]= x['DATE'].apply(lambda x: x.strftime('%d/%m/%Y')).isin(jf[n])
            
    #######################################################pour xTest     
    xTest=pd.read_csv("data/submission.txt", sep="\t") # LECTURE
    del xTest['prediction']
    souvenir = xTest.copy()
    if(selectAss != False):
        xTest = xTest[xTest['ASS_ASSIGNMENT'] == selectAss]
        souvenir = souvenir[souvenir['ASS_ASSIGNMENT'] == selectAss]
        del xTest['ASS_ASSIGNMENT']

    xTest['YEAR'] = xTest['DATE'].str[0:4]
    xTest['MONTH'] = xTest['DATE'].str[5:7]
    xTest['DAY'] = xTest['DATE'].str[8:10]
    xTest['HOUR'] = xTest['DATE'].str[-12:-8]
    xTest['DATE'] = pd.to_datetime(xTest['DAY']+'/'+xTest['MONTH']+'/'+xTest['YEAR'])
    
    tmp = pd.DataFrame()
    tmp['HOUR'] = xTest['HOUR']
    tmp['DATE'] = xTest['DATE']- timedelta(days=7)
    #tmp.join(x[['DATE','HOUR', 'CSPL_RECEIVED_CALLS' ]], on=['DATE','HOUR'], how='left')
    tmp=pd.merge(tmp,x[['DATE','HOUR', 'CSPL_RECEIVED_CALLS' ]], on=['HOUR', 'DATE'], how='left')
    tmp=tmp.rename(columns = {'CSPL_RECEIVED_CALLS':'s7'})  
    xTest['s7']=tmp['s7'].values
        
    xTest['tmp']=xTest['DATE'].dt.dayofweek # recupere le numero du jour de la semaine
    jour = pd.DataFrame(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    jour.columns = ['DAY_WE_DS']
    jour['tmp']=[0,1,2,3,4,5,6]
    xTest=pd.merge(xTest, jour) # attribue le nom du jour a chaque ligne
    xTest['WEEK_END'] = xTest['DAY_WE_DS'].isin(['Samedi', 'Dimanche']) # rajoute si c'est un week end


    file = ['joursFeries', 'vacances']
    for f in file:
        jf =pd.read_csv("data/"+f+".csv", sep=";")
        for n in list(jf):
            xTest[n]= xTest['DATE'].apply(lambda x: x.strftime('%d/%m/%Y')).isin(jf[n])

            
            
            
            
            
            
            
            
    y = x['CSPL_RECEIVED_CALLS']
    del x['CSPL_RECEIVED_CALLS']
    del x['DATE']
    x=pd.get_dummies(x)     
    del xTest['DATE']
    xTest=pd.get_dummies(xTest) # cree des colonnes pour chaque feature categoriel
    s=set(list(x))
    ss=set(list(xTest))
          
    for tmp in s.difference(ss): # supprime les features qui ne sont que dans x
        del x[tmp]
    for tmp in ss.difference(s): # supprime les features qui ne sont que dans xTest
        del xTest[tmp]
    
    xTest = xTest[list(x)] # reordonne les features pour qu'ils sont dans le meme ordre pour x et xTest
    
    return(xTest.fillna(0), x, souvenir, y)      